"""Why from-scratch RL scores zero on the place task, and what actually fixes it.

    python experiments/place_carry_gate.py --arms ramp bcrl_ramp --jobs 5

The first place design gave five from-scratch seeds 0.002 [0.000, 0.008] while
behaviour cloning from the same task scored 0.978. There are two candidate
explanations and they lead opposite ways, so both are run rather than argued.

**Budget.** Pick-and-place is a longer chain than lift-and-hold -- reach, grasp,
lift, carry, lower, release, settle, against reach, grasp, lift -- so 200 000
steps may simply be short. ``budget`` runs the unchanged reward at 600 000.

**Shaping.** Reading the runs says otherwise, and says it twice.

    ``none``   `carry` gated only on `grasped`. Both pads can be in contact with
               a box that is being *pushed*, so the policies pushed: grasp rate
               0.63-0.83, peak lift 0.010 m against a 4 cm latch, 0/20 episodes
               ever picking the box up. 0.002 across five seeds.

    ``latch``  `carry` multiplied by the binary lift latch. Sliding now pays
               nothing -- and five seeds still score 0.000, because the largest
               term in the reward is invisible until the box is 4 cm up. The
               policies grasp and sit on the table, peak lift 0.006-0.016 m.
               This is the lift task's missing-`hold`-term failure exactly, and
               it was walked into a second time.

    ``ramp_shallow``
               `carry` multiplied by clearance/4 cm, clipped at one. Sliding is
               still worth nothing; every millimetre of lift buys a share of the
               transport gradient. Cliff into hill, which is the fix
               docs/reward-design.md already records for the other task -- and
               five seeds still score 0.000 with a peak lift of 0.009 m,
               because a hill you cannot see the bottom of is still not a
               gradient you can follow. `clear` here is 3.0 x 0.06 = 0.18 a
               step.

    ``ramp``   the same, with `clear` at the lift task's own 4.0 x 0.12 = 0.48.
               Getting the box off the table is the *identical* sub-problem in
               both tasks and SAC solves it from scratch in the other one, so
               the weight is taken from there rather than searched for. 0.002.

    ``approach``
               the fifth design, and the first derived from a measurement rather
               than from a hypothesis about what went wrong. Decomposing what
               the scripted expert earns per step:

                   lift-and-hold    5.948/step positive, 51.8% of it from terms
                                    that only pay once the task is complete
                   pick-and-place   3.202/step positive, 80.7% from such terms

               The lift task's shaping is worth 2.87 a step by itself and rises
               continuously to the goal; `hold` alone pays 1.73. The place
               task's shaping was worth 0.62 and nothing in it paid *more* as
               the policy got closer to finishing. A reward that is 81% terminal
               is a sparse reward with decorations, which is what four failed
               designs had been saying. `approach` transplants the lift task's
               `hold`: a smooth bump for having the object over the target, off
               the table, still in the hand. It takes the terminal share to
               67.6% and the shaping to 1.24 a step.

Every arm is at the **original 200 000 steps** except ``budget``. Matched budget
on purpose: the mistake this repository has made before is reading a difference
between an intervention and a control that had different budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import (  # noqa: E402
    ALPHA_FLOOR,
    DEMOS,
    HIDDEN,
    LEVEL,
    REPO,
    RUNS,
    run_batch,
)

# The gate settings, and the weak-`clear` variant the first three were run with.
# `shallow` is the original 3.0 x 0.06 height term; the default is now the lift
# task's own 4.0 x 0.12 for the identical sub-task, so reproducing the first
# three arms needs it pinned.
SHALLOW = {"w_clear": 3.0, "clear_target": 0.06}
CONFIGS = {"none": os.path.join("src", "rewards", "configs", "place_ungated.json"),
           "latch": os.path.join("src", "rewards", "configs", "place_latch.json"),
           "ramp_shallow": os.path.join("src", "rewards", "configs",
                                        "place_ramp_shallow.json"),
           "ramp": os.path.join("src", "rewards", "configs",
                                "place_noapproach.json"),
           "approach": os.path.join("src", "rewards", "configs",
                                    "place_hover.json"),
           "guide": os.path.join("src", "rewards", "configs", "place_goal.json"),
           "both": None}

# run-name prefix -> (gate, demonstration-seeded, steps multiplier)
ARMS = {
    "budget": ("none", False, 3),
    "gated": ("latch", False, 1),
    "ramp_shallow": ("ramp_shallow", False, 1),
    "ramp": ("ramp", False, 1),
    "approach": ("approach", False, 1),
    "guide": ("guide", False, 1),
    "both": ("both", False, 1),
    "bcrl_both": ("both", True, 1),
}
PREFIX = {"budget": "place_sacbudget", "gated": "place_sacgated",
          "ramp_shallow": "place_sacramp", "ramp": "place_saclift",
          "approach": "place_sacapproach", "guide": "place_sacguide",
          "bcrl_guide": "place_bcrlguide", "both": "place_sacboth",
          "bcrl_both": "place_bcrlboth"}


def job(arm: str, seed: int, steps: int) -> Dict:
    gate, demos, multiplier = ARMS[arm]
    steps = steps * multiplier
    out = os.path.join(RUNS, "{}_s{}".format(PREFIX[arm], seed))
    cmd = [sys.executable, "src/train_rl.py", "--steps", str(steps),
           "--seed", str(seed), "--randomisation", LEVEL, "--task", "place",
           "--hidden", str(HIDDEN), "--eval-every", "25000",
           "--eval-episodes", "30", "--quiet", "--output", out]
    if CONFIGS[gate]:
        cmd += ["--reward-config", CONFIGS[gate]]
    if demos:
        # The demonstration-seeded settings from experiments/place_task.py,
        # unchanged, so the only difference from the run it is compared against
        # is the gate.
        init = os.path.join(RUNS, "place_bc_s{}".format(seed), "policy.pt")
        cmd += ["--demos", DEMOS, "--demo-fraction", "0.25",
                "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
                "--critic-warmup", "3000", "--target-entropy-scale", "2.0",
                "--init-alpha", "0.02", "--init-actor", init]
        return {"name": os.path.basename(out), "output": out, "needs": init,
                "cmd": cmd}
    cmd += ["--alpha-floor", str(ALPHA_FLOOR)]
    return {"name": os.path.basename(out), "output": out, "cmd": cmd}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shifted"])
    args = parser.parse_args()
    os.chdir(REPO)

    jobs: List[Dict] = []
    for arm in args.arms:
        seeds = args.seeds[:3] if arm == "budget" else args.seeds
        jobs += [job(arm, s, args.steps) for s in seeds]
    run_batch(jobs, args.jobs)

    results = {}
    for arm in args.arms:
        label = PREFIX[arm]
        out = os.path.join("experiments", "results", label + "_eval.json")
        cmd = [sys.executable, "src/evaluate.py", "--runs",
               "experiments/runs/{}_s*".format(label), "--task", "place",
               "--eval-levels", *args.eval_levels, "--episodes",
               str(args.episodes), "--label", label, "--output", out]
        print("evaluating " + label, flush=True)
        subprocess.run(cmd, cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "place_carry_gate.json")
    existing = {}
    if os.path.exists(summary):
        with open(summary, "r", encoding="utf-8") as fh:
            existing = json.load(fh).get("results", {})
    existing.update(results)
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": args.steps,
                   "arms": {a: ARMS[a] for a in ARMS},
                   "control": "place_sac (gate=none, 200k) from place_task.json",
                   "results": existing}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
