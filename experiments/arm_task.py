"""Learned policies through the six-jointed arm, which nothing here has had.

    python experiments/arm_task.py --jobs 5

`docs/limitations.md` records that the arm variant works -- the scripted expert
reaches 0.680 [0.583, 0.763] through six joints with zero table penetrations --
and then says nothing at all about *learned* performance there, because none was
measured. The arm section has been the largest unbacked claim in the repository:
"there is an arm" is not the same as "the method works with an arm".

The comparison that matters is against the weld, at everything else held equal:
same task, same observation, same action space, same budget, same seeds, same
demonstrations pipeline. What differs is that the hand is now the end of a
kinematic chain with joint limits, self-collision and an IK solver that can
fail, so the gap between the two columns is the cost of the abstraction the rest
of this repository trains under.

Two things about the arm make its numbers not directly comparable to the weld's
even so, and both are stated rather than hidden:

* the arm's own expert scores 0.680, not 1.000, so cloning starts from a worse
  teacher and a demonstration set that carries its failures;
* resets are IK-solved and occasionally give up, so the initial-state
  distribution is not identical to the weld's.

Runs land in `experiments/runs/arm_{bc,bcrl,sac}_s<seed>`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import REPO, RUNS, run_batch  # noqa: E402

DEMOS = os.path.join("demonstrations", "expert_arm_low.npz")
ALPHA_FLOOR = "0.15"
HIDDEN = "128"
LEVEL = "medium"


def job_bc(seed: int) -> Dict:
    out = os.path.join(RUNS, "arm_bc_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_il.py", "--demos", DEMOS,
                "--seed", str(seed), "--epochs", "60", "--randomisation", "low",
                "--arm", "--hidden", HIDDEN, "--eval-episodes", "50",
                "--quiet", "--output", out],
    }


def job_sac(seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "arm_sac_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL, "--arm",
                "--hidden", HIDDEN, "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--alpha-floor", ALPHA_FLOOR, "--output", out],
    }


def job_bcrl(seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "arm_bcrl_s{}".format(seed))
    init = os.path.join(RUNS, "arm_bc_s{}".format(seed), "policy.pt")
    return {
        "name": os.path.basename(out), "output": out, "needs": init,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL, "--arm",
                "--hidden", HIDDEN, "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--demos", DEMOS, "--demo-fraction", "0.25",
                "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
                "--critic-warmup", "3000", "--target-entropy-scale", "2.0",
                "--init-alpha", "0.02", "--init-actor", init, "--output", out],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shifted"])
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.exists(DEMOS):
        print("recording arm demonstrations", flush=True)
        subprocess.run(
            [sys.executable, "src/record_demos.py", "--episodes", "200",
             "--randomisation", "low", "--arm", "--output", DEMOS],
            cwd=REPO, check=True)

    run_batch([job_bc(s) for s in args.seeds], args.jobs)
    run_batch([job_sac(s, args.steps) for s in args.seeds]
              + [job_bcrl(s, args.steps) for s in args.seeds], args.jobs)

    results = {}
    for label in ("arm_sac", "arm_bcrl", "arm_bc"):
        out = os.path.join("experiments", "results", label + "_eval.json")
        cmd = [sys.executable, "src/evaluate.py", "--runs",
               "experiments/runs/{}_s*".format(label), "--arm",
               "--eval-levels", *args.eval_levels, "--episodes",
               str(args.episodes), "--label", label, "--output", out]
        if label == "arm_sac":
            cmd.append("--expert")
        print("evaluating " + label, flush=True)
        subprocess.run(cmd, cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "arm_task.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": args.steps, "seeds": args.seeds,
                   "note": "the arm's own scripted expert scores 0.680 on the "
                           "nominal world, not 1.000, so these runs clone a "
                           "worse teacher than the weld runs do",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
