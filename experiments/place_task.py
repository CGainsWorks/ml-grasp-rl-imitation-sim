"""The second task, end to end: demonstrations, cloning, RL, and RL from demos.

    python experiments/place_task.py --jobs 8

Why this exists
---------------
`docs/limitations.md` said, in three words, "One task." Every reward weight,
every claim about entropy collapse, every statement about what demonstrations
buy, came from lift-and-hold. A method validated once is an anecdote about that
one task, so this runs the same pipeline against pick-and-place --
`src/rewards/place_reward.py` explains what the second task was chosen to break.

**Deliberately the same protocol as the lift grid**, not a better one: 200 000
steps, five seeds, the same network width, the same entropy floor the level
matrix picked for `medium` (0.15), the same demonstration count and the same
BC coefficient. If the place numbers were produced with a tuned configuration
and the lift numbers were not, the comparison between them would measure the
tuning. Anything that turns out to need different settings here is itself a
finding and belongs in the docs.

The one thing that does differ is which expert is recorded, and that is forced:
`ScriptedPlaceExpert` has to put the object down.

Runs land in `experiments/runs/place_{sac,bc,bcrl}_s<seed>`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join("experiments", "runs")
LOGS = os.path.join("experiments", "logs")
DEMOS = os.path.join("demonstrations", "expert_place_low.npz")

# The floor `experiments/results/floor_by_level.json` selected for `medium` on
# the lift task. Carried over rather than re-searched, so that if it turns out
# to be wrong here that is a result about transferability of the floor, which is
# exactly what docs/limitations.md warns about.
ALPHA_FLOOR = 0.15
LEVEL = "medium"
HIDDEN = 128


def job_bc(seed: int) -> Dict:
    out = os.path.join(RUNS, "place_bc_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_il.py", "--demos", DEMOS,
                "--seed", str(seed), "--epochs", "60", "--randomisation", "low",
                "--task", "place", "--hidden", str(HIDDEN),
                "--eval-episodes", "50", "--quiet", "--output", out],
    }


def job_sac(seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "place_sac_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL,
                "--task", "place", "--hidden", str(HIDDEN),
                "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
                "--alpha-floor", str(ALPHA_FLOOR), "--output", out],
    }


def job_bcrl(seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "place_bcrl_s{}".format(seed))
    init = os.path.join(RUNS, "place_bc_s{}".format(seed), "policy.pt")
    return {
        "name": os.path.basename(out), "output": out, "needs": init,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL,
                "--task", "place", "--hidden", str(HIDDEN),
                "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
                "--demos", DEMOS, "--demo-fraction", "0.25",
                "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
                "--critic-warmup", "3000",
                "--target-entropy-scale", "2.0", "--init-alpha", "0.02",
                "--init-actor", init, "--output", out],
    }


def run_batch(jobs: List[Dict], parallel: int) -> None:
    os.makedirs(LOGS, exist_ok=True)
    queue = [j for j in jobs
             if not os.path.exists(os.path.join(j["output"], "result.json"))]
    print("{} to train, {} reused".format(len(queue), len(jobs) - len(queue)),
          flush=True)
    running: List[Dict] = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < parallel:
            spec = queue.pop(0)
            if spec.get("needs") and not os.path.exists(spec["needs"]):
                print("skip {}: missing {}".format(spec["name"], spec["needs"]),
                      flush=True)
                continue
            log = open(os.path.join(LOGS, spec["name"] + ".log"), "w",
                       encoding="utf-8")
            proc = subprocess.Popen(spec["cmd"], cwd=REPO, stdout=log,
                                    stderr=subprocess.STDOUT)
            running.append({"spec": spec, "proc": proc, "log": log})
            print("[{:>5.0f}s] start {}".format(time.time() - t0, spec["name"]),
                  flush=True)
        time.sleep(2.0)
        for entry in list(running):
            if entry["proc"].poll() is None:
                continue
            entry["log"].close()
            running.remove(entry)
            print("[{:>5.0f}s] done  {} (exit {})".format(
                time.time() - t0, entry["spec"]["name"], entry["proc"].returncode),
                flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shifted"])
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.exists(DEMOS):
        print("recording place demonstrations", flush=True)
        subprocess.run(
            [sys.executable, "src/record_demos.py", "--episodes", "200",
             "--randomisation", "low", "--task", "place", "--output", DEMOS],
            cwd=REPO, check=True)

    run_batch([job_bc(s) for s in args.seeds], args.jobs)
    run_batch([job_sac(s, args.steps) for s in args.seeds]
              + [job_bcrl(s, args.steps) for s in args.seeds], args.jobs)

    results = {}
    for label, pattern in (("place_sac", "experiments/runs/place_sac_s*"),
                           ("place_bcrl", "experiments/runs/place_bcrl_s*"),
                           ("place_bc", "experiments/runs/place_bc_s*")):
        out = os.path.join("experiments", "results", label + "_eval.json")
        cmd = [sys.executable, "src/evaluate.py", "--runs", pattern,
               "--task", "place", "--eval-levels", *args.eval_levels,
               "--episodes", str(args.episodes), "--label", label,
               "--output", out]
        if label == "place_sac":
            cmd.append("--expert")
        print("evaluating " + label, flush=True)
        subprocess.run(cmd, cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "place_task.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": args.steps, "seeds": args.seeds,
                   "alpha_floor": ALPHA_FLOOR, "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
