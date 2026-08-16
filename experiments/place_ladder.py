"""Which half of pick-and-place is the one from-scratch RL cannot do.

    python experiments/place_ladder.py --jobs 5

Four reward designs and a tripled budget have not got from-scratch SAC above
0.002 on the place task, while a clone of the same expert scores 0.978. At that
point "the reward is wrong" has been tested enough; what has *not* been tested is
which part of the task is hard. There are two candidates and they are separable:

* **transport** -- the object has to travel across the table, and the term that
  pays for travel is orthogonal to the term that pays for lifting, so the policy
  must go up (small reward) before it goes sideways (most of the reward);
* **release** -- the object has to be let go of, which is the exact behaviour
  the lift task spends nine terms teaching a policy *not* to do.

The travel distance is the knob that separates them. At a travel of nearly zero
the target sits where the object already is, so the task is "pick it up and put
it back down" with the transport removed and the release intact. If that works,
the obstacle is transport and a curriculum is the answer. If it does not, the
obstacle is release and no amount of curriculum over distance will help.

Three rungs, five seeds each, everything else identical to the runs that failed
-- same 200 000 steps, same entropy floor, same width, same reward:

    none    0.00-0.04 m   the target is where the object started
    short   0.06-0.10 m   one hand-width
    full    0.12-0.30 m   the real task, and the one that scores 0.002

The scripted expert clears all three (20/20, 18/20, 19/20 at `medium`), so any
rung that fails is failing for the learner rather than because it is impossible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import PLACE_TRAVEL_LADDER  # noqa: E402
from experiments.place_task import (  # noqa: E402
    ALPHA_FLOOR,
    HIDDEN,
    LEVEL,
    REPO,
    RUNS,
    run_batch,
)


def job(rung: str, seed: int, steps: int) -> Dict:
    lo, hi = PLACE_TRAVEL_LADDER[rung]
    out = os.path.join(RUNS, "place_rung{}_s{}".format(rung, seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL, "--task", "place",
                "--place-travel", str(lo), str(hi),
                "--hidden", str(HIDDEN), "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--alpha-floor", str(ALPHA_FLOOR), "--output", out],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", nargs="+", default=["none", "short", "full"],
                        choices=list(PLACE_TRAVEL_LADDER))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    os.chdir(REPO)

    jobs: List[Dict] = [job(r, s, args.steps)
                        for r in args.rungs for s in args.seeds]
    run_batch(jobs, args.jobs)

    results = {}
    for rung in args.rungs:
        label = "place_rung" + rung
        out = os.path.join("experiments", "results", label + "_eval.json")
        # Each rung is evaluated on *its own* travel distribution. Evaluating the
        # short rungs on the full one would measure generalisation, which is a
        # different question and would hide the answer to this one.
        lo, hi = PLACE_TRAVEL_LADDER[rung]
        print("evaluating " + label, flush=True)
        subprocess.run([sys.executable, "src/evaluate.py", "--runs",
                        "experiments/runs/{}_s*".format(label), "--task", "place",
                        "--place-travel", str(lo), str(hi),
                        "--eval-levels", "none", "--episodes", str(args.episodes),
                        "--label", label, "--output", out], cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = {"travel": [lo, hi], "eval": json.load(fh)}

    summary = os.path.join("experiments", "results", "place_ladder.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": args.steps,
                   "ladder": PLACE_TRAVEL_LADDER,
                   "note": "each rung is evaluated on the travel range it "
                           "trained on. Scoring the short rungs on the full "
                           "range would measure generalisation, which is a "
                           "different question and would hide the answer to "
                           "this one",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
