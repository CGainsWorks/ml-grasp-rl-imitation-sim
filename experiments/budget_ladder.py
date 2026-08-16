"""What another 400 000 steps actually buys the randomised from-scratch runs.

    python experiments/budget_ladder.py --jobs 4

`docs/limitations.md` says of the 200 000-step budget: "It is not enough for SAC
from scratch under randomisation, and the results say so rather than quietly
extending the budget for the conditions that needed it." That was the right call
at the time -- extending the budget only for the conditions that were losing is
how a grid gets rigged -- but it leaves the claim itself unmeasured. "Not
enough" could mean the runs were still climbing, or it could mean they had
converged to something poor.

This measures it, at 600 000 steps, three seeds, on the two randomised levels
that fail: `medium` (0.593 at 200k) and `high` (0.397). Everything else is held
at the ten-seed grid's settings, including the update-to-data ratio of 1.0 those
runs used rather than the current default, so the only difference is the budget.

The comparison this supports is *within* a level -- 200k against 600k at the
same randomisation -- not across levels, and it does not replace any headline
number. The headline grid stays at a matched 200 000 everywhere, because a table
where each cell got as much compute as it needed to look good is not a table.

Three outcomes are worth distinguishing and the seeds will say which:

* still climbing -- the budget claim was right and the ceiling is unknown;
* converged low -- the budget was never the binding constraint, and the entropy
  floor's per-level tuning or the randomisation width is;
* bimodal -- more seeds find the behaviour but the ones that collapse stay
  collapsed, which is what every other widening in this repository has done.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import REPO, RUNS, run_batch  # noqa: E402
from experiments.ten_seed_grid import FLOORS, HIDDEN, UPDATES_PER_STEP  # noqa: E402


def job(level: str, seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "budget_{}_s{}".format(level, seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", level,
                "--hidden", HIDDEN, "--eval-every", "50000",
                "--eval-episodes", "30", "--quiet",
                "--updates-per-step", UPDATES_PER_STEP,
                "--alpha-floor", str(FLOORS[level]), "--output", out],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["medium", "high"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    os.chdir(REPO)

    jobs: List[Dict] = [job(lvl, s, args.steps)
                        for lvl in args.levels for s in args.seeds]
    run_batch(jobs, args.jobs)

    results = {}
    for level in args.levels:
        label = "budget_" + level
        out = os.path.join("experiments", "results", label + "_eval.json")
        print("evaluating " + label, flush=True)
        subprocess.run([sys.executable, "src/evaluate.py", "--runs",
                        "experiments/runs/{}_s*".format(label),
                        "--eval-levels", "none", "shifted", "--episodes",
                        str(args.episodes), "--label", label, "--output", out],
                       cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "budget_ladder.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"steps": args.steps, "seeds": args.seeds, "floors": FLOORS,
                   "baseline": "the ten-seed grid at 200 000 steps, same "
                               "settings and the same update-to-data ratio",
                   "note": "this does not replace any headline number. The "
                           "grid stays at a matched 200 000 everywhere, because "
                           "a table where each cell got as much compute as it "
                           "needed to look good is not a table",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
