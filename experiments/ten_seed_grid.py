"""Ten seeds for the headline grid, because five was named as a limitation.

    python experiments/ten_seed_grid.py --jobs 8

`docs/limitations.md` says of the headline table: "Five seeds. Enough to report
a t interval across seeds instead of an anecdote, not enough to resolve a small
difference." Every interesting comparison in this repository has since turned
out to be between arms whose seed-to-seed spread is *bimodal* -- a run either
finds the behaviour or collapses -- and five samples from a bimodal distribution
is exactly the case where a mean and a t interval mislead. So both halves of the
grid go to ten.

**Nothing else changes.** Same 200 000 steps, same per-level entropy floors from
`experiments/results/floor_by_level.json`, same width, same demonstrations, and
-- the part that is easy to get wrong -- the same update-to-data ratio of 1.0
that seeds 0-4 were trained at, rather than the 0.5 that is now the default.
Half a grid on one learning rule and half on another would make the extra seeds
worse than useless.

`sacfloor_medium_s5-9` is trained by `experiments/shapes_seeds.py`, which needs
the same runs for the shape comparison; this driver reuses them if they exist.
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

DEMOS = os.path.join("demonstrations", "expert_low.npz")
# From experiments/results/floor_by_level.json; the matrix that showed one value
# does not work everywhere.
FLOORS = {"none": 0.15, "low": 0.05, "medium": 0.15, "high": 0.05}
UPDATES_PER_STEP = "1.0"
HIDDEN = "128"


def job_sacfloor(level: str, seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "sacfloor_{}_s{}".format(level, seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", level,
                "--hidden", HIDDEN, "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--updates-per-step", UPDATES_PER_STEP,
                "--alpha-floor", str(FLOORS[level]), "--output", out],
    }


def job_bc(seed: int) -> Dict:
    out = os.path.join(RUNS, "bc_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_il.py", "--demos", DEMOS,
                "--seed", str(seed), "--epochs", "60", "--randomisation", "low",
                "--hidden", HIDDEN, "--eval-episodes", "50", "--quiet",
                "--output", out],
    }


def job_bcrl(level: str, seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "bcrl_{}_s{}".format(level, seed))
    init = os.path.join(RUNS, "bc_s{}".format(seed), "policy.pt")
    return {
        "name": os.path.basename(out), "output": out, "needs": init,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", level,
                "--hidden", HIDDEN, "--eval-every", "10000",
                "--eval-episodes", "30", "--quiet",
                "--updates-per-step", UPDATES_PER_STEP,
                "--demos", DEMOS, "--demo-fraction", "0.25",
                "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
                "--critic-warmup", "3000", "--target-entropy-scale", "2.0",
                "--init-alpha", "0.02", "--init-actor", init, "--output", out],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--levels", nargs="+",
                        default=["none", "low", "medium", "high"])
    parser.add_argument("--bcrl-levels", nargs="+", default=["medium"])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shifted"])
    args = parser.parse_args()
    os.chdir(REPO)

    # Cloning first: the demonstration-seeded runs cannot start without it, and
    # it costs eleven seconds a seed.
    run_batch([job_bc(s) for s in args.seeds], args.jobs)

    jobs: List[Dict] = [job_sacfloor(lvl, s, args.steps)
                        for lvl in args.levels for s in args.seeds]
    jobs += [job_bcrl(lvl, s, args.steps)
             for lvl in args.bcrl_levels for s in args.seeds]
    run_batch(jobs, args.jobs)

    results = {}
    groups = [("sacfloor_" + lvl, "experiments/runs/sacfloor_{}_s*".format(lvl))
              for lvl in args.levels]
    groups += [("bcrl_" + lvl, "experiments/runs/bcrl_{}_s*".format(lvl))
               for lvl in args.bcrl_levels]
    groups += [("bc", "experiments/runs/bc_s*")]
    for label, pattern in groups:
        out = os.path.join("experiments", "results",
                           "ten_seed_{}.json".format(label))
        print("evaluating " + label, flush=True)
        subprocess.run([sys.executable, "src/evaluate.py", "--runs", pattern,
                        "--eval-levels", *args.eval_levels, "--episodes",
                        str(args.episodes), "--label", label, "--output", out],
                       cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "ten_seed_grid.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"seeds": "0-9", "steps": args.steps, "floors": FLOORS,
                   "updates_per_step": float(UPDATES_PER_STEP),
                   "note": "seeds 5-9 trained here at the update-to-data ratio "
                           "seeds 0-4 used (1.0), not the current default 0.5",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
