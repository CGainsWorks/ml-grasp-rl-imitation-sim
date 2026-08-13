"""The randomisation ablation: does training with wider randomisation transfer better?

    python experiments/ablation.py --episodes 100

For each training level (``none``, ``low``, ``medium``, ``high``) every seed's
final policy is evaluated twice:

* on **its own training distribution**, which measures how much the extra
  variation cost it in raw performance;
* on the **shifted** worlds, which are outside the training ranges of every
  level except partially ``high``, and which stand in for a real robot.

The gap between the two columns is the interesting quantity. A level that
scores well on its own distribution and badly on the shifted one has learned
the simulator, not the task.

Success rates are reported as the mean over seeds with a 95% t interval. The
per-seed numbers are kept in the output so the spread is inspectable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate_run  # noqa: E402
from src.utils.stats import summarise_seeds, welch_t  # noqa: E402

RUNS = os.path.join("experiments", "runs")
RESULTS = os.path.join("experiments", "results")


def evaluate_level(level: str, episodes: int, checkpoint: str, prefix: str = "sac") -> Dict:
    runs = sorted(glob.glob(os.path.join(RUNS, "{}_{}_s*".format(prefix, level))))
    runs = [r for r in runs if os.path.exists(os.path.join(r, checkpoint))]
    if not runs:
        return {}
    per_run = [evaluate_run(r, [level, "shifted"], episodes, checkpoint, 100) for r in runs]
    own = summarise_seeds(
        [r["levels"][level]["successes"] for r in per_run],
        [r["levels"][level]["episodes"] for r in per_run],
    )
    shifted = summarise_seeds(
        [r["levels"]["shifted"]["successes"] for r in per_run],
        [r["levels"]["shifted"]["episodes"] for r in per_run],
    )
    return {
        "level": level,
        "n_seeds": len(runs),
        "runs": [r["run"] for r in per_run],
        "train_level": own,
        "shifted": shifted,
        "gap": own["across_seeds"]["point"] - shifted["across_seeds"]["point"],
        "per_run_detail": per_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["none", "low", "medium", "high"])
    parser.add_argument("--prefix", default="sac",
                        help="run-directory prefix: sac (from scratch) or bcrl (imitation seeded)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", default="policy.pt")
    parser.add_argument("--output", default=os.path.join(RESULTS, "ablation.json"))
    args = parser.parse_args()

    rows: List[Dict] = []
    for level in args.levels:
        row = evaluate_level(level, args.episodes, args.checkpoint, args.prefix)
        if row:
            rows.append(row)
            print("{:<7s} own {:.3f} [{:.3f}, {:.3f}]   shifted {:.3f} [{:.3f}, {:.3f}]   "
                  "gap {:+.3f}  ({} seeds)".format(
                      level,
                      row["train_level"]["across_seeds"]["point"],
                      row["train_level"]["across_seeds"]["low"],
                      row["train_level"]["across_seeds"]["high"],
                      row["shifted"]["across_seeds"]["point"],
                      row["shifted"]["across_seeds"]["low"],
                      row["shifted"]["across_seeds"]["high"],
                      row["gap"], row["n_seeds"]), flush=True)
        else:
            print("{:<7s} no finished runs".format(level))

    comparisons = {}
    by_level = {row["level"]: row for row in rows}
    if "none" in by_level and "high" in by_level:
        comparisons["shifted_high_vs_none"] = welch_t(
            by_level["high"]["shifted"]["per_seed_rates"],
            by_level["none"]["shifted"]["per_seed_rates"],
        )
    if "none" in by_level and "medium" in by_level:
        comparisons["shifted_medium_vs_none"] = welch_t(
            by_level["medium"]["shifted"]["per_seed_rates"],
            by_level["none"]["shifted"]["per_seed_rates"],
        )

    blob = {
        "prefix": args.prefix,
        "episodes": args.episodes,
        "checkpoint": args.checkpoint,
        "n_seeds": max((row["n_seeds"] for row in rows), default=0),
        "rows": rows,
        "comparisons": comparisons,
        "note": "own = the level the policy trained on; shifted = held-out worlds "
                "outside the training ranges. Intervals are 95% t across seeds.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
