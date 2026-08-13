"""The headline table: every method, evaluated the same way.

    python experiments/summarise.py --episodes 100

Methods compared, all on the same evaluation episodes:

    expert        the scripted state machine that produced the demonstrations
    bc            behaviour cloning on 200 demonstrations
    dagger        behaviour cloning plus five DAgger rounds
    sac_<level>   SAC from scratch at each randomisation level
    bcrl_<level>  BC initialisation plus demonstrations pinned in the replay
                  buffer, then SAC, at each randomisation level

Evaluated on ``none`` (the clean nominal world), ``medium`` (a training-like
distribution) and ``shifted`` (held out). Every cell is the mean over seeds with
a 95% t interval.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate_expert, evaluate_run  # noqa: E402
from src.utils.stats import summarise_seeds, welch_t  # noqa: E402

RUNS = os.path.join("experiments", "runs")
RESULTS = os.path.join("experiments", "results")

METHODS = [
    ("bc", "bc_s*", "behaviour cloning"),
    ("dagger", "dagger_s*", "behaviour cloning + DAgger"),
    ("sac_none", "sac_none_s*", "SAC, no randomisation"),
    ("sac_low", "sac_low_s*", "SAC, low randomisation"),
    ("sac_medium", "sac_medium_s*", "SAC, medium randomisation"),
    ("sac_high", "sac_high_s*", "SAC, wide randomisation"),
    ("bcrl_none", "bcrl_none_s*", "BC + SAC, no randomisation"),
    ("bcrl_low", "bcrl_low_s*", "BC + SAC, low randomisation"),
    ("bcrl_medium", "bcrl_medium_s*", "BC + SAC, medium randomisation"),
    ("bcrl_high", "bcrl_high_s*", "BC + SAC, wide randomisation"),
]


def summarise_method(pattern: str, levels: List[str], episodes: int, checkpoint: str) -> Dict:
    runs = sorted(glob.glob(os.path.join(RUNS, pattern)))
    runs = [r for r in runs if os.path.exists(os.path.join(r, checkpoint))]
    if not runs:
        return {}
    per_run = [evaluate_run(r, levels, episodes, checkpoint, 100) for r in runs]
    out = {"n_seeds": len(runs), "runs": [r["run"] for r in per_run], "levels": {}}
    for level in levels:
        out["levels"][level] = summarise_seeds(
            [r["levels"][level]["successes"] for r in per_run],
            [r["levels"][level]["episodes"] for r in per_run],
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["none", "medium", "shifted"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", default="policy.pt")
    parser.add_argument("--output", default=os.path.join(RESULTS, "summary.json"))
    args = parser.parse_args()

    table: Dict[str, Dict] = {}
    for key, pattern, label in METHODS:
        result = summarise_method(pattern, args.levels, args.episodes, args.checkpoint)
        if result:
            result["label"] = label
            table[key] = result

    expert = evaluate_expert(args.levels, args.episodes, 100)

    header = "{:<14s} {:>6s}".format("method", "seeds") + "".join(
        "  {:>22s}".format(level) for level in args.levels)
    print(header)
    print("-" * len(header))
    print("{:<14s} {:>6s}".format("scripted expert", "-") + "".join(
        "  {:>22s}".format("{:.3f}".format(expert["levels"][level]["success_rate"]))
        for level in args.levels))
    for key, _, label in METHODS:
        if key not in table:
            continue
        row = table[key]
        cells = []
        for level in args.levels:
            interval = row["levels"][level]["across_seeds"]
            cells.append("  {:>22s}".format("{:.3f} [{:.3f}, {:.3f}]".format(
                interval["point"], interval["low"], interval["high"])))
        print("{:<14s} {:>6d}".format(key, row["n_seeds"]) + "".join(cells))

    comparisons = {}
    if "sac_medium" in table and "bcrl_medium" in table:
        for level in args.levels:
            comparisons["bcrl_vs_sac_" + level] = welch_t(
                table["bcrl_medium"]["levels"][level]["per_seed_rates"],
                table["sac_medium"]["levels"][level]["per_seed_rates"],
            )
    if "bc" in table and "sac_high" in table:
        comparisons["sac_high_vs_bc_shifted"] = welch_t(
            table["sac_high"]["levels"].get("shifted", {}).get("per_seed_rates", []),
            table["bc"]["levels"].get("shifted", {}).get("per_seed_rates", []),
        )

    headline = {
        key: {
            level: table[key]["levels"][level]["across_seeds"] for level in args.levels
        } for key in table
    }
    headline["expert"] = {
        level: {"point": expert["levels"][level]["success_rate"]} for level in args.levels
    }

    blob = {
        "episodes_per_seed": args.episodes,
        "checkpoint": args.checkpoint,
        "levels": args.levels,
        "expert": expert,
        "methods": table,
        "comparisons": comparisons,
        "headline": headline,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("\nwrote " + args.output)


if __name__ == "__main__":
    main()
