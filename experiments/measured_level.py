"""What do the policies score when the ranges come from measurements?

    python experiments/measured_level.py

`docs/randomisation-sources.md` checks this repository's hand-picked
randomisation ranges against published measurements and finds them optimistic on
latency and sensing, and missing orientation error entirely.
`src/randomisation/configs/measured.json` is those published numbers turned into
an evaluation distribution.

Three columns, because the interesting question is not only "how much worse" but
"because of what":

``shifted``          the existing held-out proxy, for continuity
``measured``         the sourced ranges
``measured_norot``   the same, with the orientation error removed
``measured_corr``    the same, with the error correlated in time (rho 0.9)

The fourth column is the other half of the realism question. Independent
per-step noise is the easy model and it flatters anything that filters, because
averaging kills it -- and the expert filters, and every clone of it inherits the
input-output behaviour of something that filtered. Correlating the error at the
same magnitude removes that free lunch.

The third column is the ablation. Orientation error is the axis this repository
never had, and the hand cannot rotate, so the plausible guess is that it costs
little -- the policy could not act on a better estimate anyway. That guess is
wrong, and the column shows by how much.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate_expert, evaluate_run  # noqa: E402
from src.utils.stats import summarise_seeds  # noqa: E402

RUNS = os.path.join("experiments", "runs")
LEVELS = ["shifted", "measured", "measured_norot", "measured_corr"]
METHODS = [
    ("bc", "bc_s*", "behaviour cloning"),
    ("bcrl_high", "bcrl_high_s*", "BC + SAC, wide randomisation"),
    ("bcrl_medium", "bcrl_medium_s*", "BC + SAC, medium randomisation"),
    ("sacfloor_high", "sacfloor_high_s*", "SAC + entropy floor, wide randomisation"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "measured_level.json"))
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    out = {"episodes_per_seed": args.episodes, "levels": LEVELS, "methods": {}}
    for key, pattern, label in METHODS:
        runs = sorted(glob.glob(os.path.join(RUNS, pattern)))
        runs = [r for r in runs if os.path.exists(os.path.join(r, "policy.pt"))]
        if not runs:
            continue
        per = [evaluate_run(r, LEVELS, args.episodes, "policy.pt", 100) for r in runs]
        out["methods"][key] = {
            "label": label,
            "n_seeds": len(per),
            "levels": {lv: summarise_seeds(
                [p["levels"][lv]["successes"] for p in per],
                [p["levels"][lv]["episodes"] for p in per]) for lv in LEVELS},
        }
        print("{:<16s} {}".format(key, {
            lv: round(out["methods"][key]["levels"][lv]["across_seeds"]["point"], 3)
            for lv in LEVELS}), flush=True)

    out["expert"] = evaluate_expert(LEVELS, args.episodes, 100)["levels"]
    print("expert           {}".format(
        {lv: round(out["expert"][lv]["success_rate"], 3) for lv in LEVELS}), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
