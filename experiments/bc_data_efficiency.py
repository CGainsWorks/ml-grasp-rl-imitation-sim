"""How many demonstrations does behaviour cloning need?

    python experiments/bc_data_efficiency.py --seeds 0 1 2 --episodes 50

Trains a clone on the first N demonstration episodes for several N, three seeds
each, and evaluates all of them the same way. This is the figure that shows
compounding error for what it is: with few demonstrations the clone's action
error is small and its success rate is still poor, because the errors it does
make take it somewhere the expert never went.

Runs in a couple of minutes; the training itself is seconds per point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_il import build_parser as il_parser  # noqa: E402
from src.train_il import train as il_train  # noqa: E402
from src.utils.stats import t_interval  # noqa: E402

RESULTS = os.path.join("experiments", "results")
SCRATCH = os.path.join("experiments", "runs", "_bc_sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", default=os.path.join("demonstrations", "expert_low.npz"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 25, 50, 100, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-level", default="low")
    parser.add_argument("--output", default=os.path.join(RESULTS, "bc_data_efficiency.json"))
    args = parser.parse_args()

    rows: List[Dict] = []
    for size in args.sizes:
        rates, mses = [], []
        for seed in args.seeds:
            il_args = il_parser().parse_args([])
            il_args.demos = args.demos
            il_args.seed = seed
            il_args.epochs = args.epochs
            il_args.hidden = 128
            il_args.randomisation = args.eval_level
            il_args.eval_episodes = args.episodes
            il_args.max_demo_episodes = size
            il_args.quiet = True
            il_args.output = os.path.join(SCRATCH, "n{}_s{}".format(size, seed))
            summary = il_train(il_args)
            rates.append(summary["final_success_rate"])
            mses.append(summary["final_val_mse"])
        interval = t_interval(rates)
        rows.append({
            "demo_episodes": size,
            "success": interval.as_dict(),
            "per_seed_success": rates,
            "mean_val_mse": float(sum(mses) / len(mses)),
        })
        print("{:>4d} demos  success {:.3f} [{:.3f}, {:.3f}]  val mse {:.5f}".format(
            size, interval.point, interval.low, interval.high,
            rows[-1]["mean_val_mse"]), flush=True)

    expert_success = None
    baseline = os.path.join(RESULTS, "expert_baseline.json")
    if os.path.exists(baseline):
        with open(baseline, "r", encoding="utf-8") as fh:
            expert_success = json.load(fh)["levels"].get(args.eval_level, {}).get("success_rate")

    blob = {
        "demos": args.demos,
        "eval_level": args.eval_level,
        "episodes_per_seed": args.episodes,
        "seeds": args.seeds,
        "expert_success": expert_success,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
