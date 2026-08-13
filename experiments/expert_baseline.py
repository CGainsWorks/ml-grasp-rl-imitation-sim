"""The scripted expert's success rate on every randomisation level.

    python experiments/expert_baseline.py --episodes 100

This is the reference line for everything else, and it is also the environment's
health check: if the expert stops succeeding on the nominal world, the scene has
been broken, not the learning.

The expert is deterministic given a world, so the interval here is the binomial
one (Wilson) over episodes. There are no seeds to average: there is only one
expert.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.randomisation.domain_rand import LEVELS  # noqa: E402
from src.utils.rollout import evaluate_policy  # noqa: E402
from src.utils.stats import wilson_interval  # noqa: E402

RESULTS = os.path.join("experiments", "results")
FINAL_SEED_BLOCK = 900_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default=os.path.join(RESULTS, "expert_baseline.json"))
    args = parser.parse_args()

    rows = {}
    for level in args.levels:
        env = make_env(level, seed=1234)
        expert = ScriptedExpert()
        result = evaluate_policy(
            env, expert.act, n_episodes=args.episodes, seed=FINAL_SEED_BLOCK,
            on_episode_start=expert.reset,
        )
        env.close()
        interval = wilson_interval(result["successes"], result["n_episodes"])
        rows[level] = {
            "success_rate": result["success_rate"],
            "wilson_low": interval.low,
            "wilson_high": interval.high,
            "grasp_rate": result["grasp_rate"],
            "mean_return": result["mean_return"],
            "mean_max_lift": result["mean_max_lift"],
            "episodes": result["n_episodes"],
        }
        print("{:<9s} success {:.3f}  95% Wilson [{:.3f}, {:.3f}]  grasp {:.3f}".format(
            level, result["success_rate"], interval.low, interval.high,
            result["grasp_rate"]), flush=True)

    blob = {"episodes": args.episodes, "seed_block": FINAL_SEED_BLOCK, "levels": rows,
            "policy": "src/policies/scripted_expert.py"}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
