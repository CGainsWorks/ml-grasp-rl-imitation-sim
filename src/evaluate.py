"""Evaluate trained policies and report success rates with confidence intervals.

    python src/evaluate.py --runs experiments/runs/sac_medium_s0 experiments/runs/sac_medium_s1 \
        --eval-levels none medium shifted --episodes 100 \
        --output experiments/results/sac_medium.json

One run directory is one seed. The script evaluates every run on every named
evaluation level, then aggregates across seeds:

* the headline number is the mean across seeds with a 95% t interval, because
  the quantity of interest is what the *training procedure* produces, and the
  seed-to-seed spread on this task is several times the binomial spread within
  a seed;
* the pooled Wilson interval is reported next to it so the difference between
  the two is visible rather than hidden.

A single-seed number is printed with a warning attached. It is not a result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import ACT_DIM, OBS_DIM, make_env  # noqa: E402
from src.policies.networks import SquashedGaussianActor  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert
from src.policies.scripted_place_expert import ScriptedPlaceExpert  # noqa: E402
from src.utils.rollout import evaluate_policy, torch_policy  # noqa: E402
from src.utils.stats import summarise_seeds  # noqa: E402

FINAL_SEED_BLOCK = 900_000  # held apart from training resets and training-time evals


def load_actor(path: str) -> SquashedGaussianActor:
    """Load an actor from either a SAC checkpoint or a BC/DAgger checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    hidden = state.get("hidden")
    if hidden is None:
        cfg = state.get("config", {})
        hidden_cfg = cfg.get("hidden", (256, 256))
        hidden = int(hidden_cfg[0])
    actor = SquashedGaussianActor(
        state.get("obs_dim", OBS_DIM), state.get("act_dim", ACT_DIM), (hidden, hidden)
    )
    actor.load_state_dict(state["actor"])
    actor.eval()
    return actor


def evaluate_run(
    run_dir: str,
    eval_levels: List[str],
    episodes: int,
    checkpoint: str,
    max_steps: int,
    task: str = "lift",
    arm: bool = False,
    place_travel=None,
    handled: bool = False,
) -> Dict:
    path = os.path.join(run_dir, checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError("no checkpoint at " + path)
    actor = load_actor(path)

    config_path = os.path.join(run_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

    out: Dict[str, Dict] = {}
    for level in eval_levels:
        env = make_env(level, seed=1234, max_steps=max_steps, task=task, arm=arm,
                       travel_range=place_travel,
                       handled=handled, wrist=handled)
        result = evaluate_policy(
            env, torch_policy(actor, deterministic=True),
            n_episodes=episodes, seed=FINAL_SEED_BLOCK,
        )
        env.close()
        out[level] = {
            "successes": result["successes"],
            "episodes": result["n_episodes"],
            "success_rate": result["success_rate"],
            "grasp_rate": result["grasp_rate"],
            "mean_return": result["mean_return"],
            "mean_max_lift": result["mean_max_lift"],
        }
    return {"run": run_dir, "train_randomisation": config.get("randomisation"),
            "seed": config.get("seed"), "levels": out}


def evaluate_expert(eval_levels: List[str], episodes: int, max_steps: int,
                    task: str = "lift", arm: bool = False,
                    place_travel=None) -> Dict:
    """The scripted expert, on the same episodes, as a reference line."""
    out = {}
    expert_cls = ScriptedPlaceExpert if task == "place" else ScriptedExpert
    for level in eval_levels:
        env = make_env(level, seed=1234, max_steps=max_steps, task=task, arm=arm,
                       travel_range=place_travel)
        expert = expert_cls()

        def policy(obs, _expert=expert):
            return _expert.act(obs)

        result = evaluate_policy(
            env, policy, n_episodes=episodes, seed=FINAL_SEED_BLOCK,
            on_episode_start=expert.reset,
        )
        env.close()
        out[level] = {
            "successes": result["successes"],
            "episodes": result["n_episodes"],
            "success_rate": result["success_rate"],
            "grasp_rate": result["grasp_rate"],
            "mean_return": result["mean_return"],
            "mean_max_lift": result["mean_max_lift"],
        }
    return {"run": "scripted-expert", "train_randomisation": None, "seed": None, "levels": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[],
                        help="run directories, or globs such as experiments/runs/sac_medium_s*")
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shifted"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", default="policy.pt",
                        help="policy.pt (final) or best.pt (best during training)")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--task", default="lift", choices=("lift", "place"),
                        help="which task the policies were trained on; the "
                             "observation is identical, so nothing else warns "
                             "you if you evaluate a lift policy on place")
    parser.add_argument("--place-travel", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="evaluate the place task at this travel range. A "
                             "policy trained on a short range must be scored on "
                             "the range it trained on, or the number measures "
                             "generalisation instead of learning")
    parser.add_argument("--arm", action="store_true",
                        help="evaluate through the six-jointed arm")
    parser.add_argument("--handled", action="store_true",
                        help="the grasp-point-selection shape; implies --wrist")
    parser.add_argument("--label", default=None, help="name for this group of seeds")
    parser.add_argument("--expert", action="store_true", help="also evaluate the scripted expert")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    run_dirs: List[str] = []
    for pattern in args.runs:
        matches = sorted(glob.glob(pattern))
        run_dirs.extend(matches if matches else [pattern])
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]

    t0 = time.time()
    per_run = [
        evaluate_run(d, args.eval_levels, args.episodes, args.checkpoint,
                     args.max_steps, args.task, args.arm, args.place_travel,
                     args.handled)
        for d in run_dirs
    ]
    if args.expert:
        per_run.append(evaluate_expert(args.eval_levels, args.episodes,
                                       args.max_steps, args.task, args.arm, args.place_travel))

    policy_runs = [r for r in per_run if r["run"] != "scripted-expert"]
    aggregate: Dict[str, Dict] = {}
    for level in args.eval_levels:
        successes = [r["levels"][level]["successes"] for r in policy_runs]
        trials = [r["levels"][level]["episodes"] for r in policy_runs]
        if successes:
            aggregate[level] = summarise_seeds(successes, trials)
            aggregate[level]["mean_grasp_rate"] = float(
                np.mean([r["levels"][level]["grasp_rate"] for r in policy_runs])
            )

    blob = {
        "label": args.label or (run_dirs[0] if run_dirs else "expert"),
        "checkpoint": args.checkpoint,
        "episodes_per_seed": args.episodes,
        "eval_levels": args.eval_levels,
        "n_seeds": len(policy_runs),
        "single_seed_warning": (
            "one seed only: this is an anecdote, not a result" if len(policy_runs) == 1 else None
        ),
        "runs": per_run,
        "aggregate": aggregate,
        "wall_seconds": round(time.time() - t0, 1),
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)

    print("{}  ({} seeds x {} episodes)".format(blob["label"], len(policy_runs), args.episodes))
    for level in args.eval_levels:
        if level in aggregate:
            across = aggregate[level]["across_seeds"]
            pooled = aggregate[level]["pooled_wilson"]
            print("  {:<9s} success {:.3f}  95% CI across seeds [{:.3f}, {:.3f}]  "
                  "pooled Wilson [{:.3f}, {:.3f}]".format(
                      level, across["point"], across["low"], across["high"],
                      pooled["low"], pooled["high"]))
    for run in per_run:
        if run["run"] == "scripted-expert":
            print("  scripted expert: " + "  ".join(
                "{} {:.3f}".format(k, v["success_rate"]) for k, v in run["levels"].items()))
    if blob["single_seed_warning"]:
        print("  WARNING: " + blob["single_seed_warning"])


if __name__ == "__main__":
    main()
