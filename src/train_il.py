"""Imitation learning: behaviour cloning, and DAgger on top of it.

    python src/train_il.py --demos demonstrations/expert_low.npz \
        --epochs 60 --seed 0 --output experiments/runs/bc_s0

    python src/train_il.py --demos demonstrations/expert_low.npz --dagger \
        --dagger-rounds 5 --output experiments/runs/dagger_s0

Behaviour cloning
-----------------
Supervised regression from observation to expert action, mean squared error on
the tanh-squashed mode of the same actor network SAC uses. Nothing clever: the
point of having it is that it is the honest baseline the RL results have to
beat, and the starting point the imitation-plus-RL run is initialised from.

BC's failure mode is not a mystery and it shows up clearly on this task. The
clone is trained on states the expert visits; the moment it makes a small error
it is somewhere the expert never was, and its next action is an extrapolation.
Errors compound. That is why the clone's success rate is well below the
expert's even though its action error looks small.

DAgger
------
The fix that keeps the supervised setup: roll out the *learner*, label the
states it actually visited with the expert's action, add them to the dataset,
retrain. Each round the dataset covers more of the learner's own state
distribution. It costs expert queries at run time, which here are free because
the expert is a function; on a real robot they are a human with a joystick, and
that is exactly why DAgger is less popular in practice than its results
deserve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import ACT_DIM, OBS_DIM, make_env  # noqa: E402
from src.policies.behaviour_cloning import fit  # noqa: E402
from src.policies.networks import SquashedGaussianActor  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.utils.rollout import evaluate_policy, torch_policy  # noqa: E402

EVAL_SEED_BLOCK = 500_000


def load_demos(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {k: archive[k] for k in archive.files}


def dagger_round(
    env,
    actor: SquashedGaussianActor,
    expert: ScriptedExpert,
    episodes: int,
    beta: float,
    seed: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Roll out a mixture of learner and expert, label every state with the expert.

    ``beta`` is the probability of executing the expert's action instead of the
    learner's on any given step. It is annealed to zero across rounds, which is
    the standard schedule: early rounds stay near states where the expert is
    competent, later rounds go wherever the learner actually goes.
    """
    obs_out, act_out = [], []
    successes = 0
    for ep in range(episodes):
        expert.reset()
        obs, _ = env.reset(seed=seed + ep)
        info: Dict = {}
        while True:
            expert_action = expert.act(obs)
            obs_out.append(obs.copy())
            act_out.append(expert_action.copy())
            if rng.random() < beta:
                action = expert_action
            else:
                action = actor.act(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += int(bool(info.get("is_success", False)))
    return (
        np.asarray(obs_out, dtype=np.float32),
        np.asarray(act_out, dtype=np.float32),
        successes / max(1, episodes),
    )


def train(args: argparse.Namespace) -> Dict:
    os.makedirs(args.output, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    demos = load_demos(args.demos)
    obs = demos["observations"]
    act = demos["actions"]
    demo_meta = json.loads(str(demos["meta"])) if "meta" in demos else {}

    if args.max_demo_episodes:
        # Truncate to the first N episodes, for the data-efficiency sweep.
        lengths = demos["episode_lengths"]
        keep = int(np.sum(lengths[: args.max_demo_episodes]))
        obs, act = obs[:keep], act[:keep]

    actor = SquashedGaussianActor(OBS_DIM, ACT_DIM, (args.hidden, args.hidden))
    env = make_env(args.randomisation, seed=args.seed, max_steps=args.max_steps,
                   task=args.task, arm=args.arm)
    eval_env = make_env(args.randomisation, seed=args.seed + 999,
                        max_steps=args.max_steps, task=args.task, arm=args.arm)
    expert = ScriptedExpert(noise=0.0, rng=rng)

    t0 = time.time()
    history: List[Dict] = []

    train_curve, val_curve = fit(
        actor, obs, act, args.epochs, args.batch_size, args.lr, args.weight_decay, rng
    )
    result = evaluate_policy(
        eval_env, torch_policy(actor, deterministic=True),
        n_episodes=args.eval_episodes, seed=EVAL_SEED_BLOCK,
    )
    history.append({
        "round": 0, "transitions": int(len(obs)),
        "train_mse": train_curve[-1], "val_mse": val_curve[-1],
        "success_rate": result["success_rate"], "grasp_rate": result["grasp_rate"],
    })
    if not args.quiet:
        print("bc      transitions {:>6d}  val mse {:.5f}  success {:.3f}".format(
            len(obs), val_curve[-1], result["success_rate"]), flush=True)

    if args.dagger:
        for rnd in range(1, args.dagger_rounds + 1):
            beta = max(0.0, args.dagger_beta * (1.0 - (rnd - 1) / max(1, args.dagger_rounds - 1)))
            new_obs, new_act, learner_success = dagger_round(
                env, actor, expert, args.dagger_episodes, beta,
                seed=800_000 + rnd * 1_000 + args.seed * 10_000, rng=rng,
            )
            obs = np.concatenate([obs, new_obs])
            act = np.concatenate([act, new_act])
            train_curve, val_curve = fit(
                actor, obs, act, args.dagger_epochs, args.batch_size,
                args.lr, args.weight_decay, rng,
            )
            result = evaluate_policy(
                eval_env, torch_policy(actor, deterministic=True),
                n_episodes=args.eval_episodes, seed=EVAL_SEED_BLOCK,
            )
            history.append({
                "round": rnd, "transitions": int(len(obs)), "beta": beta,
                "rollout_success": learner_success,
                "train_mse": train_curve[-1], "val_mse": val_curve[-1],
                "success_rate": result["success_rate"], "grasp_rate": result["grasp_rate"],
            })
            if not args.quiet:
                print("dagger {:d}  transitions {:>6d}  val mse {:.5f}  success {:.3f}".format(
                    rnd, len(obs), val_curve[-1], result["success_rate"]), flush=True)

    torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
                "hidden": args.hidden}, os.path.join(args.output, "policy.pt"))

    summary = {
        "algorithm": "dagger" if args.dagger else "bc",
        "final_success_rate": result["success_rate"],
        "final_grasp_rate": result["grasp_rate"],
        "final_val_mse": val_curve[-1],
        "transitions": int(len(obs)),
        "demo_episodes": int(len(demos["episode_lengths"])) if not args.max_demo_episodes
        else int(args.max_demo_episodes),
        "history": history,
        "seed": args.seed,
        "randomisation": args.randomisation,
        "demos": args.demos,
        "demo_meta": demo_meta,
        "epochs": args.epochs,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.output, "progress.csv"), "w", encoding="utf-8") as fh:
        fh.write("round,transitions,train_mse,val_mse,success_rate,grasp_rate\n")
        for row in history:
            fh.write("{},{},{:.6f},{:.6f},{:.4f},{:.4f}\n".format(
                row["round"], row["transitions"], row["train_mse"], row["val_mse"],
                row["success_rate"], row["grasp_rate"]))
    with open(os.path.join(args.output, "loss_curve.csv"), "w", encoding="utf-8") as fh:
        fh.write("epoch,train_mse,val_mse\n")
        for i, (tr, va) in enumerate(zip(train_curve, val_curve)):
            fh.write("{},{:.6f},{:.6f}\n".format(i + 1, tr, va))

    env.close()
    eval_env.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", default="demonstrations/expert_low.npz")
    parser.add_argument("--output", default="experiments/runs/bc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomisation", default="none")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--task", default="lift", choices=("lift", "place"))
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-demo-episodes", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    # DAgger
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-episodes", type=int, default=25)
    parser.add_argument("--dagger-epochs", type=int, default=25)
    parser.add_argument("--dagger-beta", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
