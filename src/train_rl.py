"""Train a SAC policy on the MuJoCo grasp task.

    python src/train_rl.py --steps 200000 --randomisation medium --seed 0 \
        --output experiments/runs/sac_medium_s0

Writes into the output directory:

    config.json      every hyperparameter, plus the randomisation ranges used
    progress.csv     one row per evaluation point: step, success rate, return
    policy.pt        the final actor and critic
    best.pt          the actor with the best evaluation success rate
    result.json      the final and best evaluation numbers

The evaluation during training uses a *fixed* set of episode seeds, held apart
from the seeds used for training resets, so the training curve is not measuring
luck. The final headline numbers come from ``src/evaluate.py``, which uses a
third, larger seed block again.

Imitation options (``--demos``, ``--bc-coef``, ``--init-actor``) turn the same
script into the imitation-plus-RL run; see ``docs/imitation.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import ACT_DIM, OBS_DIM, make_env  # noqa: E402
from src.policies.sac import SAC, SACConfig  # noqa: E402
from src.randomisation.domain_rand import load_randomisation  # noqa: E402
from src.rewards.grasp_reward import load_reward_config  # noqa: E402
from src.utils.rollout import evaluate_policy, torch_policy  # noqa: E402

EVAL_SEED_BLOCK = 500_000   # evaluation episodes during training
TRAIN_SEED_BLOCK = 0        # training resets


def load_demos(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {k: archive[k] for k in archive.files}


def train(args: argparse.Namespace) -> Dict:
    os.makedirs(args.output, exist_ok=True)
    torch.set_num_threads(args.threads)

    cfg = SACConfig(
        hidden=(args.hidden, args.hidden),
        gamma=args.gamma,
        lr_actor=args.lr,
        lr_critic=args.lr,
        lr_alpha=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        start_steps=args.start_steps,
        update_every=args.update_every,
        updates_per_step=args.updates_per_step,
        init_alpha=args.init_alpha,
        target_entropy_scale=args.target_entropy_scale,
        critic_warmup_updates=args.critic_warmup,
        bc_coef=args.bc_coef,
        bc_decay_steps=args.bc_decay_steps,
        bc_q_filter=args.bc_q_filter,
        demo_sample_fraction=args.demo_fraction,
    )

    env = make_env(args.randomisation, seed=args.seed, max_steps=args.max_steps,
                   reward_config=args.reward_config)
    eval_env = make_env(args.randomisation, seed=args.seed + 999, max_steps=args.max_steps,
                        reward_config=args.reward_config)

    agent = SAC(OBS_DIM, ACT_DIM, cfg, seed=args.seed)

    demo_meta: Optional[Dict] = None
    if args.demos:
        demos = load_demos(args.demos)
        n = agent.buffer.add_demonstrations(
            demos["observations"], demos["actions"], demos["rewards"],
            demos["next_observations"], demos["dones"],
        )
        agent.observe_normalisation(demos["observations"])
        demo_meta = json.loads(str(demos["meta"])) if "meta" in demos else {}
        demo_meta["transitions_loaded"] = int(n)
        print("seeded replay buffer with {} demonstration transitions".format(n))

    if args.init_actor:
        state = torch.load(args.init_actor, map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(state["actor"])
        print("initialised actor from {}".format(args.init_actor))

    config_blob = {
        "algorithm": "sac",
        "sac": cfg.to_dict(),
        "steps": args.steps,
        "seed": args.seed,
        "randomisation": args.randomisation,
        "randomisation_ranges": load_randomisation(args.randomisation).to_dict(),
        "reward_config": args.reward_config,
        "reward_weights": load_reward_config(args.reward_config).to_dict(),
        "max_steps": args.max_steps,
        "eval_episodes": args.eval_episodes,
        "eval_every": args.eval_every,
        "demos": args.demos,
        "demo_meta": demo_meta,
        "init_actor": args.init_actor,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
    }
    with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_blob, fh, indent=2)

    progress_path = os.path.join(args.output, "progress.csv")
    with open(progress_path, "w", encoding="utf-8") as fh:
        fh.write("step,success_rate,grasp_rate,mean_return,mean_max_lift,"
                 "train_return,alpha,critic_loss,actor_loss,bc_coef,wall_seconds\n")

    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=TRAIN_SEED_BLOCK + args.seed * 1_000_000)
    episode_return = 0.0
    recent_returns: list = []
    obs_window: list = [obs]
    last_metrics: Dict[str, float] = {"critic_loss": 0.0, "actor_loss": 0.0,
                                      "alpha": cfg.init_alpha, "bc_coef": cfg.bc_coef}
    best_success = -1.0
    t0 = time.time()
    updates_owed = 0.0
    episodes = 0

    for step in range(1, args.steps + 1):
        if step <= cfg.start_steps and not args.demos:
            action = rng.uniform(-1.0, 1.0, ACT_DIM).astype(np.float32)
        elif step <= cfg.start_steps // 4 and args.demos:
            # With demonstrations in the buffer there is no need for a long
            # uniform-random phase; a short one still decorrelates the start.
            action = rng.uniform(-1.0, 1.0, ACT_DIM).astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        obs_window.append(next_obs)
        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            episodes += 1
            recent_returns.append(episode_return)
            recent_returns = recent_returns[-20:]
            episode_return = 0.0
            obs, _ = env.reset()

        if step % cfg.update_every == 0 and agent.buffer.size >= cfg.batch_size:
            agent.observe_normalisation(np.asarray(obs_window, dtype=np.float32))
            obs_window = [obs]
            updates_owed += cfg.update_every * cfg.updates_per_step
            n_updates = int(updates_owed)
            updates_owed -= n_updates
            for _ in range(n_updates):
                last_metrics = agent.update(step)

        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate_policy(
                eval_env, torch_policy(agent.actor, deterministic=True),
                n_episodes=args.eval_episodes, seed=EVAL_SEED_BLOCK,
            )
            with open(progress_path, "a", encoding="utf-8") as fh:
                row = ("{},{:.4f},{:.4f},{:.3f},{:.4f},{:.3f},{:.4f},{:.4f},"
                       "{:.4f},{:.4f},{:.1f}\n")
                fh.write(row.format(
                        step, result["success_rate"], result["grasp_rate"],
                        result["mean_return"], result["mean_max_lift"],
                        float(np.mean(recent_returns)) if recent_returns else float("nan"),
                        last_metrics.get("alpha", 0.0), last_metrics.get("critic_loss", 0.0),
                        last_metrics.get("actor_loss", 0.0), last_metrics.get("bc_coef", 0.0),
                        time.time() - t0,
                    )
                )
            if result["success_rate"] > best_success:
                best_success = result["success_rate"]
                torch.save(agent.state_dict(), os.path.join(args.output, "best.pt"))
            if not args.quiet:
                print("step {:>7d}  success {:.3f}  grasp {:.3f}  return {:8.1f}  "
                      "alpha {:.3f}  {:.0f}s".format(
                          step, result["success_rate"], result["grasp_rate"],
                          result["mean_return"], last_metrics.get("alpha", 0.0),
                          time.time() - t0), flush=True)

    torch.save(agent.state_dict(), os.path.join(args.output, "policy.pt"))
    final = evaluate_policy(
        eval_env, torch_policy(agent.actor, deterministic=True),
        n_episodes=args.eval_episodes, seed=EVAL_SEED_BLOCK,
    )
    summary = {
        "final_success_rate": final["success_rate"],
        "final_grasp_rate": final["grasp_rate"],
        "best_success_rate": best_success,
        "episodes": episodes,
        "steps": args.steps,
        "wall_seconds": round(time.time() - t0, 1),
        "seed": args.seed,
        "randomisation": args.randomisation,
    }
    with open(os.path.join(args.output, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    env.close()
    eval_env.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomisation", default="none")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--reward-config", default=None,
                        help="JSON of reward weights; defaults to the documented ones")
    parser.add_argument("--output", default="experiments/runs/sac")
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1,
                        help="torch threads. One per process, because runs are "
                             "parallelised across seeds rather than within a run.")
    parser.add_argument("--quiet", action="store_true")
    # SAC
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=400_000)
    parser.add_argument("--start-steps", type=int, default=5_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--updates-per-step", type=float, default=1.0)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256, help="width of both hidden layers")
    parser.add_argument("--target-entropy-scale", type=float, default=1.0,
                        help="target entropy is -act_dim times this; raise it to keep a "
                             "policy closer to deterministic")
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="critic-only gradient steps before the actor is allowed to "
                             "move; use with --init-actor")
    # Imitation
    parser.add_argument("--demos", default=None, help="npz of demonstrations to seed the buffer")
    parser.add_argument("--bc-coef", type=float, default=0.0)
    parser.add_argument("--bc-decay-steps", type=int, default=0)
    parser.add_argument("--bc-q-filter", action="store_true",
                        help="apply the BC term only where the critic prefers the expert "
                             "action; off by default, see src/policies/sac.py")
    parser.add_argument("--demo-fraction", type=float, default=0.0)
    parser.add_argument("--init-actor", default=None,
                        help="checkpoint to initialise the actor from")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
