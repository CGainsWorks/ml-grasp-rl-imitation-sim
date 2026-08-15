"""Can this task be learned without the hand-designed reward?

    python src/train_her.py --steps 200000 --seed 0 --her

`docs/limitations.md` has carried this as an open item since the beginning:
"nothing here shows RL discovering grasping from a sparse signal". The dense
reward has nine terms and `docs/reward-design.md` records two shapings that
failed before the current one worked, so a fair reading is that most of this
task's behaviour comes from the shaping rather than from the algorithm. This
tests that directly.

The reward is `src/rewards/configs/sparse.json`: 1.0 on the step the success
condition holds, 0 everywhere else. No reach, no lift, no place, no hold. On a
100-step episode a random policy sees that signal essentially never.

**Hindsight relabelling** (Andrychowicz et al., *Hindsight Experience Replay*,
NeurIPS 2017, https://arxiv.org/abs/1707.01495) is the standard answer. The
task is goal-conditioned -- the hold point sits in the observation at indices
26:29, with the goal-minus-object vector at 29:32 -- so a failed episode can be
relabelled as a successful one for whichever goal it *did* achieve, and the
sparse signal stops being empty.

The relabelling has to rewrite three things per transition, and getting any of
them wrong produces a buffer that trains happily on nonsense:

1. the goal entries of the observation and the next observation,
2. the goal-minus-object entries, which are derived from the goal,
3. the reward, recomputed against the new goal from the stored object position
   and grasp flag -- not copied, and not recomputed from the observation, whose
   object entries carry sensing noise.

Two arms, and the control is the point: sparse without hindsight should fail,
and if it does not then the hindsight arm proves nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.policies.sac import SAC, SACConfig  # noqa: E402
from src.rewards.grasp_reward import load_reward_config, success_condition  # noqa: E402
from src.utils.rollout import evaluate_policy  # noqa: E402

GOAL = slice(26, 29)
GOAL_MINUS_OBJ = slice(29, 32)
OBJ = slice(8, 11)


def relabel(obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray) -> None:
    """Rewrite the goal-dependent entries of a transition, in place."""
    for row in (obs, next_obs):
        row[GOAL] = goal
        row[GOAL_MINUS_OBJ] = goal - row[OBJ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomisation", default="none")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--her", action="store_true",
                        help="relabel with hindsight goals; off is the control arm")
    parser.add_argument("--her-k", type=int, default=4,
                        help="relabelled copies per real transition")
    parser.add_argument("--alpha-floor", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--output", default="experiments/runs/her")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    reward_cfg = load_reward_config("src/rewards/configs/sparse.json")
    env = make_env(args.randomisation, seed=args.seed, max_steps=args.max_steps,
                   reward_config="src/rewards/configs/sparse.json")
    eval_env = make_env(args.randomisation, seed=args.seed + 999,
                        max_steps=args.max_steps,
                        reward_config="src/rewards/configs/sparse.json")

    cfg = SACConfig(hidden=(args.hidden, args.hidden), alpha_floor=args.alpha_floor)
    agent = SAC(env.obs_dim, env.act_dim, cfg, seed=args.seed)

    with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"algorithm": "sac+her" if args.her else "sac",
                   "reward": "sparse", "her": args.her, "her_k": args.her_k,
                   "steps": args.steps, "seed": args.seed,
                   "randomisation": args.randomisation,
                   "sac": cfg.to_dict()}, fh, indent=2)

    progress = os.path.join(args.output, "progress.csv")
    with open(progress, "w", encoding="utf-8") as fh:
        fh.write("step,success_rate,grasp_rate,mean_return,mean_max_lift,"
                 "train_return,alpha,critic_loss,actor_loss,bc_coef,wall_seconds\n")

    obs, _ = env.reset()
    episode: List[Dict] = []
    best = -1.0
    metrics: Dict[str, float] = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0}
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if step <= cfg.start_steps:
            action = rng.uniform(-1.0, 1.0, env.act_dim).astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Keep what relabelling needs: the true object position and the grasp
        # flag, which the observation cannot supply once sensing noise is on.
        episode.append({
            "obs": obs.copy(), "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": float(reward), "next_obs": next_obs.copy(),
            "done": float(terminated),
            # The *true* object position, read from the simulator rather than
            # from the observation: under sensing noise the observation's object
            # entries are wrong, and relabelling against a wrong achieved goal
            # teaches the critic a reward that never happened.
            "object": env._object_pos().copy(),
            "grasped": float(info.get("grasped", 0.0)),
        })
        obs = next_obs

        if terminated or truncated:
            for i, tr in enumerate(episode):
                agent.buffer.add(tr["obs"], tr["action"], tr["reward"],
                                 tr["next_obs"], tr["done"])
                if not args.her:
                    continue
                # "future" strategy: goals the episode actually reached later.
                for _ in range(args.her_k):
                    j = int(rng.integers(i, len(episode)))
                    goal = episode[j]["object"].copy()
                    o, n = tr["obs"].copy(), tr["next_obs"].copy()
                    relabel(o, n, goal)
                    achieved = success_condition(
                        tr["object"][None, :], goal[None, :],
                        np.array([tr["grasped"]]), reward_cfg)
                    agent.buffer.add(o, tr["action"],
                                     float(reward_cfg.w_success * float(achieved[0])),
                                     n, tr["done"])
            episode = []
            obs, _ = env.reset()

        if step % cfg.update_every == 0 and agent.buffer.size >= cfg.batch_size:
            agent.observe_normalisation(obs[None, :])
            for _ in range(cfg.update_every):
                metrics = agent.update(step)

        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate_policy(
                eval_env, lambda o: agent.act(o, deterministic=True),
                n_episodes=args.eval_episodes, seed=10_000)
            with open(progress, "a", encoding="utf-8") as fh:
                row = ("{},{:.4f},{:.4f},{:.3f},{:.4f},0.0,{:.4f},{:.4f},"
                       "{:.4f},0.0,{:.1f}\n")
                fh.write(row.format(
                    step, result["success_rate"], result["grasp_rate"],
                    result["mean_return"], result["mean_max_lift"],
                    metrics.get("alpha", 0.0), metrics.get("critic_loss", 0.0),
                    metrics.get("actor_loss", 0.0), time.time() - t0))
            if result["success_rate"] > best:
                best = result["success_rate"]
            print("step {:>7d}  success {:.3f}  grasp {:.3f}  {:.0f}s".format(
                step, result["success_rate"], result["grasp_rate"],
                time.time() - t0), flush=True)

    final = evaluate_policy(eval_env, lambda o: agent.act(o, deterministic=True),
                            n_episodes=args.eval_episodes, seed=10_000)
    torch.save(agent.state_dict(), os.path.join(args.output, "policy.pt"))
    with open(os.path.join(args.output, "result.json"), "w", encoding="utf-8") as fh:
        json.dump({"final_success_rate": final["success_rate"],
                   "best_success_rate": best, "steps": args.steps,
                   "seed": args.seed, "her": args.her,
                   "reward": "sparse",
                   "wall_seconds": round(time.time() - t0, 1)}, fh, indent=2)
    print("final success {:.3f}, best {:.3f}".format(final["success_rate"], best))


if __name__ == "__main__":
    main()
