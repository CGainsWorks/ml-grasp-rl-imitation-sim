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

``--task place`` runs the same thing on pick-and-place, and there it is aimed at
a specific finding rather than at the general question. Seven reward designs and
two curricula established that shaping on that task buys *segments* and does not
chain them: from-scratch policies grasp and never lift, curriculum policies lift,
carry and release and never grasp. Hindsight is the one method left that attacks
the chain directly, because relabelling turns "picked the box up and put it down
somewhere" into a success for wherever it was put down -- and that is the whole
sequence, rewarded, without a demonstration or a shaping term for any part of it.

The place relabelling needs one thing the lift version does not. Its success
condition also reads the lift latch and the object's speed, so both are stored
per transition and recomputed, for the same reason the object position is: the
observation's copies carry sensing noise, and a buffer trained on noisy labels
looks exactly like a buffer trained on real ones until the numbers come out
wrong.
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
from src.rewards.place_reward import (  # noqa: E402
    load_place_config,
    place_success_condition,
)
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
    parser.add_argument("--task", default="lift", choices=("lift", "place"))
    parser.add_argument("--no-lift-latch", action="store_true",
                        help="drop the pick requirement from the place success "
                             "condition. Not a task to report numbers on -- it "
                             "exists to test whether the latch is what makes "
                             "hindsight relabelling inapplicable")
    parser.add_argument("--start-progress", type=float, default=0.0,
                        help="reverse curriculum: begin episodes this far "
                             "along the scripted trajectory (Florensa et "
                             "al. 2017). Sparse reward plus hindsight "
                             "cannot explore its way to a first success; "
                             "this supplies one")
    parser.add_argument("--start-range", type=float, nargs=2, default=None,
                        help="low and high of the start-progress band, so every "
                             "batch spans the whole task rather than one stage "
                             "at a time")
    parser.add_argument("--observe-latch", action="store_true",
                        help="append the lift latch to the observation, "
                             "making the place task Markovian")
    parser.add_argument("--her", action="store_true",
                        help="relabel with hindsight goals; off is the control arm")
    parser.add_argument("--her-k", type=int, default=4,
                        help="relabelled copies per real transition")
    parser.add_argument("--alpha-floor", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads; 1 is fastest when running "
                             "several trainings at once, which is the normal case")
    parser.add_argument("--output", default="experiments/runs/her")
    args = parser.parse_args()

    # One compute thread per process. These networks are 128x128 at batch 256:
    # far too small for intra-op parallelism to pay, and the grid runs eight to
    # twelve processes at once, so the default of eight threads each asks for
    # ~96 threads on 16 cores. Measured under exactly that load, one thread is
    # 2.6x faster per update than eight (10.5 ms against 27.5 ms). The runs are
    # parallel across processes; there is nothing left to parallelise inside one.
    torch.set_num_threads(args.threads)

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # How often relabelling actually manufactures a success. If this is zero the
    # hindsight arm is doing nothing and its result means nothing, which is a
    # failure mode that looks identical to "the method did not help".
    relabelled_hits = relabelled_total = 0

    place = args.task == "place"
    sparse_cfg = (("src/rewards/configs/place_sparse_nolatch.json"
                   if args.no_lift_latch else "src/rewards/configs/place_sparse.json")
                  if place else "src/rewards/configs/sparse.json")
    reward_cfg = (load_place_config if place else load_reward_config)(sparse_cfg)
    env = make_env(args.randomisation, seed=args.seed, max_steps=args.max_steps,
                   task=args.task, reward_config=sparse_cfg,
                   observe_latch=args.observe_latch,
                   start_progress=args.start_progress,
                   start_progress_range=args.start_range)
    # Evaluation always starts at the beginning of the task. A curriculum that
    # is also applied at evaluation reports how well the policy finishes a job
    # someone else started.
    eval_env = make_env(args.randomisation, seed=args.seed + 999,
                        max_steps=args.max_steps, task=args.task,
                        reward_config=sparse_cfg,
                        observe_latch=args.observe_latch)

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
            "rest_z": float(env._object_rest_z),
            "grasped": float(info.get("grasped", 0.0)),
            "lifted": float(info.get("lifted", 0.0)),
            "speed": float(info.get("object_speed", 0.0)),
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
                    if place:
                        # A relabelled target has to sit at the height the object
                        # rests at. Relabelling to an achieved *airborne*
                        # position would manufacture a goal that the task's own
                        # success condition can never satisfy, and the hindsight
                        # arm would then be training on impossible goals while
                        # looking exactly like one that was working.
                        goal[2] = tr["rest_z"]
                    o, n = tr["obs"].copy(), tr["next_obs"].copy()
                    relabel(o, n, goal)
                    if place:
                        # The place condition also reads the lift latch and the
                        # object's speed, both stored rather than re-derived.
                        achieved = place_success_condition(
                            tr["object"][None, :], goal[None, :],
                            np.array([tr["grasped"]]), np.array([tr["lifted"]]),
                            np.array([tr["speed"]]), reward_cfg)
                    else:
                        achieved = success_condition(
                            tr["object"][None, :], goal[None, :],
                            np.array([tr["grasped"]]), reward_cfg)
                    agent.buffer.add(o, tr["action"],
                                     float(reward_cfg.w_success * float(achieved[0])),
                                     n, tr["done"])
                    relabelled_hits += int(bool(achieved[0]))
                    relabelled_total += 1
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
                   "relabelled_success_fraction": (
                       relabelled_hits / relabelled_total
                       if relabelled_total else None),
                   "relabelled_transitions": relabelled_total,
                   "wall_seconds": round(time.time() - t0, 1)}, fh, indent=2)
    print("final success {:.3f}, best {:.3f}".format(final["success_rate"], best))


if __name__ == "__main__":
    main()
