"""Record expert demonstrations to a compressed archive.

Usage
-----
    python src/record_demos.py --episodes 200 --randomisation low \
        --output demonstrations/expert_low.npz

The archive is a flat set of transitions plus episode boundaries, which is the
format both behaviour cloning and the replay-buffer seeding read. Every file
records the environment settings, the randomisation level and the git-free
provenance fields needed to tell two demonstration sets apart six months later.

Failed episodes are dropped by default (``--keep-failures`` overrides). Cloning
an expert's failures teaches a policy to fail in the same places; if you want
recovery behaviour, the right tool is DAgger, which is in ``src/train_il.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from envs.mujoco.perception_env import (  # noqa: E402
    DEFAULT_CHECKPOINT as PERCEPTION_DEFAULT, make_perception_env)
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.policies.scripted_place_expert import ScriptedPlaceExpert  # noqa: E402


def record(
    episodes: int,
    randomisation: str,
    seed: int,
    expert_noise: float,
    keep_failures: bool,
    max_steps: int,
    task: str = "lift",
    arm: bool = False,
    handled: bool = False,
    wrist: bool = False,
    privileged: bool = False,
    history: int = 1,
    max_half_size=None,
    perception=None,
    clutter: int = 0,
) -> dict:
    factory = make_perception_env if perception else make_env
    extra = {"checkpoint": perception} if perception else {}
    env = factory(randomisation, seed=seed, max_steps=max_steps, task=task,
                  arm=arm, handled=handled, wrist=handled or wrist,
                  history=history, max_half_size=max_half_size,
                  clutter=clutter, **extra)
    rng = np.random.default_rng(seed)
    expert_cls = ScriptedPlaceExpert if task == "place" else ScriptedExpert
    if handled:
        # The handled shape's graspable point is not its reported pose, so the
        # expert is told where it is. That is the supervision a demonstration
        # carries; the policy cloning it gets the observation only.
        from envs.mujoco.grasp_env import HANDLE_CENTRE, HANDLE_HEIGHT
        expert = expert_cls(noise=expert_noise, rng=rng, wrist=True,
                            grasp_offset=HANDLE_CENTRE,
                            grasp_yaw_offset=np.pi / 2.0,
                            grasp_height=HANDLE_HEIGHT)
    elif wrist:
        expert = expert_cls(noise=expert_noise, rng=rng, wrist=True)
    else:
        expert = expert_cls(noise=expert_noise, rng=rng)

    obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
    episode_starts, episode_lengths, episode_success = [], [], []
    attempted = kept = 0
    t0 = time.time()

    while kept < episodes:
        attempted += 1
        expert.reset()
        obs, _ = env.reset(seed=seed + 100_000 + attempted)
        ep_obs, ep_act, ep_rew, ep_next, ep_done = [], [], [], [], []
        info: dict = {}
        while True:
            # Privileged distillation: the expert acts on the *clean* state, the
            # transition stores the *noisy* observation the policy will have.
            #
            # This is the only honest way to demonstrate a task whose sensing is
            # too poor to act on directly. At the camera's measured 0.0513 m the
            # object's reported position is wrong by more than the box is wide,
            # and the error is correlated in time at 0.947 -- so it cannot be
            # averaged away by a filter or a short window, which is exactly why
            # both of those failed. A demonstrator that can see teaches a policy
            # that cannot, and the policy has to learn what to do rather than
            # where the box is.
            action = expert.act(env.clean_observation() if privileged else obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_obs.append(obs)
            ep_act.append(action)
            ep_rew.append(reward)
            ep_next.append(next_obs)
            # `terminated` only: a time-limit truncation is not a real terminal
            # state and bootstrapping through it is correct.
            ep_done.append(float(terminated))
            obs = next_obs
            if terminated or truncated:
                break

        success = bool(info.get("is_success", False))
        if not success and not keep_failures:
            continue

        episode_starts.append(len(obs_list))
        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
        rew_list.extend(ep_rew)
        next_list.extend(ep_next)
        done_list.extend(ep_done)
        episode_lengths.append(len(ep_obs))
        episode_success.append(int(success))
        kept += 1

    env.close()
    return {
        "observations": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.float32),
        "rewards": np.asarray(rew_list, dtype=np.float32),
        "next_observations": np.asarray(next_list, dtype=np.float32),
        "dones": np.asarray(done_list, dtype=np.float32),
        "episode_starts": np.asarray(episode_starts, dtype=np.int64),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
        "episode_success": np.asarray(episode_success, dtype=np.int64),
        "meta": json.dumps(
            {
                "episodes_requested": episodes,
                "episodes_attempted": attempted,
                "episodes_kept": kept,
                "expert_success_rate": kept / max(1, attempted) if not keep_failures else None,
                "randomisation": randomisation,
                "seed": seed,
                "expert_noise": expert_noise,
                "keep_failures": keep_failures,
                "max_steps": max_steps,
                "task": task,
                "arm": arm,
                "handled": handled,
                "wrist": wrist,
                "privileged": privileged,
                "history": history,
                "max_half_size": max_half_size,
                "perception": perception,
                "clutter": clutter,
                "wall_seconds": round(time.time() - t0, 1),
                "source": ("src/policies/scripted_place_expert.py" if task == "place"
                           else "src/policies/scripted_expert.py"),
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--randomisation", default="low")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expert-noise", type=float, default=0.02,
                        help="Gaussian noise added to expert actions. A little noise "
                             "widens the state distribution the clone sees, which is "
                             "the cheapest available cure for compounding error.")
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--task", default="lift", choices=("lift", "place"))
    parser.add_argument("--max-half-size", type=float, default=None,
                        help="cap on object half-size in metres. The "
                             "default (0.024) sits below the band where "
                             "yaw binds against the 0.078 m pad gap; "
                             "0.028-0.039 is where a square box fits "
                             "aligned and does not fit rotated")
    parser.add_argument("--perception", nargs="?", const=PERCEPTION_DEFAULT,
                        default=None,
                        help="put the pose CNN in the loop: the object's "
                             "position in every observation comes from a "
                             "64x64 render instead of the simulator. "
                             "About 250x slower per step")
    parser.add_argument("--clutter", type=int, default=0,
                        help="distractor objects on the table, up to 3")
    parser.add_argument("--history", type=int, default=1,
                        help="stack this many observation frames, matching "
                             "the policy the demonstrations will train")
    parser.add_argument("--privileged", action="store_true",
                        help="the expert sees the noise-free state while the "
                             "stored observation stays noisy. For sensing levels "
                             "no demonstrator could act through")
    parser.add_argument("--wrist", action="store_true",
                        help="the 5-D wrist-yaw variant: the pads can be turned "
                             "square to a face before closing")
    parser.add_argument("--handled", action="store_true",
                        help="the grasp-point-selection shape: a cube that "
                             "cannot be grasped anywhere, with an offset handle "
                             "that can. Implies --wrist")
    parser.add_argument("--arm", action="store_true",
                        help="record through the six-jointed arm rather than "
                             "the mocap weld")
    parser.add_argument("--output", default="demonstrations/expert.npz")
    args = parser.parse_args()

    data = record(
        args.episodes, args.randomisation, args.seed,
        args.expert_noise, args.keep_failures, args.max_steps, args.task,
        args.arm, args.handled, args.wrist, args.privileged, args.history,
        args.max_half_size, args.perception, args.clutter,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **data)

    meta = json.loads(data["meta"])
    size_mb = os.path.getsize(args.output) / 1e6
    print(
        "wrote {} - {} episodes, {} transitions, {:.2f} MB\n"
        "  expert success while recording: {}\n"
        "  randomisation: {}  seed: {}  noise: {}".format(
            args.output, len(data["episode_lengths"]), len(data["observations"]),
            size_mb,
            "n/a (failures kept)" if meta["expert_success_rate"] is None
            else "{:.3f}".format(meta["expert_success_rate"]),
            meta["randomisation"], meta["seed"], meta["expert_noise"],
        )
    )


if __name__ == "__main__":
    main()
