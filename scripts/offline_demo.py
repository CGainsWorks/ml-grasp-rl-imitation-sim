"""The one-command demo: the scripted expert grasping, with the reward broken out.

    python scripts/offline_demo.py
    python scripts/offline_demo.py --episodes 40 --randomisation medium

Needs no trained policy, no GPU, no display, and about ten seconds. It is also
the environment's smoke test: if this stops printing a high success rate on the
nominal world, the scene is broken.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.policies.scripted_expert import (  # noqa: E402
    APPROACH,
    CLOSE,
    DESCEND,
    LIFT,
    ScriptedExpert,
)
from src.utils.stats import wilson_interval  # noqa: E402

PHASE_NAMES = {APPROACH: "approach", DESCEND: "descend", CLOSE: "close", LIFT: "lift"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--randomisation", default="none")
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--trace", action="store_true", help="print one episode step by step")
    args = parser.parse_args()

    env = make_env(args.randomisation, seed=1234)
    expert = ScriptedExpert()

    if args.trace:
        print("one episode, one line per five steps\n")
        print("{:>5s} {:>9s} {:>7s} {:>7s} {:>7s} {:>8s}".format(
            "step", "phase", "lift", "width", "grasp", "reward"))
        obs, _ = env.reset(seed=args.seed)
        expert.reset()
        step = 0
        while True:
            action = expert.act(obs)
            phase = expert.phase
            obs, reward, terminated, truncated, info = env.step(action)
            if step % 5 == 0:
                print("{:>5d} {:>9s} {:>7.3f} {:>7.3f} {:>7.0f} {:>8.2f}".format(
                    step, PHASE_NAMES[phase], info["object_height"],
                    float(obs[6]), info["grasped"], reward))
            step += 1
            if terminated or truncated:
                break
        print("\nterms on the final step:")
        for name, value in info["reward_terms"].items():
            if abs(value) > 1e-9:
                print("  {:<9s} {:+7.3f}".format(name, value))
        print()

    successes = 0
    returns = []
    lifts = []
    t0 = time.time()
    for i in range(args.episodes):
        expert.reset()
        obs, _ = env.reset(seed=args.seed + i)
        total = 0.0
        peak = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(expert.act(obs))
            total += reward
            peak = max(peak, info["object_height"])
            if terminated or truncated:
                break
        successes += int(info["is_success"])
        returns.append(total)
        lifts.append(peak)
    env.close()

    interval = wilson_interval(successes, args.episodes)
    print("scripted expert, randomisation '{}', {} episodes in {:.1f}s".format(
        args.randomisation, args.episodes, time.time() - t0))
    print("  success rate   {:.3f}  95% Wilson [{:.3f}, {:.3f}]".format(
        interval.point, interval.low, interval.high))
    print("  mean return    {:.1f}".format(float(np.mean(returns))))
    print("  mean peak lift {:.3f} m".format(float(np.mean(lifts))))

    if args.randomisation == "none" and interval.point < 0.9:
        raise SystemExit(
            "expert success on the nominal world dropped to {:.2f}; the scene is "
            "broken, not the policy".format(interval.point)
        )


if __name__ == "__main__":
    main()
