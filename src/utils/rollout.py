"""Policy evaluation: one place where "success rate" is defined.

Every script that quotes a success rate calls :func:`evaluate_policy`, so the
protocol cannot drift between training-time logging, the ablation table and the
README. The protocol is:

* ``n_episodes`` episodes, each with an explicit seed taken from a fixed
  sequence, so two policies are compared on exactly the same worlds;
* actions are the deterministic mode of the policy, not a sample;
* success is the environment's ``is_success`` flag at the **final** step.

The last point is the one that matters. Counting "success at any point during
the episode" would reward a policy that lifts the box and drops it, and that is
the single easiest way to make grasping numbers look good.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

Policy = Callable[[np.ndarray], np.ndarray]


def evaluate_policy(
    env,
    policy: Policy,
    n_episodes: int = 100,
    seed: int = 10_000,
    on_episode_start: Optional[Callable[[], None]] = None,
    collect_trajectories: bool = False,
) -> Dict:
    """Run ``n_episodes`` deterministic episodes and summarise the outcome."""
    successes = 0
    returns: List[float] = []
    lifts: List[float] = []
    grasped_any: List[float] = []
    episode_success: List[int] = []
    trajectories: List[Dict] = []

    for i in range(n_episodes):
        if on_episode_start is not None:
            on_episode_start()
        obs, _ = env.reset(seed=seed + i)
        done = False
        total = 0.0
        max_lift = 0.0
        ever_grasped = 0.0
        frames: List[np.ndarray] = []
        info: Dict = {}
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            max_lift = max(max_lift, float(info.get("object_height", 0.0)))
            ever_grasped = max(ever_grasped, float(info.get("grasped", 0.0)))
            if collect_trajectories:
                frames.append(obs.copy())
            done = terminated or truncated

        ok = bool(info.get("is_success", False))
        successes += int(ok)
        episode_success.append(int(ok))
        returns.append(total)
        lifts.append(max_lift)
        grasped_any.append(ever_grasped)
        if collect_trajectories:
            trajectories.append({"obs": np.asarray(frames), "success": ok})

    result = {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / max(1, n_episodes),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_max_lift": float(np.mean(lifts)) if lifts else float("nan"),
        "grasp_rate": float(np.mean(grasped_any)) if grasped_any else float("nan"),
        "episode_success": episode_success,
        "seed": seed,
    }
    if collect_trajectories:
        result["trajectories"] = trajectories
    return result


def torch_policy(actor, deterministic: bool = True) -> Policy:
    """Wrap a torch actor as a plain ``obs -> action`` callable."""

    def _policy(obs: np.ndarray) -> np.ndarray:
        return actor.act(obs, deterministic=deterministic)

    return _policy
