"""Can hindsight relabelling manufacture a success on the place task at all?

    python scripts/her_relabel_probe.py

`src/rewards/place_reward.py` used to say hindsight was *structurally
inapplicable* to pick-and-place, because the success condition reads a lift
latch -- "this box was once 4 cm off the table" -- which is a fact about the
episode's history rather than about the achieved state. Relabelling can move the
target to wherever the box ended up; it cannot retroactively pick the box up.

That reasoning is wrong, and this measures why. `src/train_her.py` already
stores ``lifted`` per transition and recomputes the success condition with it,
so the latch does travel with the relabelled transition. The claim confused
"the latch is history" with "the latch is unavailable", and only the first is
true.

What actually produced the documented zero is upstream and much simpler: with a
sparse reward and no curriculum the policy never lifts the box, so no transition
in the buffer has the latch set, and every relabelled goal is evaluated against
``lifted = 0``. Hindsight can reinterpret experience. It cannot invent it.

Reverse curriculum generation (Florensa et al., 2017) is the standard remedy for
exactly this: sparse goal-reaching needs one state in which the task is
achieved, and then start states are drawn near the goal and walked outward.
`GraspEnv` already supports it through ``start_progress_range``.

So this runs a *random* policy -- deliberately, so the result is about the
relabeller rather than about any trained agent -- and counts how often
relabelling produces a success, with and without curriculum start states. A
random policy is the hardest case: if relabelling fires for random actions, it
will fire for anything.

Measured, 40 episodes and k=4 relabels per transition:

    no curriculum        0 / 16000 successes, latch set on 0.00 of frames
    curriculum 0.2-0.8   8273 / 16000 successes, latch set on 0.60 of frames

The first row reproduces the number the documentation reports. The second shows
it was a statement about exploration, not about hindsight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.rewards.place_reward import (  # noqa: E402
    load_place_config, place_success_condition)

SPARSE = "src/rewards/configs/place_sparse.json"


def probe(label: str, episodes: int, her_k: int, seed: int, **kwargs) -> dict:
    cfg = load_place_config(SPARSE)
    rng = np.random.default_rng(seed)
    env = make_env("none", seed=seed, task="place", observe_latch=True,
                   reward_config=SPARSE, **kwargs)
    hits = total = latch_frames = released_frames = frames = 0
    for ep in range(episodes):
        env.reset(seed=800 + ep)
        traj = []
        while True:
            action = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)
            traj.append({"object": env._object_pos().copy(),
                         "rest_z": float(env._object_rest_z),
                         "grasped": float(info.get("grasped", 0.0)),
                         "lifted": float(info.get("lifted", 0.0)),
                         "speed": float(info.get("object_speed", 0.0))})
            if terminated or truncated:
                break
        for i, tr in enumerate(traj):
            frames += 1
            latch_frames += int(tr["lifted"] > 0.5)
            released_frames += int(tr["grasped"] < 0.5)
            for _ in range(her_k):
                j = int(rng.integers(i, len(traj)))
                goal = traj[j]["object"].copy()
                # A relabelled target has to sit at rest height: an airborne
                # goal is one the success condition can never satisfy.
                goal[2] = tr["rest_z"]
                achieved = place_success_condition(
                    tr["object"][None, :], goal[None, :],
                    np.array([tr["grasped"]]), np.array([tr["lifted"]]),
                    np.array([tr["speed"]]), cfg)
                hits += int(bool(achieved[0]))
                total += 1
    env.close()
    row = {"label": label, "relabelled_successes": hits,
           "relabelled_total": total, "hit_rate": hits / max(total, 1),
           "latch_fraction": latch_frames / max(frames, 1),
           "released_fraction": released_frames / max(frames, 1)}
    print("  %-22s %6d / %6d (%.3f)   latch %.2f  released %.2f"
          % (label, hits, total, row["hit_rate"], row["latch_fraction"],
             row["released_fraction"]), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--her-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output",
                        default="experiments/results/her_relabel_probe.json")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("hindsight relabelling under a random policy\n")
    rows = [
        probe("no curriculum", args.episodes, args.her_k, args.seed),
        probe("curriculum 0.2-0.8", args.episodes, args.her_k, args.seed,
              start_progress=0.0, start_progress_range=(0.2, 0.8)),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({
            "policy": "uniform random, deliberately: this measures the "
                      "relabeller rather than any trained agent",
            "note": "the zero without a curriculum is an exploration result, "
                    "not a structural property of hindsight. The latch travels "
                    "with the transition and is recomputed correctly; there is "
                    "simply never a lifted transition to relabel.",
            "rows": rows}, fh, indent=2)
    print("\nwrote " + args.output)


if __name__ == "__main__":
    main()
