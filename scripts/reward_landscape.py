"""Print the reward-per-step table that docs/reward-design.md quotes.

    python scripts/reward_landscape.py

Every number in that table comes from here, so the document cannot drift away
from the code that implements it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.grasp_reward import GraspRewardConfig, grasp_reward  # noqa: E402

TABLE_HEIGHT = 0.40
REST_Z = 0.422          # box half-size 0.022 resting on the table
GOAL_Z = TABLE_HEIGHT + 0.15

STATES = [
    ("box on the table", REST_Z),
    ("lifted 5 cm", REST_Z + 0.05),
    ("lifted 8 cm", REST_Z + 0.08),
    ("at the hold point", GOAL_Z),
    ("5 cm above the hold point", GOAL_Z + 0.05),
    ("at the ceiling of the workspace", 0.64),
]


def reward_at(object_z: float, grasped: float, cfg: GraspRewardConfig) -> float:
    """Hand at the box, box at ``object_z``, nothing moving."""
    obj = np.array([[0.0, 0.0, object_z]])
    grip = np.array([[0.0, 0.0, object_z]])
    goal = np.array([[0.0, 0.0, GOAL_Z]])
    value, _ = grasp_reward(
        grip, obj, goal, np.array([REST_Z]), np.array([grasped]),
        np.array([0.0]), np.zeros((1, 4)), cfg,
    )
    return float(value[0])


def main() -> None:
    cfg = GraspRewardConfig()
    print("reward per step, hand at the box\n")
    print("{:<34s} {:>9s} {:>13s}".format("state", "grasped", "not grasped"))
    for label, z in STATES:
        print("{:<34s} {:>9.2f} {:>13.2f}".format(
            label, reward_at(z, 1.0, cfg), reward_at(z, 0.0, cfg)))
    print("\nweights: " + ", ".join(
        "{}={}".format(k, v) for k, v in cfg.to_dict().items() if k.startswith("w_")))


if __name__ == "__main__":
    main()
