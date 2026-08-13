"""Solve the Franka pre-grasp joint configuration baked into the Isaac task.

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_pregrasp.py

The numbers in ``GraspTaskCfg.robot.init_state.joint_pos`` came from here, so
they are reproducible rather than magic. The method is empirical on purpose:
pin the environment's Cartesian setpoint to the canonical reset pose, let its
own differential IK settle for 200 steps, and read the arm back.

Why not the Franka's shipped home pose: it puts the fingertips at z = 0.383,
below the 0.40 m table top. The arm starts inside the table and flicks the box
off it on the way out.

The printed grip height will *not* equal the commanded setpoint. The arm's
implicit PD holds against gravity with a finite stiffness, so there is a
standing error of several centimetres that grows as the arm extends. That error
is the main open problem with this port; see ``envs/isaac/README.md``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import (  # noqa: E402
    RESET_HEIGHT,
    TABLE_CENTRE_X,
    TABLE_HEIGHT,
    GraspTask,
    GraspTaskCfg,
    _quat_to_matrix,
)

cfg = GraspTaskCfg()
cfg.scene.num_envs = 1
cfg.randomisation_level = "none"
env = GraspTask(cfg)
env.reset()

canonical = torch.tensor(
    [[TABLE_CENTRE_X, 0.0, TABLE_HEIGHT + RESET_HEIGHT]], device=env.device
)
idle = torch.zeros((1, 4), device=env.device)
idle[:, 3] = -1.0  # fingers open

for _ in range(200):
    env._target_pos[:] = canonical
    env.step(idle)

rot = _quat_to_matrix(env._robot.data.body_state_w[:, env._ee_idx, 3:7])
joints = env._robot.data.joint_pos[0, env._arm_dof].cpu().numpy()

print("commanded setpoint : {}".format(np.round(canonical[0].cpu().numpy(), 4)))
print("achieved grip point: {}".format(np.round(env._grip_pos()[0].cpu().numpy(), 4)))
print("finger axis        : {}  (0, 0, -1 is straight down)".format(
    np.round(rot[0, :, 2].cpu().numpy(), 3)))
print("joint_pos          : {}".format([round(float(v), 4) for v in joints]))

env.close()
_app.close()
