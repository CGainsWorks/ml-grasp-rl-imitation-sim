"""Isaac Lab port of the lift-and-hold grasp task.

    STATUS: written, reviewed, and NOT executed. There is no Isaac Sim
    installation on the machine this repository was built on, so this file has
    never been run. It is not covered by the tests and it is not the source of
    any number in the README. Treat it as a port that still needs its first
    bring-up, not as a working environment. ``envs/isaac/README.md`` lists what
    to check first.

What is shared with the MuJoCo environment
------------------------------------------
The reward, the success condition and the randomisation ranges are imported
from ``src/`` rather than reimplemented:

    src/rewards/grasp_reward.py       the reward terms and success test
    src/randomisation/configs/*.json  the randomisation ranges

``grasp_reward`` is written against whichever array library it is handed, so
the same function that scores one MuJoCo world scores ``num_envs`` Isaac worlds
on the GPU. ``tests/test_reward_parity.py`` checks numpy and torch agree to
single precision, which is the part of the port that can be tested without a
simulator.

What is necessarily different
-----------------------------
* **Vectorisation.** Isaac steps thousands of environments at once, so
  observations are ``(num_envs, 32)`` and resets are partial: a subset of
  environments resets while the rest keep running.
* **Control.** The MuJoCo scene drags a free-floating hand with a mocap weld.
  Here the same abstraction is a differential inverse-kinematics controller on
  a Franka arm, because Isaac Lab ships that arm and because a floating hand is
  the less honest of the two options once an arm is available. The action space
  stays four-dimensional: three Cartesian deltas plus a gripper command.
* **Randomisation.** Isaac Lab has its own event manager for domain
  randomisation. The ranges are read from the same JSON files so the two
  simulators randomise the same quantities over the same intervals, but the
  mechanism applying them is Isaac's, not ours.

Because the control path differs, a policy trained in MuJoCo is *not* expected
to transfer to this environment unchanged. Cross-simulator transfer is a
separate experiment, listed as unfinished in ``docs/limitations.md``.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch

try:
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, spawners
    from isaaclab.utils import configclass
    from isaaclab.utils.math import subtract_frame_transforms

    ISAAC_AVAILABLE = True
except ImportError:  # pragma: no cover - the expected path outside Isaac Sim
    ISAAC_AVAILABLE = False

    def configclass(cls):  # type: ignore[misc]
        """No-op stand-in so this module imports for inspection without Isaac."""
        return cls

    DirectRLEnv = object  # type: ignore[assignment,misc]
    DirectRLEnvCfg = object  # type: ignore[assignment,misc]

import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.randomisation.domain_rand import CONFIG_DIR  # noqa: E402
from src.rewards.grasp_reward import (  # noqa: E402
    GraspRewardConfig,
    grasp_reward,
    success_condition,
)

TABLE_HEIGHT = 0.40
HOLD_HEIGHT = 0.15
OBS_DIM = 32
ACT_DIM = 4
POS_STEP = 0.02


@configclass
class GraspSceneCfg(InteractiveSceneCfg if ISAAC_AVAILABLE else object):
    """Franka on a table with a single box to lift."""

    if ISAAC_AVAILABLE:
        num_envs: int = 4096
        env_spacing: float = 2.5

        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=spawners.UsdFileCfg(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/IsaacLab/Robots/"
                         "FrankaEmika/panda_instanceable.usd",
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0, "panda_joint2": -0.569, "panda_joint3": 0.0,
                    "panda_joint4": -2.810, "panda_joint5": 0.0, "panda_joint6": 3.037,
                    "panda_joint7": 0.741, "panda_finger_joint.*": 0.04,
                },
            ),
        )

        obj: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=spawners.CuboidCfg(
                size=(0.044, 0.044, 0.044),
                rigid_props=spawners.RigidBodyPropertiesCfg(),
                mass_props=spawners.MassPropertiesCfg(mass=0.08),
                physics_material=spawners.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0
                ),
                visual_material=spawners.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 0.85)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, TABLE_HEIGHT + 0.022)),
        )


@configclass
class GraspTaskCfg(DirectRLEnvCfg if ISAAC_AVAILABLE else object):
    """Task configuration. Episode length matches the MuJoCo env: 100 steps of 40 ms."""

    decimation: int = 2
    episode_length_s: float = 4.0
    action_space: int = ACT_DIM
    observation_space: int = OBS_DIM
    state_space: int = 0
    randomisation_level: str = "medium"

    if ISAAC_AVAILABLE:
        sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
        scene: GraspSceneCfg = GraspSceneCfg()


def load_randomisation_ranges(level: str) -> dict:
    """Read the same JSON ranges the MuJoCo environment uses."""
    path = os.path.join(CONFIG_DIR, "{}.json".format(level))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class GraspTask(DirectRLEnv):  # type: ignore[misc]
    """Direct-workflow Isaac Lab task mirroring ``envs/mujoco/grasp_env.py``."""

    cfg: "GraspTaskCfg"

    def __init__(self, cfg: "GraspTaskCfg", render_mode: str | None = None, **kwargs) -> None:
        if not ISAAC_AVAILABLE:
            raise RuntimeError(
                "Isaac Lab is not installed. This task runs inside Isaac Sim; see "
                "envs/isaac/README.md for the launch command."
            )
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_cfg = GraspRewardConfig()
        self.ranges = load_randomisation_ranges(cfg.randomisation_level)

        self.robot: Articulation = self.scene["robot"]
        self.object: RigidObject = self.scene["obj"]

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=True,
                                        ik_method="dls"),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._ee_idx = self.robot.find_bodies("panda_hand")[0][0]
        self._arm_dof = self.robot.find_joints("panda_joint.*")[0]
        self._finger_dof = self.robot.find_joints("panda_finger_joint.*")[0]

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rest_z = torch.full((self.num_envs,), TABLE_HEIGHT + 0.022,
                                        device=self.device)
        self.last_action = torch.zeros((self.num_envs, ACT_DIM), device=self.device)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_action = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """Cartesian delta through differential IK, plus a gripper position command."""
        ee_pose_w = self.robot.data.body_state_w[:, self._ee_idx, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        self._ik.set_command(self.last_action[:, :3] * POS_STEP, ee_pos_b, ee_quat_b)
        joint_pos = self.robot.data.joint_pos[:, self._arm_dof]
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._ee_idx - 1, :, self._arm_dof]
        target = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(target, joint_ids=self._arm_dof)

        # Gripper: -1 open, +1 closed, mapped onto the Franka finger travel.
        width = (1.0 - self.last_action[:, 3:4]) * 0.5 * 0.04
        self.robot.set_joint_position_target(
            width.repeat(1, len(self._finger_dof)), joint_ids=self._finger_dof
        )

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        grip_pos = self.robot.data.body_state_w[:, self._ee_idx, 0:3] - self.scene.env_origins
        grip_vel = self.robot.data.body_state_w[:, self._ee_idx, 7:10]
        finger_pos = self.robot.data.joint_pos[:, self._finger_dof]
        finger_vel = self.robot.data.joint_vel[:, self._finger_dof]
        width = finger_pos.sum(dim=-1, keepdim=True)
        width_rate = finger_vel.sum(dim=-1, keepdim=True)

        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        obj_quat = self.object.data.root_quat_w
        rot = _quat_to_matrix(obj_quat)
        obj_lin = self.object.data.root_lin_vel_w
        obj_ang = self.object.data.root_ang_vel_w

        obs = torch.cat(
            [
                grip_pos, grip_vel, width, width_rate,
                obj_pos, obj_pos - grip_pos,
                rot[..., 0], rot[..., 1],
                obj_lin, obj_ang,
                self.goal_pos, self.goal_pos - obj_pos,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        grip_pos = self.robot.data.body_state_w[:, self._ee_idx, 0:3] - self.scene.env_origins
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        grasped = self._grasped().float()
        dropped = (obj_pos[:, 2] < TABLE_HEIGHT - 0.06).float()

        reward, _ = grasp_reward(
            grip_pos, obj_pos, self.goal_pos, self.object_rest_z,
            grasped, dropped, self.last_action, self.reward_cfg,
        )
        return reward

    def _get_dones(self) -> tuple:
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        dropped = obj_pos[:, 2] < TABLE_HEIGHT - 0.06
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, time_out

    def _grasped(self) -> torch.Tensor:
        """Contact-based grasp test.

        The MuJoCo environment reads the contact list directly. The Isaac
        equivalent needs a ``ContactSensor`` on each finger; until one is added
        to the scene config this falls back to a geometric test, which is
        weaker and will report a grasp for a pinch that is merely close. That
        is a known gap, listed in envs/isaac/README.md.
        """
        grip_pos = self.robot.data.body_state_w[:, self._ee_idx, 0:3] - self.scene.env_origins
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        near = torch.linalg.norm(obj_pos - grip_pos, dim=-1) < 0.05
        closed = self.robot.data.joint_pos[:, self._finger_dof].sum(dim=-1) < 0.06
        return near & closed

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        n = len(env_ids)
        offsets = torch.zeros((n, 3), device=self.device)
        offsets[:, 0] = 0.5 + torch.empty(n, device=self.device).uniform_(-0.10, 0.10)
        offsets[:, 1] = torch.empty(n, device=self.device).uniform_(-0.12, 0.12)
        offsets[:, 2] = TABLE_HEIGHT + 0.022

        root_state = self.object.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = offsets + self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(root_state, env_ids)

        self.object_rest_z[env_ids] = offsets[:, 2]
        self.goal_pos[env_ids] = torch.stack(
            [offsets[:, 0], offsets[:, 1],
             torch.full((n,), TABLE_HEIGHT + HOLD_HEIGHT, device=self.device)],
            dim=-1,
        )
        self._ik.reset(env_ids)

    def success(self) -> torch.Tensor:
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        return success_condition(obj_pos, self.goal_pos, self._grasped().float(), self.reward_cfg)


def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) quaternions to rotation matrices, batched."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=1,
    )
