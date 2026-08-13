"""Isaac Lab port of the lift-and-hold grasp task.

Brought up against **Isaac Sim 5.1.0 / Isaac Lab 2.3.2** on an RTX 4060. The
bring-up log, and the list of things that were wrong when this file was written
blind against the documentation, are in ``envs/isaac/README.md``.

What is shared with the MuJoCo task
-----------------------------------
The reward, the success condition and the randomisation ranges are imported
rather than reimplemented:

    src/rewards/grasp_reward.py       reward terms, success and drop tests
    src/randomisation/configs/*.json  randomisation ranges per level

``grasp_reward`` dispatches on the array library it is handed, so the same
function that scores one MuJoCo world with numpy scores ``num_envs`` Isaac
worlds with torch on the GPU. ``tests/test_reward_parity.py`` checks the two
backends agree, and ``tests/test_isaac_port.py`` checks this file really does
import them rather than carrying a copy.

What is deliberately different
------------------------------
* **Embodiment.** MuJoCo drives a free-floating hand through a mocap weld.
  Here the same 4-D action drives a Franka Panda through a differential
  inverse-kinematics controller, because Isaac Lab ships the arm and because a
  floating hand is the less honest of the two options once an arm is available.
* **Vectorisation.** ``num_envs`` worlds, partial resets.
* **Timing.** ``sim.dt = 1/200`` with ``decimation = 8`` gives a 0.04 s control
  step and 100 steps per 4 s episode, matching the MuJoCo environment exactly.

Because the control path differs, a policy trained in MuJoCo is *not* expected
to transfer here unchanged. Cross-simulator transfer is a separate experiment
and has not been run.

Usage
-----
The simulation app has to be launched before this module is imported, which is
the normal Isaac Lab pattern::

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    from envs.isaac.grasp_task import GraspTask, GraspTaskCfg

``scripts/isaac_bringup.py`` does exactly that.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Sequence

import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    ISAAC_AVAILABLE = True
except ImportError:  # pragma: no cover - the path taken outside Isaac Sim
    ISAAC_AVAILABLE = False

    def configclass(cls):  # type: ignore[misc]
        """No-op stand-in so this module imports for inspection without Isaac."""
        return cls

    DirectRLEnv = object  # type: ignore[assignment,misc]
    DirectRLEnvCfg = object  # type: ignore[assignment,misc]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.randomisation.domain_rand import CONFIG_DIR  # noqa: E402
from src.rewards.grasp_reward import (  # noqa: E402
    GraspRewardConfig,
    grasp_reward,
    success_condition,
)

TABLE_HEIGHT = 0.40      # top face of the table, metres (matches MuJoCo)
HOLD_HEIGHT = 0.15       # hold point above the table
OBS_DIM = 32
ACT_DIM = 4
POS_STEP = 0.02          # metres per unit action, matches MuJoCo
OBJECT_HALF_SIZE = 0.022
TABLE_CENTRE_X = 0.55    # in front of the Franka base, inside its reach
# Offset from the panda_hand frame to the point between the fingertips. Isaac
# Lab's own Franka lift task uses the same number; without it the controller
# servos the wrist to the box and the fingers close 10 cm short of it.
GRIP_OFFSET = 0.107
# Height of the setpoint above the table at reset. The MuJoCo hand starts
# 11-19 cm above the box; matching that matters more here than it looks,
# because the expert's descent is the phase with the tightest time budget.
RESET_HEIGHT = 0.20
# Workspace box for the commanded *setpoint*, base frame. The floor sits below
# the table top on purpose. The arm's implicit PD holds against gravity with a
# finite stiffness, so the achieved pose lags the setpoint by several
# centimetres; clamping the setpoint at the table top therefore makes it
# impossible for the fingertips to ever reach a box resting on it. The table is
# a real collision body, so a setpoint below it costs nothing -- the fingers
# stop on the surface.
WS_LOW = (0.35, -0.28, TABLE_HEIGHT - 0.03)
WS_HIGH = (0.75, 0.28, TABLE_HEIGHT + 0.35)


def load_randomisation_ranges(level: str) -> dict:
    """Read the same JSON ranges the MuJoCo environment uses."""
    with open(os.path.join(CONFIG_DIR, "{}.json".format(level)), "r", encoding="utf-8") as fh:
        return json.load(fh)


@configclass
class GraspTaskCfg(DirectRLEnvCfg if ISAAC_AVAILABLE else object):
    """Task configuration.

    Assets are fields on the environment config and are instantiated in
    ``_setup_scene``; the scene config itself only carries the vectorisation
    settings. That is the Isaac Lab direct-workflow convention, and getting it
    wrong is the first thing that stopped this file from running.
    """

    decimation = 8
    episode_length_s = 4.0
    action_space = ACT_DIM
    observation_space = OBS_DIM
    state_space = 0
    randomisation_level = "medium"

    if ISAAC_AVAILABLE:
        sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=8)

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=64, env_spacing=3.0, replicate_physics=True
        )

        # The high-PD variant, as Isaac Lab's own lift task uses: the default
        # gains track an IK target too softly to grasp anything.
        #
        # The joint angles are a top-down pre-grasp above the table, solved by
        # driving this environment's own IK to a downward pose and reading the
        # arm back (scripts/isaac_pregrasp.py). The Franka's shipped home pose
        # puts the fingertips at z = 0.383, which is *below* the 0.40 table top:
        # the arm starts inside the table, and on the way out it flicks the box
        # off it.
        robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": -0.039,
                    "panda_joint2": 0.493,
                    "panda_joint3": 0.004,
                    "panda_joint4": -0.324,
                    "panda_joint5": -0.002,
                    "panda_joint6": 0.702,
                    "panda_joint7": 0.751,
                    "panda_finger_joint.*": 0.04,
                },
            ),
        )

        table: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.9, TABLE_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.48)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(TABLE_CENTRE_X, 0.0, TABLE_HEIGHT / 2.0)
            ),
        )

        obj: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.CuboidCfg(
                size=(2 * OBJECT_HALF_SIZE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.55, 0.85)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(TABLE_CENTRE_X, 0.0, TABLE_HEIGHT + OBJECT_HALF_SIZE)
            ),
        )


class GraspTask(DirectRLEnv):  # type: ignore[misc]
    """Direct-workflow Isaac Lab task mirroring ``envs/mujoco/grasp_env.py``."""

    cfg: "GraspTaskCfg"

    def __init__(self, cfg: "GraspTaskCfg", render_mode: str | None = None, **kwargs) -> None:
        if not ISAAC_AVAILABLE:
            raise RuntimeError(
                "Isaac Lab is not installed, or the simulation app has not been "
                "launched yet. See envs/isaac/README.md."
            )
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_cfg = GraspRewardConfig()
        self.ranges = load_randomisation_ranges(cfg.randomisation_level)

        self._ee_idx = self._robot.find_bodies("panda_hand")[0][0]
        self._arm_dof = self._robot.find_joints("panda_joint.*")[0]
        self._finger_dof = self._robot.find_joints("panda_finger_joint.*")[0]
        # A fixed-base articulation has no Jacobian row for the base, so the
        # Jacobian body index is one less than the body index.
        self._jacobi_body_idx = self._ee_idx - 1 if self._robot.is_fixed_base else self._ee_idx

        # Absolute *pose* commands, not relative position ones. The MuJoCo hand
        # cannot rotate -- its orientation is pinned by the weld -- so the Isaac
        # hand is held at a fixed downward orientation instead of being left to
        # drift wherever the arm's null space takes it. Relative-position mode
        # left the gripper at the home pose's 45 degrees, and a 45-degree
        # gripper cannot execute a top-down grasp.
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        # (w, x, y, z): 180 degrees about x, so the hand's local +z (the finger
        # axis) points along world -z.
        self._down_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device).repeat(
            self.num_envs, 1
        )
        self._ws_low = torch.tensor(WS_LOW, device=self.device)
        self._ws_high = torch.tensor(WS_HIGH, device=self.device)
        # Persistent Cartesian setpoint, base frame -- the equivalent of the
        # MuJoCo mocap body. The action moves this target; the IK chases it.
        # Recomputing the target from the *measured* pose each step instead,
        # which is what this file did first, makes a zero action mean "stay
        # wherever gravity has dragged me", and the arm sags out of the
        # workspace within a couple of seconds.
        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rest_z = torch.full(
            (self.num_envs,), TABLE_HEIGHT + OBJECT_HALF_SIZE, device=self.device
        )
        self.last_action = torch.zeros((self.num_envs, ACT_DIM), device=self.device)

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        self._object = RigidObject(self.cfg.obj)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["object"] = self._object

        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_action = actions.clamp(-1.0, 1.0).to(self.device)
        # Integrate once per control step, not once per physics substep.
        self._target_pos = torch.clamp(
            self._target_pos + self.last_action[:, :3] * POS_STEP,
            self._ws_low,
            self._ws_high,
        )

    def _apply_action(self) -> None:
        """Cartesian delta through differential IK, plus a gripper command."""
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        ee_pose_w = self._robot.data.body_state_w[:, self._ee_idx, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # The setpoint is for the grip point; the IK controls the hand frame, so
        # subtract the fixed downward finger offset to get the hand target.
        hand_target = self._target_pos.clone()
        hand_target[:, 2] += GRIP_OFFSET
        self._ik.set_command(torch.cat([hand_target, self._down_quat], dim=-1))
        joint_pos = self._robot.data.joint_pos[:, self._arm_dof]
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._arm_dof
        ]
        target = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self._robot.set_joint_position_target(target, joint_ids=self._arm_dof)

        # Gripper: -1 open, +1 closed, over the Franka finger travel (0 .. 0.04).
        width = (1.0 - self.last_action[:, 3:4]) * 0.5 * 0.04
        self._robot.set_joint_position_target(
            width.repeat(1, len(self._finger_dof)), joint_ids=self._finger_dof
        )

    # ------------------------------------------------------------------
    def _grip_pos(self) -> torch.Tensor:
        """The point between the fingertips, which is what MuJoCo's grip site is."""
        ee_pose_w = self._robot.data.body_state_w[:, self._ee_idx, 0:7]
        rot = _quat_to_matrix(ee_pose_w[:, 3:7])
        forward = rot[..., 2] * GRIP_OFFSET
        return ee_pose_w[:, 0:3] + forward - self.scene.env_origins

    def _gripper_width(self) -> torch.Tensor:
        return self._robot.data.joint_pos[:, self._finger_dof].sum(dim=-1, keepdim=True)

    def _object_pos(self) -> torch.Tensor:
        return self._object.data.root_pos_w - self.scene.env_origins

    def _grasped(self) -> torch.Tensor:
        """Geometric grasp test: object close to the grip point and fingers closed.

        Weaker than the MuJoCo version, which reads the contact list. Replacing
        this with a ``ContactSensor`` on each finger is the first item in
        envs/isaac/README.md.
        """
        near = torch.linalg.norm(self._object_pos() - self._grip_pos(), dim=-1) < 0.06
        closed = self._gripper_width().squeeze(-1) < 0.07
        return (near & closed).float()

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        grip_pos = self._grip_pos()
        grip_vel = self._robot.data.body_state_w[:, self._ee_idx, 7:10]
        width = self._gripper_width()
        width_rate = self._robot.data.joint_vel[:, self._finger_dof].sum(dim=-1, keepdim=True)

        obj_pos = self._object_pos()
        rot = _quat_to_matrix(self._object.data.root_quat_w)
        obj_lin = self._object.data.root_lin_vel_w
        obj_ang = self._object.data.root_ang_vel_w

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
        obj_pos = self._object_pos()
        dropped = (obj_pos[:, 2] < TABLE_HEIGHT - 0.06).float()
        reward, _ = grasp_reward(
            self._grip_pos(), obj_pos, self.goal_pos, self.object_rest_z,
            self._grasped(), dropped, self.last_action, self.reward_cfg,
        )
        return reward

    def _get_dones(self) -> tuple:
        dropped = self._object_pos()[:, 2] < TABLE_HEIGHT - 0.06
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, time_out

    def success(self) -> torch.Tensor:
        return success_condition(
            self._object_pos(), self.goal_pos, self._grasped(), self.reward_cfg
        )

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        offsets = torch.zeros((n, 3), device=self.device)
        offsets[:, 0] = TABLE_CENTRE_X + torch.empty(n, device=self.device).uniform_(-0.08, 0.08)
        offsets[:, 1] = torch.empty(n, device=self.device).uniform_(-0.10, 0.10)
        offsets[:, 2] = TABLE_HEIGHT + OBJECT_HALF_SIZE

        root_state = self._object.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = offsets + self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        self._object.write_root_state_to_sim(root_state, env_ids)

        self.object_rest_z[env_ids] = offsets[:, 2]
        self.goal_pos[env_ids] = torch.stack(
            [
                offsets[:, 0],
                offsets[:, 1],
                torch.full((n,), TABLE_HEIGHT + HOLD_HEIGHT, device=self.device),
            ],
            dim=-1,
        )
        self._ik.reset(env_ids)
        # Start the setpoint above the box, as the MuJoCo hand does.
        self._target_pos[env_ids] = torch.stack(
            [
                offsets[:, 0] + torch.empty(n, device=self.device).uniform_(-0.04, 0.04),
                offsets[:, 1] + torch.empty(n, device=self.device).uniform_(-0.04, 0.04),
                torch.full((n,), TABLE_HEIGHT + RESET_HEIGHT, device=self.device),
            ],
            dim=-1,
        )


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
