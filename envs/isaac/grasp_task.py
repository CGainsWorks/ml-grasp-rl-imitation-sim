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
import math
import os
import sys
from typing import Sequence

import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
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
from src.rewards.place_reward import (  # noqa: E402
    PlaceRewardConfig,
    place_reward,
    place_success_condition,
)
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
# Pick-and-place target placement, matching envs/mujoco/grasp_env.py.
PLACE_TARGET_X = 0.15
PLACE_TARGET_Y = 0.18
PLACE_MIN_TRAVEL = 0.12
PLACE_MAX_TRAVEL = 0.30
TABLE_CENTRE_X = 0.48    # in front of the Franka base, inside its comfortable reach
# Offset from the panda_hand frame to the point between the fingertips. Isaac
# Lab's own Franka lift task uses the same number; without it the controller
# servos the wrist to the box and the fingers close 10 cm short of it.
GRIP_OFFSET = 0.107
# Height of the setpoint above the table at reset. The MuJoCo hand starts
# 11-19 cm above the box; matching that matters more here than it looks,
# because the expert's descent is the phase with the tightest time budget.
RESET_HEIGHT = 0.20
# Newtons. Above sensor noise, well below the force needed to hold the box.
CONTACT_FORCE_THRESHOLD = 0.05
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


def _scaled(spec: dict, scale: float) -> tuple:
    """Turn one shared range spec into an Isaac (low, high) tuple.

    Reproduces ``ParamSpec.sample`` from ``src/randomisation/domain_rand.py``:
    multiplicative ranges are widened about 1.0 by the level's scale, additive
    ones about their midpoint. Keeping the arithmetic identical is the point --
    both simulators then randomise over the same interval.
    """
    if spec.get("mode", "scale") == "scale":
        return (1.0 + (spec["low"] - 1.0) * scale, 1.0 + (spec["high"] - 1.0) * scale)
    mid = 0.5 * (spec["low"] + spec["high"])
    half = 0.5 * (spec["high"] - spec["low"]) * scale
    return (mid - half, mid + half)


def build_events(level: str):
    """Build Isaac Lab event terms from the shared randomisation config.

    Mapped through Isaac's own event manager: object mass, object and table
    friction, and gripper actuator gains. Gravity is applied per-scene rather
    than per-environment, so it is left at nominal. ``hand_compliance`` has no
    Isaac analogue (it is a property of the MuJoCo weld), ``action_latency``
    would need a command queue in this class, and ``object_half_size`` needs a
    pre-startup scale term; those four are not mapped and are listed as such in
    envs/isaac/README.md. Sensing and action noise are applied through the
    environment's noise models rather than events -- see ``GraspTaskCfg``.
    """
    if not ISAAC_AVAILABLE:
        return None
    raw = load_randomisation_ranges(level)
    scale, params = float(raw["scale"]), raw.get("params", {})

    @configclass
    class EventCfg:
        pass

    terms = {}
    if scale > 0.0 and "object_mass" in params:
        terms["object_mass"] = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": _scaled(params["object_mass"], scale),
                "operation": "scale",
            },
        )
    if scale > 0.0 and "object_friction" in params:
        low, high = _scaled(params["object_friction"], scale)
        terms["object_material"] = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (low, high),
                "dynamic_friction_range": (low, high),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
    if scale > 0.0 and "table_friction" in params:
        low, high = _scaled(params["table_friction"], scale)
        terms["table_material"] = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table"),
                "static_friction_range": (low, high),
                "dynamic_friction_range": (low, high),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
    if scale > 0.0 and "object_half_size" in params:
        # Scale terms have to run before the simulation starts: geometry cannot
        # be resized once physics is initialised.
        low, high = _scaled(params["object_half_size"], scale)
        terms["object_scale"] = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "scale_range": {"x": (low, high), "y": (low, high), "z": (low, high)},
            },
        )
    if scale > 0.0 and "gravity" in params:
        # Isaac applies gravity per scene rather than per environment, so this
        # draws one value for all of them each interval. MuJoCo draws one per
        # episode per world; the interval is the same, the granularity is not.
        low, high = _scaled(params["gravity"], scale)
        terms["gravity"] = EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="interval",
            is_global_time=True,
            interval_range_s=(4.0, 4.0),
            params={
                "gravity_distribution_params": ([0.0, 0.0, -9.81 * low],
                                                [0.0, 0.0, -9.81 * high]),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
    if scale > 0.0 and "hand_compliance" in params:
        # MuJoCo's hand_compliance is the solref of the weld that drags the
        # hand: how softly the hand follows its setpoint. There is no weld here,
        # and the closest honest analogue is the arm's joint stiffness, which
        # governs exactly the same thing -- how hard the arm insists on reaching
        # the pose it was told to. The mapping is inverted: a *more* compliant
        # weld is a *less* stiff arm.
        low, high = _scaled(params["hand_compliance"], scale)
        terms["arm_compliance"] = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
                "stiffness_distribution_params": (1.0 / max(high, 1e-6),
                                                  1.0 / max(low, 1e-6)),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
    if scale > 0.0 and "gripper_gain" in params:
        terms["gripper_gain"] = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
                "stiffness_distribution_params": _scaled(params["gripper_gain"], scale),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    for name, term in terms.items():
        setattr(EventCfg, name, term)
        EventCfg.__annotations__[name] = EventTerm
    return EventCfg()


def build_noise_models(level: str):
    """Action noise, from the same shared ranges.

    Observation noise is deliberately *not* returned as an Isaac noise model.
    Isaac's model perturbs every element of the vector independently, which
    breaks an invariant the MuJoCo environment maintains: a real system has one
    pose estimate, and the relative-position entries are computed from it, so
    they stay consistent with the absolute ones. ``_get_observations`` applies
    the noise the same way MuJoCo does instead.
    """
    if not ISAAC_AVAILABLE:
        return None, None
    raw = load_randomisation_ranges(level)
    scale, params = float(raw["scale"]), raw.get("params", {})
    if scale <= 0.0:
        return None, None

    obs_model = act_model = None
    if "action_noise" in params:
        _, high = _scaled(params["action_noise"], scale)
        act_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=max(high, 1e-6), operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=max(high * 0.25, 1e-6),
                                            operation="add"),
        )
    return obs_model, act_model


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
    # "lift" or "place". The second task shares this file rather than forking it,
    # for the same reason the reward is shared: two copies of a task definition
    # drift, and "it is the same task in both simulators" then stops meaning
    # anything. See src/rewards/place_reward.py.
    task = "lift"

    def __post_init__(self):
        """Attach the randomisation, driven by the shared JSON ranges."""
        if ISAAC_AVAILABLE:
            self.events = build_events(self.randomisation_level)
            _, act_noise = build_noise_models(self.randomisation_level)
            self.action_noise_model = act_noise

    if ISAAC_AVAILABLE:
        sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=8)

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=64, env_spacing=3.0, replicate_physics=True
        )

        # The high-PD variant, as Isaac Lab's own lift task uses: the default
        # gains track an IK target too softly to grasp anything.
        #
        # The joint angles are a top-down pre-grasp above the table, found by
        # searching configurations with the simulator's own forward kinematics
        # (scripts/isaac_pregrasp.py). Two poses were rejected first: the
        # Franka's shipped home pose puts the fingertips at z = 0.383, *below*
        # the 0.40 m table top, so the arm starts inside the table and flicks
        # the box off it on the way out; and a pose obtained by letting the IK
        # settle from that start left the elbow nearly straight
        # (panda_joint4 = -0.32 against a -0.07 limit), which is close enough
        # to a singularity that the damped-least-squares solver oscillated
        # between two configurations instead of converging. This one keeps
        # joint 4 at -1.42, clear of both limits.
        robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.1,
                    "panda_joint3": 0.0,
                    "panda_joint4": -1.417,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.367,
                    "panda_joint7": 0.785,
                    "panda_finger_joint.*": 0.04,
                },
            ),
        )

        # One contact sensor per finger, filtered to the box. This is what makes
        # the grasp test a real contact read rather than "the box is nearby and
        # the fingers are shut", which reports a grasp for a near miss.
        contact_left: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
            update_period=0.0,
            history_length=0,
        )
        contact_right: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
            update_period=0.0,
            history_length=0,
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
        # Rebuild the randomisation from the level actually on the config.
        # `__post_init__` fires when GraspTaskCfg() is constructed, which is
        # before a caller can set `randomisation_level` -- so relying on it
        # alone silently ran every level with the default's ranges, including
        # `none`. The bring-up check for "randomisation is inert at level
        # 'none'" is what caught it.
        cfg.events = build_events(cfg.randomisation_level)
        _, act_noise = build_noise_models(cfg.randomisation_level)
        cfg.action_noise_model = act_noise

        # Randomising object *scale* is a USD-level edit, and Isaac refuses to
        # combine that with scene replication: replicated instances share their
        # properties, so a per-environment size would silently apply to all of
        # them. Replication is therefore switched off exactly when the level
        # randomises size -- it costs scene-setup time, and the alternative is
        # a randomisation that lies.
        if cfg.events is not None and hasattr(cfg.events, "object_scale"):
            cfg.scene.replicate_physics = False

        super().__init__(cfg, render_mode, **kwargs)

        self.task = getattr(cfg, "task", "lift")
        self.place = self.task == "place"
        self.reward_cfg = PlaceRewardConfig() if self.place else GraspRewardConfig()
        self.ranges = load_randomisation_ranges(cfg.randomisation_level)
        # Sensing noise, applied exactly as the MuJoCo environment applies it.
        obs_params = self.ranges.get("params", {})
        rand_scale = float(self.ranges["scale"])
        self._obs_noise_pos = 0.0
        self._obs_noise_vel = 0.0
        if rand_scale > 0.0 and "obs_noise_pos" in obs_params:
            self._obs_noise_pos = _scaled(obs_params["obs_noise_pos"], rand_scale)[1]
        if rand_scale > 0.0 and "obs_noise_vel" in obs_params:
            self._obs_noise_vel = _scaled(obs_params["obs_noise_vel"], rand_scale)[1]

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

        # Per-environment command queue, the equivalent of the MuJoCo latency
        # deque. Row 0 is the newest command; an environment with latency k
        # executes row k, so the queue depth is the largest latency the level
        # can draw.
        latency_spec = self.ranges.get("params", {}).get("action_latency")
        rand_scale_l = float(self.ranges["scale"])
        if latency_spec is not None and rand_scale_l > 0.0:
            self._max_latency = int(round(_scaled(latency_spec, rand_scale_l)[1]))
        else:
            self._max_latency = 0
        self._latency_range = (
            _scaled(latency_spec, rand_scale_l)
            if latency_spec is not None and rand_scale_l > 0.0
            else (0.0, 0.0)
        )
        self._action_queue = torch.zeros(
            (self.num_envs, self._max_latency + 1, ACT_DIM), device=self.device
        )
        self._latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rest_z = torch.full(
            (self.num_envs,), TABLE_HEIGHT + OBJECT_HALF_SIZE, device=self.device
        )
        self.last_action = torch.zeros((self.num_envs, ACT_DIM), device=self.device)
        # Pick-and-place state. `lifted` is a per-episode latch, set once the
        # object clears the table while grasped and never cleared, which is what
        # stops a policy from sliding the box to the target and scoring.
        self.lifted = torch.zeros(self.num_envs, device=self.device)
        self.object_start = torch.zeros((self.num_envs, 3), device=self.device)

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        # Contact reporting has to be switched on when the robot is spawned,
        # not when the sensor is created.
        self.cfg.robot.spawn.activate_contact_sensors = True

        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        self._object = RigidObject(self.cfg.obj)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["object"] = self._object

        self._contact_left = ContactSensor(self.cfg.contact_left)
        self._contact_right = ContactSensor(self.cfg.contact_right)
        self.scene.sensors["contact_left"] = self._contact_left
        self.scene.sensors["contact_right"] = self._contact_right

        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        commanded = actions.clamp(-1.0, 1.0).to(self.device)
        if self._max_latency > 0:
            self._action_queue = torch.roll(self._action_queue, shifts=1, dims=1)
            self._action_queue[:, 0] = commanded
            index = self._latency.view(-1, 1, 1).expand(-1, 1, ACT_DIM)
            commanded = torch.gather(self._action_queue, 1, index).squeeze(1)
        self.last_action = commanded
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
        """Both fingers in contact with the box, read from the contact sensors.

        The MuJoCo environment reads its contact list for exactly this reason: a
        policy that closes on thin air must not be told it has a grasp. The
        filtered force matrix is ``(num_envs, bodies, filters, 3)``; there is one
        body and one filter here, so the norm over the last axis is the force
        between that finger and the box.
        """
        left = self._contact_left.data.force_matrix_w
        right = self._contact_right.data.force_matrix_w
        if left is None or right is None:  # sensor not ready on the first frame
            return torch.zeros(self.num_envs, device=self.device)
        left_touch = torch.linalg.norm(left.view(self.num_envs, -1, 3), dim=-1).max(dim=1).values
        right_touch = torch.linalg.norm(right.view(self.num_envs, -1, 3), dim=-1).max(dim=1).values
        return ((left_touch > CONTACT_FORCE_THRESHOLD)
                & (right_touch > CONTACT_FORCE_THRESHOLD)).float()

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

        if self._obs_noise_pos > 0.0:
            grip_pos = grip_pos + torch.randn_like(grip_pos) * self._obs_noise_pos
            obj_pos = obj_pos + torch.randn_like(obj_pos) * self._obs_noise_pos
        if self._obs_noise_vel > 0.0:
            grip_vel = grip_vel + torch.randn_like(grip_vel) * self._obs_noise_vel
            obj_lin = obj_lin + torch.randn_like(obj_lin) * self._obs_noise_vel

        # Relative entries are derived *after* noising, so the observation stays
        # internally consistent: one noisy pose estimate, not two.
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

    def _object_speed(self) -> torch.Tensor:
        return torch.linalg.norm(
            self._object.data.root_lin_vel_w, dim=-1)

    def _update_lift_latch(self, obj_pos: torch.Tensor,
                           grasped: torch.Tensor) -> None:
        """Read from the simulator's true height, never from the observation.

        Sensing noise must not be able to hand a policy credit for a lift that
        did not happen, which is the same rule the MuJoCo environment follows.
        """
        clear = obj_pos[:, 2] - self.object_rest_z
        picked = (grasped > 0.5) & (clear >= self.reward_cfg.lift_threshold)
        self.lifted = torch.where(picked, torch.ones_like(self.lifted),
                                  self.lifted)

    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._object_pos()
        dropped = (obj_pos[:, 2] < TABLE_HEIGHT - 0.06).float()
        grasped = self._grasped()
        if self.place:
            self._update_lift_latch(obj_pos, grasped)
            reward, _ = place_reward(
                self._grip_pos(), obj_pos, self.goal_pos, self.object_start,
                self.object_rest_z, grasped, self.lifted, dropped,
                self._object_speed(), self.last_action, self.reward_cfg,
            )
            return reward
        reward, _ = grasp_reward(
            self._grip_pos(), obj_pos, self.goal_pos, self.object_rest_z,
            grasped, dropped, self.last_action, self.reward_cfg,
        )
        return reward

    def _get_dones(self) -> tuple:
        dropped = self._object_pos()[:, 2] < TABLE_HEIGHT - 0.06
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, time_out

    def success(self) -> torch.Tensor:
        if self.place:
            return place_success_condition(
                self._object_pos(), self.goal_pos, self._grasped(), self.lifted,
                self._object_speed(), self.reward_cfg,
            )
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
        self.object_start[env_ids] = offsets
        self.lifted[env_ids] = 0.0
        if self.place:
            # Rejection sampling would need a per-environment loop on the GPU,
            # so the target is drawn as a bearing plus a radius inside the
            # allowed band and then clipped to the table. Clipping biases the
            # distribution towards the edges, which the MuJoCo version avoids by
            # rejecting -- the band is well inside the table here, so the clip
            # almost never binds, and envs/isaac/README.md records the
            # difference rather than implying the two are identical.
            bearing = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
            radius = torch.empty(n, device=self.device).uniform_(
                PLACE_MIN_TRAVEL, PLACE_MAX_TRAVEL)
            self.goal_pos[env_ids] = torch.stack(
                [
                    (offsets[:, 0] + radius * torch.cos(bearing)).clamp(
                        TABLE_CENTRE_X - PLACE_TARGET_X,
                        TABLE_CENTRE_X + PLACE_TARGET_X),
                    (offsets[:, 1] + radius * torch.sin(bearing)).clamp(
                        -PLACE_TARGET_Y, PLACE_TARGET_Y),
                    offsets[:, 2],
                ],
                dim=-1,
            )
        else:
            self.goal_pos[env_ids] = torch.stack(
                [
                    offsets[:, 0],
                    offsets[:, 1],
                    torch.full((n,), TABLE_HEIGHT + HOLD_HEIGHT, device=self.device),
                ],
                dim=-1,
            )
        self._ik.reset(env_ids)
        if self._max_latency > 0:
            self._action_queue[env_ids] = 0.0
            low, high = self._latency_range
            self._latency[env_ids] = torch.randint(
                int(round(low)), int(round(high)) + 1, (n,), device=self.device
            )
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
