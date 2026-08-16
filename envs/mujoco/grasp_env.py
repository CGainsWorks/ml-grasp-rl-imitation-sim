"""MuJoCo grasp environment: lift-and-hold, and pick-and-place.

Task
----
A parallel-jaw hand starts above a table. A box sits somewhere on the table at
a random position and yaw.

``task="lift"`` -- the default, and what every headline number in this
repository was produced with: close on the box, lift it to a hold point 0.15 m
above the table, and still be holding it there when the episode ends. Success is
read at the **final** step, so letting go early scores zero.

``task="place"`` -- carry the box to a target patch elsewhere on the table and
**let go of it there**. Success is read at the final step too, and requires the
object to have been picked up rather than slid across. Same observation, same
action space, a different goal placement and a different reward; the reasoning
is in ``src/rewards/place_reward.py``. It exists because a reward design
validated on exactly one task is not evidence about the design *method*, which
is what ``docs/limitations.md`` said before this task was added.

Control
-------
The hand is a free body welded to a mocap body. The action commands a Cartesian
displacement of that mocap target plus a gripper opening; the weld constraint
drags the hand towards the target and the solver resolves any contact on the
way. Nothing is teleported: a badly aligned finger pair pushes the box away
instead of passing through it. This is the standard mocap-driven end-effector
abstraction, chosen because it keeps the action space four-dimensional without
pretending that contact is kinematic.

There is no arm. The hand is free-floating. That is a real limitation and it is
stated in ``docs/limitations.md``: joint limits, self-collision and arm inertia
are all absent, so a policy trained here would need a reachability check before
it went anywhere near hardware.

Observation (32 dimensions)
---------------------------
=====  ====  ==========================================================
Index  Size  Quantity
=====  ====  ==========================================================
0:3     3    grip site position, world frame
3:6     3    grip site linear velocity
6:7     1    gripper opening (distance between the pads)
7:8     1    gripper opening rate
8:11    3    object position
11:14   3    object position relative to the grip site
14:20   6    object orientation, first two columns of the rotation matrix
20:23   3    object linear velocity
23:26   3    object angular velocity
26:29   3    hold point position
29:32   3    hold point relative to the object
=====  ====  ==========================================================

A 6-D rotation representation is used rather than a quaternion because it is
continuous, which matters when a behaviour-cloning network has to regress
through it.

Action (4 dimensions, all in [-1, 1])
-------------------------------------
=====  ==========================================================
Index  Meaning
=====  ==========================================================
0:3    mocap displacement in x, y, z, scaled by ``pos_step`` (2 cm)
3      gripper command: -1 fully open, +1 fully closed
=====  ==========================================================
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

try:  # gymnasium is optional: the env works standalone, the API is the same
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
except Exception:  # pragma: no cover - exercised only where gymnasium is absent
    gym = None
    spaces = None
    _BASE = object

import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.randomisation.domain_rand import (  # noqa: E402
    RandomisationConfig,
    SampledWorld,
    load_randomisation,
    sample_world,
)
from src.rewards.grasp_reward import (  # noqa: E402
    GraspRewardConfig,
    dropped_condition,
    grasp_reward,
    success_condition,
)
from src.rewards.place_reward import (  # noqa: E402
    PlaceRewardConfig,
    place_reward,
    place_success_condition,
)

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "grasp_scene.xml")
ARM_SCENE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "grasp_scene_arm.xml")
# Damped least squares for the arm variant's IK. The damping is what makes a
# singular configuration produce a small motion instead of an enormous one --

# the mocap weld cannot. Two iterations per control step is enough at a 2 cm
# command; more just tracks the setpoint more stiffly than a real servo would.
IK_DAMPING = 0.08
IK_ITERS = 20
# Attempts allowed when placing the arm at reset. IK is collision-blind, and
# reaching across a table it regularly returns a pose that folds the arm through
# it; the fix is not a cleverer solver but retrying from a different starting
# configuration and keeping the first solution that is actually contact-free.
#
# A nullspace posture bias was tried first and is mathematically vacuous here:
# six joints against a six-dimensional pose target leaves no redundancy, so
# I - J+J is zero except at singularities. It cost an hour and is recorded so
# the next person does not repeat it.
IK_RESET_ATTEMPTS = 120

TABLE_HEIGHT = 0.40           # top face of the table, metres
GRIPPER_OPEN_WIDTH = 0.078    # pad-to-pad gap with both slide joints at zero
GRASP_DEBOUNCE_STEPS = 3      # environment steps the grasp flag stays latched
OBS_DIM = 32
ACT_DIM = 4

# The wrist variant. Adding a yaw degree of freedom is the change
# docs/limitations.md calls the single biggest realism gain available here: it
# turns "close the fingers" into "align, then close", which is where the real
# difficulty in grasping lives. It needs no MJCF change -- the weld already
# constrains the hand's orientation to the mocap body's, and this environment
# simply never commanded it.
#
# The dimensions differ, so every policy, demonstration file and results table
# from the four-dimensional task is incompatible with this one. That is why it
# is a flag and not a replacement: the existing numbers stay valid and the two
# tasks are reported separately.
WRIST_OBS_DIM = 34            # + sin, cos of the wrist angle
WRIST_ACT_DIM = 5             # + yaw rate
WRIST_LIMIT = np.pi / 2       # +/- 90 degrees is enough: a box repeats every 90
WRIST_STEP = np.deg2rad(9.0)  # per control step at full command
# With a wrist the size cap can rise: the pads no longer have to swallow the
# diagonal of a badly-yawed box. 34 mm half-size is 68 mm across a face against
# a 78 mm gap, and 96 mm across the diagonal -- comfortably graspable aligned,
# impossible misaligned, which is exactly the regime the wrist exists for.
WRIST_MAX_HALF_SIZE = 0.034

# The pick-and-place variant. See src/rewards/place_reward.py for why a second
# task exists at all and what it is meant to break.
#
# The observation and action spaces are *identical* to the lift task: the hold
# point already occupies indices 26:29, so moving it onto the table is a change
# of where the goal is, not of what the policy can see. Every network shape,
# demonstration file and training script carries over.
#
# Where the target may be placed, and how far it must be from the object. The
# lower bound is what makes this a transport task rather than a lift with extra
# steps -- at 5 cm the goal tolerance and the start position overlap. The upper
# bound keeps the pair inside the workspace with room for the hand.
PLACE_TARGET_X = 0.15
PLACE_TARGET_Y = 0.18
PLACE_MIN_TRAVEL = 0.12
PLACE_MAX_TRAVEL = 0.30
TASKS = ("lift", "place")


class GraspEnv(_BASE):
    """Gymnasium-style environment. Also usable without gymnasium installed."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        reward_cfg: Optional[GraspRewardConfig] = None,
        randomisation: str = "none",
        max_steps: int = 100,
        n_substeps: int = 20,
        pos_step: float = 0.02,
        hold_height: float = 0.15,
        render_mode: Optional[str] = None,
        camera: str = "scene_cam",
        width: int = 480,
        height: int = 360,
        seed: Optional[int] = None,
        wrist: bool = False,
        arm: bool = False,
        max_half_size: Optional[float] = None,
        task: str = "lift",
    ) -> None:
        if task not in TASKS:
            raise ValueError("task must be one of {}, got {!r}".format(TASKS, task))
        self.task = task
        self.place = task == "place"
        self.arm = bool(arm)
        self.wrist = bool(wrist)
        # Overridable so the wrist can be ablated properly: the same box
        # distribution has to be presented to a hand that can rotate and one
        # that cannot, or the comparison is between two different tasks.
        self._max_half_size = float(
            max_half_size if max_half_size is not None
            else (WRIST_MAX_HALF_SIZE if self.wrist else 0.024))
        self.obs_dim = WRIST_OBS_DIM if self.wrist else OBS_DIM
        self.act_dim = WRIST_ACT_DIM if self.wrist else ACT_DIM
        self.model = mujoco.MjModel.from_xml_path(
            ARM_SCENE_PATH if self.arm else SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        self.reward_cfg = reward_cfg or (
            PlaceRewardConfig() if self.place else GraspRewardConfig())
        if self.place and not isinstance(self.reward_cfg, PlaceRewardConfig):
            raise TypeError("the place task needs a PlaceRewardConfig, got "
                            + type(self.reward_cfg).__name__)
        self.rand_cfg: RandomisationConfig = (
            randomisation
            if isinstance(randomisation, RandomisationConfig)
            else load_randomisation(randomisation)
        )
        self.max_steps = int(max_steps)
        self.n_substeps = int(n_substeps)
        self.pos_step = float(pos_step)
        self.hold_height = float(hold_height)
        self.render_mode = render_mode
        self.camera = camera
        self._render_size = (height, width)
        self._renderer: Optional[Any] = None

        self.np_random = np.random.default_rng(seed)

        # Cached model handles
        # The arm variant has no mocap body: the hand is bolted to a flange and
        # driven through IK instead of dragged by a weld.
        self._mocap_id = (
            None if self.arm else self.model.body("mocap").mocapid[0])
        self._object_bid = self.model.body("object").id
        self._object_gid = self.model.geom("object").id
        self._table_gid = self.model.geom("table_top").id
        self._grip_sid = self.model.site("grip_site").id
        if self.arm:
            names = ["j1", "j2", "j3", "j4", "j5", "j6"]
            self._arm_qpos = np.array(
                [self.model.jnt_qposadr[self.model.joint(n).id] for n in names])
            self._arm_dofs = np.array(
                [self.model.jnt_dofadr[self.model.joint(n).id] for n in names])
            self._arm_ctrl = np.array(
                [self.model.actuator(a).id for a in ["a1", "a2", "a3", "a4", "a5", "a6"]])
            self._grip_ctrl = np.array(
                [self.model.actuator(a).id for a in ["left_drive", "right_drive"]])
            # The orientation the pads should hold, stated rather than read off
            # the model's home configuration. Home is the arm standing straight
            # up, where the hand's orientation is an accident of the chain and
            # has nothing to do with grasping; using it as the IK target asked
            # the arm to hold the pads facing the ceiling. This is the frame the
            # weld version holds by construction: grip z down the table normal.
            mujoco.mj_kinematics(self.model, self.data)
            # Identity, which is what the weld version holds by construction:
            # the hand's local +z points *up*, the palm sits above it at +22 mm
            # and the pads hang below at -46 mm, so the fingers face the table.
            # Setting this to z-down instead -- the intuitive reading of "pads
            # facing the table" -- turns the hand over, and IK then solves the
            # grip site to the right height with the palm 4 cm inside the table.
            self._rest_frame = np.eye(3)
            self._arm_home = self.data.qpos[self._arm_qpos].copy()
            # The posture the nullspace term pulls towards. Not hand-picked:
            # hand-picked ones were collision-free but sat a metre above the
            # workspace, so the bias fought the Cartesian target and made both
            # reach and penetration worse. This one comes from a search over
            # 400 000 random configurations, keeping those that are contact-free,
            # near the middle of the workspace, and hold the pads facing down --
            # the same search-don't-guess approach scripts/isaac_pregrasp.py uses
            # for the Franka's start pose.
            self._arm_posture = np.array(
                [-1.5730, -0.9058, -2.1502, 2.4167, -0.8003, 0.2837])
            self._arm_gain0 = self.model.actuator_gainprm[self._arm_ctrl, 0].copy()
            self._arm_target = np.zeros(3)
            self._arm_placements = 0
            self._arm_place_failures = 0
            self._arm_last_good = None
            self._start_pool = []
            self._build_start_pool()
        else:
            self._grip_ctrl = np.arange(self.model.nu)
        self._goal_sid = self.model.site("goal").id
        if self.place:
            # Draw the target as a flat patch on the table rather than the lift
            # task's floating sphere, because that is what it is: somewhere to
            # put something down. The radius is exactly the success tolerance,
            # so anyone watching a rollout video can see for themselves whether
            # the box finished inside it.
            self.model.site_type[self._goal_sid] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.site_size[self._goal_sid] = np.array(
                [self.reward_cfg.goal_tolerance, 0.001, 0.0])
            self.model.site_rgba[self._goal_sid] = np.array([0.95, 0.62, 0.15, 0.65])
        self._pad_gids = (self.model.geom("left_pad").id, self.model.geom("right_pad").id)
        self._object_qadr = self.model.jnt_qposadr[self.model.joint("object_free").id]
        # The arm variant's hand has no free joint: it is a link in a chain.
        self._hand_qadr = (
            None if self.arm
            else self.model.jnt_qposadr[self.model.joint("hand_free").id])
        self._finger_qadr = (
            self.model.jnt_qposadr[self.model.joint("left_slide").id],
            self.model.jnt_qposadr[self.model.joint("right_slide").id],
        )
        self._finger_vadr = (
            self.model.jnt_dofadr[self.model.joint("left_slide").id],
            self.model.jnt_dofadr[self.model.joint("right_slide").id],
        )
        self._grip_range = (
            float(self.model.actuator("left_drive").ctrlrange[0]),
            float(self.model.actuator("left_drive").ctrlrange[1]),
        )
        # The hand weld is the only equality constraint, and the arm variant
        # does not have one.
        self._weld_id = None if self.arm else 0

        # Workspace box for the mocap target. The lower z bound keeps the pads
        # about 4 mm clear of the table so the hand cannot wedge itself under it.
        self._ws_low = np.array([-0.20, -0.28, 0.462])
        self._ws_high = np.array([0.20, 0.28, 0.72])

        self.world: SampledWorld = sample_world(
            self.rand_cfg, self.np_random, self._max_half_size)
        self._latency_queue: deque = deque()
        self._steps = 0
        self._goal = np.zeros(3)
        self._object_start = np.zeros(3)
        self._lifted = 0.0
        self._grasp_latch = 0
        self._object_rest_z = TABLE_HEIGHT
        self._prev_grip = np.zeros(3)
        self._last_terms: Dict[str, float] = {}
        self._wrist_yaw = 0.0
        self._noise_state: Dict[str, np.ndarray] = {}

        if spaces is not None:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (self.obs_dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)

    # ------------------------------------------------------------------
    # Model mutation from the sampled world
    # ------------------------------------------------------------------
    def _apply_world(self, world: SampledWorld) -> None:
        """Push sampled parameters into the compiled model.

        Editing ``MjModel`` in place is what makes per-episode randomisation
        cheap: no recompile, no reallocation of ``MjData``.
        """
        hs = world.object_half_size
        # Shape, as a runtime edit of the geom rather than a second scene file.
        # The size vector means different things per type, which is the whole
        # reason this is a switch and not a scale: for a box it is three
        # half-extents, for a cylinder a radius and a half-height, for a sphere
        # a radius alone. All three are set so the width the pads have to close
        # on is the same, which is what makes the comparison across shapes fair.
        shape = int(round(world.object_shape))
        if shape == 1:
            self.model.geom_type[self._object_gid] = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            self.model.geom_size[self._object_gid] = np.array([hs, hs, 0.0])
        elif shape == 2:
            self.model.geom_type[self._object_gid] = int(mujoco.mjtGeom.mjGEOM_SPHERE)
            self.model.geom_size[self._object_gid] = np.array([hs, 0.0, 0.0])
        else:
            self.model.geom_type[self._object_gid] = int(mujoco.mjtGeom.mjGEOM_BOX)
            self.model.geom_size[self._object_gid] = np.array([hs, hs, hs])
        # Keep the box a solid of constant density-free mass: mass is sampled
        # independently of size, and the inertia is recomputed to match.
        mass = world.object_mass
        self.model.body_mass[self._object_bid] = mass
        inertia = mass * (2.0 * (2.0 * hs) ** 2) / 12.0
        self.model.body_inertia[self._object_bid] = np.array([inertia] * 3)

        self.model.geom_friction[self._object_gid, 0] = world.object_friction
        self.model.geom_friction[self._table_gid, 0] = world.table_friction

        for act in ("left_drive", "right_drive"):
            aid = self.model.actuator(act).id
            # A MuJoCo position actuator computes gainprm[0] * ctrl +
            # biasprm[1] * qpos, so the gain has to be written to both fields
            # with opposite signs or the actuator stops being a position
            # servo and becomes a constant force.
            self.model.actuator_gainprm[aid, 0] = world.gripper_gain
            self.model.actuator_biasprm[aid, 1] = -world.gripper_gain

        if self.arm:
            # There is no weld to soften. `hand_compliance` is the solref of the
            # weld that drags the free-body hand, and its nearest analogue here
            # is how hard the arm insists on its commanded joint angles, which
            # is the same substitution the Isaac port documents. Scaled from the
            # nominal 0.02 so the same JSON drives both variants.
            gain = float(np.clip(0.02 / max(world.hand_compliance, 1e-4), 0.2, 5.0))
            self.model.actuator_gainprm[self._arm_ctrl, 0] = self._arm_gain0 * gain
        else:
            self.model.eq_solref[self._weld_id, 0] = world.hand_compliance
        self.model.opt.gravity[2] = -world.gravity

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.world = sample_world(
            self.rand_cfg, self.np_random, self._max_half_size)
        self._apply_world(self.world)

        mujoco.mj_resetData(self.model, self.data)

        hs = self.world.object_half_size
        jitter = self.world.init_xy_jitter
        ox = self.np_random.uniform(-jitter, jitter)
        oy = self.np_random.uniform(-jitter * 1.2, jitter * 1.2)
        yaw = self.np_random.uniform(-self.world.init_yaw_jitter, self.world.init_yaw_jitter)

        qadr = self._object_qadr
        self.data.qpos[qadr : qadr + 3] = [ox, oy, TABLE_HEIGHT + hs + 0.001]
        self.data.qpos[qadr + 3 : qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        # Hand starts above the table, offset from the object so the policy has
        # to do the reaching, fingers open.
        hand_xy = np.array([ox, oy]) + self.np_random.uniform(-0.06, 0.06, size=2)
        hand_pos = np.array([hand_xy[0], hand_xy[1], self.np_random.uniform(0.58, 0.66)])
        hand_pos[:2] = np.clip(hand_pos[:2], self._ws_low[:2], self._ws_high[:2])
        if not self.arm:
            hadr = self._hand_qadr
            self.data.qpos[hadr : hadr + 3] = hand_pos
            self.data.qpos[hadr + 3 : hadr + 7] = [1.0, 0.0, 0.0, 0.0]
        if self.arm:
            # Home the arm, then IK onto the same starting pose the weld version
            # begins from, so the two variants start the episode alike.
            self.data.qpos[self._arm_qpos] = self._arm_home
            mujoco.mj_kinematics(self.model, self.data)
            self._arm_target = hand_pos.copy()
            self._place_arm(hand_pos)
        else:
            self.data.mocap_pos[self._mocap_id] = hand_pos
            self.data.mocap_quat[self._mocap_id] = [1.0, 0.0, 0.0, 0.0]
        self._wrist_yaw = 0.0
        self._noise_state = {}
        # Open the fingers. Only the finger actuators: on the arm variant a
        # blanket write lands on the six joint setpoints too, leaving them at
        # zero while the joints sit at the IK solution. The actuators then close
        # a 2.7 radian gap on the first step and the arm hurls the box across
        # the room -- which is what a peak lift of 13.9 m in the expert check
        # turned out to be.
        self.data.ctrl[self._grip_ctrl] = self._grip_range[0]

        mujoco.mj_forward(self.model, self.data)

        self._object_rest_z = float(self.data.xpos[self._object_bid][2])
        self._object_start = self._object_pos().copy()
        self._lifted = 0.0
        if self.place:
            self._goal = self._sample_target(ox, oy)
        else:
            self._goal = np.array([ox, oy, TABLE_HEIGHT + self.hold_height])
        self.model.site_pos[self._goal_sid] = self._goal
        # The latency queue starts full of zero actions: for the first few
        # steps of a laggy episode the hand holds still and the gripper sits
        # mid-travel, because zero is the centre of the commanded range rather
        # than "open". That is a real (if small) artefact of modelling latency
        # as a queue; it is identical across all randomisation levels, so it
        # does not bias the ablation.
        self._latency_queue = deque(
            [np.zeros(self.act_dim, dtype=np.float64)] * int(self.world.action_latency)
        )
        self._steps = 0
        self._grasp_latch = 0
        self._prev_grip = self._grip_pos().copy()
        self._last_terms = {}

        # Second forward pass: the goal site moved after the first one, and
        # the observation reads site positions.
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info(False, False, 0.0)

    def _sample_target(self, ox: float, oy: float) -> np.ndarray:
        """Where the object has to end up, for the place task.

        Rejection-sampled rather than drawn as an offset at a random bearing,
        because an offset can land off the table and clipping it back piles
        targets up against the edges -- the policy would then learn a boundary,
        not a task. Rejection keeps the marginal distribution uniform over the
        region that is actually usable.

        The height is the object's *resting* height, measured after the object
        has been placed, so the success check can simply ask how far the object
        is from the goal in z. That matters once shapes are randomised: a
        cylinder on its side and a cube do not rest at the same height, and a
        hard-coded table offset would quietly make one of them impossible.
        """
        for _ in range(200):
            tx = self.np_random.uniform(-PLACE_TARGET_X, PLACE_TARGET_X)
            ty = self.np_random.uniform(-PLACE_TARGET_Y, PLACE_TARGET_Y)
            travel = float(np.hypot(tx - ox, ty - oy))
            if PLACE_MIN_TRAVEL <= travel <= PLACE_MAX_TRAVEL:
                return np.array([tx, ty, self._object_rest_z])
        # Unreachable with the current constants; if it ever fires, a fixed
        # fallback beats a silently biased sample.
        return np.array([ox, oy + PLACE_MIN_TRAVEL, self._object_rest_z])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(self.act_dim), -1.0, 1.0)
        commanded = action.copy()

        if self.world.action_noise > 0.0:
            commanded = np.clip(
                commanded + self.np_random.normal(
                    0.0, self.world.action_noise, self.act_dim),
                -1.0,
                1.0,
            )
        if self.world.action_latency > 0:
            self._latency_queue.append(commanded)
            commanded = self._latency_queue.popleft()

        if self.arm:
            # The setpoint is integrated, not read back from the arm. Reading it
            # back would make a zero action mean "stay wherever the arm has
            # sagged to", which is the mistake the Isaac port made first.
            self._arm_target = np.clip(
                self._arm_target + commanded[:3] * self.pos_step,
                self._ws_low, self._ws_high)
            target = self._arm_target
        else:
            target = self.data.mocap_pos[self._mocap_id] + commanded[:3] * self.pos_step
            self.data.mocap_pos[self._mocap_id] = np.clip(
                target, self._ws_low, self._ws_high)

        if self.wrist:
            self._wrist_yaw = float(np.clip(
                self._wrist_yaw + commanded[3] * WRIST_STEP, -WRIST_LIMIT, WRIST_LIMIT))
            if not self.arm:
                half = 0.5 * self._wrist_yaw
                self.data.mocap_quat[self._mocap_id] = [
                    np.cos(half), 0.0, 0.0, np.sin(half)]

        lo, hi = self._grip_range
        grip_cmd = lo + (commanded[-1] + 1.0) * 0.5 * (hi - lo)
        if self.arm:
            self._solve_ik(self._arm_target, self._wrist_yaw if self.wrist else 0.0)
            self.data.ctrl[self._grip_ctrl] = grip_cmd
        else:
            self.data.ctrl[:] = grip_cmd

        prev_grip = self._grip_pos().copy()
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._prev_grip = prev_grip
        self._steps += 1

        obs = self._observation()
        grasped = float(self._grasped())
        object_pos = self._object_pos()
        dropped = bool(dropped_condition(object_pos, TABLE_HEIGHT))

        if self.place:
            # The pick latch. Set once, never cleared: it records that the
            # object was carried rather than shoved, which is a fact about the
            # episode and not about the current step. Read from the simulator's
            # true height, not the observation, so sensing noise cannot hand a
            # policy credit for a lift that did not happen.
            if (grasped > 0.5 and object_pos[2] - self._object_rest_z
                    >= self.reward_cfg.lift_threshold):
                self._lifted = 1.0
            speed = float(np.linalg.norm(self.data.cvel[self._object_bid][3:6]))
            success = bool(place_success_condition(
                object_pos[None, :], self._goal[None, :], np.array([grasped]),
                np.array([self._lifted]), np.array([speed]), self.reward_cfg)[0])
            reward, terms = place_reward(
                self._grip_pos()[None, :],
                object_pos[None, :],
                self._goal[None, :],
                self._object_start[None, :],
                np.array([self._object_rest_z]),
                np.array([grasped]),
                np.array([self._lifted]),
                np.array([float(dropped)]),
                np.array([speed]),
                action[None, :],
                self.reward_cfg,
            )
        else:
            success = bool(
                success_condition(
                    object_pos[None, :], self._goal[None, :], np.array([grasped]),
                    self.reward_cfg
                )[0]
            )

            reward, terms = grasp_reward(
                self._grip_pos()[None, :],
                object_pos[None, :],
                self._goal[None, :],
                np.array([self._object_rest_z]),
                np.array([grasped]),
                np.array([float(dropped)]),
                action[None, :],
                self.reward_cfg,
                yaw_error=(np.array([self._yaw_error()]) if self.wrist else None),
            )
        reward = float(reward[0])
        self._last_terms = {
            k: float(np.asarray(v).reshape(-1)[0]) for k, v in terms.as_dict().items()
        }

        terminated = dropped
        truncated = self._steps >= self.max_steps
        return obs, reward, terminated, truncated, self._info(success, dropped, grasped)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def _grip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._grip_sid].copy()

    def _object_pos(self) -> np.ndarray:
        return self.data.xpos[self._object_bid].copy()

    def _gripper_width(self) -> float:
        """Distance between the inner pad faces, in metres.

        Each pad face sits 39 mm from the centreline when its slide joint is at
        zero, so the fully open gap is 78 mm and each millimetre of travel on
        either joint closes the gap by the same millimetre.
        """
        travel = float(
            self.data.qpos[self._finger_qadr[0]] + self.data.qpos[self._finger_qadr[1]]
        )
        return GRIPPER_OPEN_WIDTH - travel

    def _gripper_rate(self) -> float:
        return -float(
            self.data.qvel[self._finger_vadr[0]] + self.data.qvel[self._finger_vadr[1]]
        )

    def _grasped_raw(self) -> bool:
        """True when both pads are in contact with the object *this instant*.

        Read straight out of the contact list rather than inferred from finger
        positions: a policy that closes on thin air must not be told it has a
        grasp, and one that pinches the box off-centre must be.
        """
        left = right = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self._object_gid not in pair:
                continue
            other = pair[0] if pair[1] == self._object_gid else pair[1]
            if other == self._pad_gids[0]:
                left = True
            elif other == self._pad_gids[1]:
                right = True
            if left and right:
                return True
        return False

    def _grasped(self) -> bool:
        """Debounced grasp flag.

        The instantaneous flag chatters. A held box vibrates against the pads by
        a few tens of microns, so on any given step one of the two contacts can
        be absent while the object is unambiguously still in the hand. Since
        success is read at the *final* step of the episode, that chatter alone
        turned roughly half of the genuinely successful holds into failures.

        The flag is therefore latched for ``GRASP_DEBOUNCE_STEPS`` environment
        steps (0.12 s) after the last two-pad contact. That is short enough that
        a policy which opens its fingers is marked as having let go almost
        immediately, and long enough to ride out solver chatter. Must be called
        exactly once per environment step.
        """
        if self._grasped_raw():
            self._grasp_latch = GRASP_DEBOUNCE_STEPS
        elif self._grasp_latch > 0:
            self._grasp_latch -= 1
        return self._grasp_latch > 0

    def _observation(self) -> np.ndarray:
        grip = self._grip_pos()
        dt = self.model.opt.timestep * self.n_substeps
        grip_vel = (grip - self._prev_grip) / max(dt, 1e-6)
        obj = self._object_pos()
        rot = self.data.xmat[self._object_bid].reshape(3, 3)
        obj_lin = self.data.cvel[self._object_bid][3:6].copy()
        obj_ang = self.data.cvel[self._object_bid][0:3].copy()

        obs = np.concatenate(
            [
                grip,
                grip_vel,
                [self._gripper_width()],
                [self._gripper_rate()],
                obj,
                obj - grip,
                rot[:, 0],
                rot[:, 1],
                obj_lin,
                obj_ang,
                self._goal,
                self._goal - obj,
            ]
        ).astype(np.float64)

        if self.world.obs_noise_pos > 0.0:
            obs[0:3] += self._sensor_noise("grip", self.world.obs_noise_pos)
            obs[8:11] += self._sensor_noise("object", self.world.obs_noise_pos)
            obs[11:14] = obs[8:11] - obs[0:3]
            obs[29:32] = obs[26:29] - obs[8:11]
        if self.world.obs_noise_vel > 0.0:
            obs[3:6] += self.np_random.normal(0, self.world.obs_noise_vel, 3)
            obs[20:23] += self.np_random.normal(0, self.world.obs_noise_vel, 3)
        if self.wrist:
            # The policy has to know where its own wrist is. sin/cos rather than
            # the angle, so there is no discontinuity to learn around the limit.
            obs = np.concatenate(
                [obs, [np.sin(self._wrist_yaw), np.cos(self._wrist_yaw)]])
        if self.world.obs_noise_rot > 0.0:
            # Perturb the *frame*, not the six numbers independently. A pose
            # estimator returns one orientation with one error; noising the two
            # reported columns separately would hand the policy a pair of axes
            # that are no longer orthogonal, which no real estimator produces.
            noisy = self._rotation_error() @ rot
            obs[14:17] = noisy[:, 0]
            obs[17:20] = noisy[:, 1]

        return obs.astype(np.float32)

    def _sensor_noise(self, channel: str, sigma: float) -> np.ndarray:
        """Pose error, optionally correlated in time.

        Independent Gaussian noise per step is the easy model and the wrong
        one: a real pose estimator's error is dominated by viewpoint, occlusion
        and calibration, all of which persist across frames. Independent noise
        also flatters any policy that filters, because averaging kills it --
        and both the scripted expert and every policy cloned from it filter.

        ``obs_noise_corr`` is the correlation, as an Ornstein-Uhlenbeck-style
        first-order filter: 0.0 reproduces the old independent draw exactly, so
        every existing level and every existing number is unchanged, and 0.9
        gives an error that drifts over roughly ten frames.
        """
        rho = float(self.world.obs_noise_corr)
        draw = self.np_random.normal(0.0, sigma, 3)
        if rho <= 0.0:
            return draw
        prev = self._noise_state.get(channel)
        if prev is None:
            prev = self.np_random.normal(0.0, sigma, 3)
        # Scaled so the stationary standard deviation stays sigma whatever rho is:
        # a correlated error should not also be a bigger one.
        state = rho * prev + np.sqrt(1.0 - rho * rho) * draw
        self._noise_state[channel] = state
        return state

    def _build_start_pool(self, size: int = 256) -> None:
        """Precompute collision-free start configurations.

        Sampling a Cartesian start pose and solving IK to it is the obvious
        approach and it fails here: IK is collision-blind, so a converged
        solution is often one with a link inside the table, and retrying from
        random seeds found a clean solution on barely a third of resets.

        Sampling in *configuration* space instead makes validity constructive.
        A configuration is kept only if it is contact-free, puts the grip inside
        the workspace above the table, and holds the pads facing down. The exact
        start pose does not matter to the task -- the hand only has to begin
        above the table and offset from the object -- so choosing the pose to
        suit the arm rather than the other way round costs nothing.
        """
        names = ("j1", "j2", "j3", "j4", "j5", "j6")
        lo = np.array([self.model.joint(n).range[0] for n in names])
        hi = np.array([self.model.joint(n).range[1] for n in names])
        rng = np.random.default_rng(0)
        saved = self.data.qpos.copy()
        pool = []
        for _ in range(200_000):
            if len(pool) >= size:
                break
            q = rng.uniform(lo, hi)
            self.data.qpos[self._arm_qpos] = q
            mujoco.mj_kinematics(self.model, self.data)
            grip = self.data.site_xpos[self._grip_sid]
            if not (abs(grip[0]) < 0.17 and abs(grip[1]) < 0.17
                    and 0.55 < grip[2] < 0.68):
                continue
            if self.data.site_xmat[self._grip_sid].reshape(3, 3)[2, 2] < 0.80:
                continue
            mujoco.mj_forward(self.model, self.data)
            if any(self.data.contact[i].dist < -0.001 for i in range(self.data.ncon)):
                continue
            pool.append((q.copy(), grip.copy()))
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        self._start_pool = pool

    def _place_arm(self, target_pos: np.ndarray) -> None:
        """Start the arm from a precomputed collision-free configuration.

        ``target_pos`` is advisory: the pool entry closest to it is used, and
        the commanded setpoint becomes that entry's actual grip position so the
        controller does not begin the episode chasing an error.
        """
        if not self._start_pool:
            self._arm_place_failures += 1
            return
        idx = int(np.argmin([np.linalg.norm(g - target_pos) for _, g in self._start_pool]))
        q, grip = self._start_pool[idx]
        self.data.qpos[self._arm_qpos] = q
        self.data.ctrl[self._arm_ctrl] = q
        mujoco.mj_forward(self.model, self.data)
        self._arm_target = grip.copy()
        self._arm_placements += 1

    def _solve_ik(self, target_pos: np.ndarray, target_yaw: float,
                  iters: int = IK_ITERS, hold: bool = True) -> None:
        """Move the arm so the grip site reaches a Cartesian pose.

        Damped least squares on the stacked position and orientation error.
        Orientation is solved, not ignored: the Isaac port taught this the
        expensive way -- position-only IK leaves the wrist free to rotate, the
        pads drift off square to the table, and a top-down grasp stops being
        possible. The target orientation is the hand's rest pose turned by
        ``target_yaw`` about the vertical, which is fixed at zero unless the
        wrist variant is also on.

        Nothing here clamps to the joint limits by hand: the actuators' control
        ranges do that, so an unreachable command produces the same thing a real
        arm produces, which is an arm that stops short.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        cols = self._arm_dofs
        restore = self.data.qpos[self._arm_qpos].copy() if hold else None
        for _ in range(iters):
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._grip_sid)
            pos_err = target_pos - self.data.site_xpos[self._grip_sid]

            # Orientation error as a rotation vector, from the current frame to
            # the desired one.
            cur = self.data.site_xmat[self._grip_sid].reshape(3, 3)
            des = self._desired_grip_frame(target_yaw)
            rel = des @ cur.T
            angle = np.arccos(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
            if angle < 1e-6:
                rot_err = np.zeros(3)
            else:
                axis = np.array([rel[2, 1] - rel[1, 2],
                                 rel[0, 2] - rel[2, 0],
                                 rel[1, 0] - rel[0, 1]]) / (2.0 * np.sin(angle))
                rot_err = axis * angle

            err = np.concatenate([pos_err, 0.35 * rot_err])
            jac = np.vstack([jacp[:, cols], jacr[:, cols]])
            hess = jac @ jac.T + (IK_DAMPING ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(hess, err)

            self.data.qpos[self._arm_qpos] += dq
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

        solution = self.data.qpos[self._arm_qpos].copy()
        if restore is not None:
            # Put the simulator back where it was. The iteration above walks
            # `qpos` because the Jacobian has to be evaluated at each trial
            # configuration, but that is a *solver* stepping through candidates,
            # not the arm moving: leaving it applied teleports the arm every
            # control step and the physics then fights the actuators. The first
            # version did exactly that, and the hand climbed while its target
            # descended.
            self.data.qpos[self._arm_qpos] = restore
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        self.data.ctrl[self._arm_ctrl] = solution

    def _desired_grip_frame(self, yaw: float) -> np.ndarray:
        """Hand pointing down the table normal, optionally yawed about it."""
        c, s_ = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ self._rest_frame

    def _yaw_error(self) -> float:
        """Signed angle between the closing axis and the nearest box face.

        Folded into +/-45 degrees, because a box repeats every 90 and there is
        never a reason to turn further than that. Zero when the pads are square
        to a face, which is the alignment the wrist exists to reach.
        """
        rot = self.data.xmat[self._object_bid].reshape(3, 3)
        object_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        error = object_yaw - self._wrist_yaw
        return float((error + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)

    def _rotation_error(self) -> np.ndarray:
        """A small random rotation, angle ~ N(0, obs_noise_rot) about a random axis."""
        axis = self.np_random.normal(size=3)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        angle = float(self.np_random.normal(0.0, self.world.obs_noise_rot))
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return (np.eye(3) + np.sin(angle) * skew
                + (1.0 - np.cos(angle)) * (skew @ skew))

    def _info(self, success: bool, dropped: bool, grasped: float) -> Dict:
        return {
            "is_success": bool(success),
            "dropped": bool(dropped),
            "grasped": float(grasped),
            "object_height": float(self._object_pos()[2] - self._object_rest_z),
            "lifted": float(self._lifted),
            "goal_distance": float(np.linalg.norm(
                (self._object_pos() - self._goal)[:2] if self.place
                else self._object_pos() - self._goal)),
            "goal": self._goal.copy(),
            "reward_terms": dict(self._last_terms),
            "world": self.world.as_dict(),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self._render_size)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(
    randomisation: str = "none",
    seed: Optional[int] = None,
    reward_config: Optional[str] = None,
    **kwargs: Any,
) -> GraspEnv:
    """Convenience constructor used by every script in ``src/``."""
    from src.rewards.grasp_reward import load_reward_config
    from src.rewards.place_reward import load_place_config

    # The two tasks have different weight sets, so which loader applies is
    # decided by the task rather than by the file: passing a lift config to the
    # place task would silently drop every place-specific weight to a default.
    load = load_place_config if kwargs.get("task") == "place" else load_reward_config
    env = GraspEnv(
        reward_cfg=load(reward_config),
        randomisation=randomisation,
        seed=seed,
        **kwargs,
    )
    return env
