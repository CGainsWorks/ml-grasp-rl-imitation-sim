"""MuJoCo lift-and-hold grasp environment.

Task
----
A parallel-jaw hand starts above a table. A box sits somewhere on the table at
a random position and yaw. The hand must close on the box, lift it to a hold
point 0.15 m above the table, and still be holding it there when the episode
ends. Success is read at the **final** step, so letting go early scores zero.

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

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "grasp_scene.xml")

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
        max_half_size: Optional[float] = None,
    ) -> None:
        self.wrist = bool(wrist)
        # Overridable so the wrist can be ablated properly: the same box
        # distribution has to be presented to a hand that can rotate and one
        # that cannot, or the comparison is between two different tasks.
        self._max_half_size = float(
            max_half_size if max_half_size is not None
            else (WRIST_MAX_HALF_SIZE if self.wrist else 0.024))
        self.obs_dim = WRIST_OBS_DIM if self.wrist else OBS_DIM
        self.act_dim = WRIST_ACT_DIM if self.wrist else ACT_DIM
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        self.reward_cfg = reward_cfg or GraspRewardConfig()
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
        self._mocap_id = self.model.body("mocap").mocapid[0]
        self._object_bid = self.model.body("object").id
        self._object_gid = self.model.geom("object").id
        self._table_gid = self.model.geom("table_top").id
        self._grip_sid = self.model.site("grip_site").id
        self._goal_sid = self.model.site("goal").id
        self._pad_gids = (self.model.geom("left_pad").id, self.model.geom("right_pad").id)
        self._object_qadr = self.model.jnt_qposadr[self.model.joint("object_free").id]
        self._hand_qadr = self.model.jnt_qposadr[self.model.joint("hand_free").id]
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
        self._weld_id = 0  # the hand weld is the only equality constraint

        # Workspace box for the mocap target. The lower z bound keeps the pads
        # about 4 mm clear of the table so the hand cannot wedge itself under it.
        self._ws_low = np.array([-0.20, -0.28, 0.462])
        self._ws_high = np.array([0.20, 0.28, 0.72])

        self.world: SampledWorld = sample_world(
            self.rand_cfg, self.np_random, self._max_half_size)
        self._latency_queue: deque = deque()
        self._steps = 0
        self._goal = np.zeros(3)
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
        hadr = self._hand_qadr
        self.data.qpos[hadr : hadr + 3] = hand_pos
        self.data.qpos[hadr + 3 : hadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.mocap_pos[self._mocap_id] = hand_pos
        self.data.mocap_quat[self._mocap_id] = [1.0, 0.0, 0.0, 0.0]
        self._wrist_yaw = 0.0
        self._noise_state = {}
        self.data.ctrl[:] = self._grip_range[0]

        mujoco.mj_forward(self.model, self.data)

        self._object_rest_z = float(self.data.xpos[self._object_bid][2])
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

        target = self.data.mocap_pos[self._mocap_id] + commanded[:3] * self.pos_step
        self.data.mocap_pos[self._mocap_id] = np.clip(target, self._ws_low, self._ws_high)

        if self.wrist:
            self._wrist_yaw = float(np.clip(
                self._wrist_yaw + commanded[3] * WRIST_STEP, -WRIST_LIMIT, WRIST_LIMIT))
            half = 0.5 * self._wrist_yaw
            self.data.mocap_quat[self._mocap_id] = [np.cos(half), 0.0, 0.0, np.sin(half)]

        lo, hi = self._grip_range
        grip_cmd = lo + (commanded[-1] + 1.0) * 0.5 * (hi - lo)
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
        success = bool(
            success_condition(
                object_pos[None, :], self._goal[None, :], np.array([grasped]), self.reward_cfg
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

    env = GraspEnv(
        reward_cfg=load_reward_config(reward_config),
        randomisation=randomisation,
        seed=seed,
        **kwargs,
    )
    return env
