"""A scripted expert for the grasp task, used to generate demonstrations.

This is a state machine, not a planner and not a policy: it reads the object
pose out of the observation vector and drives the hand through four phases.

    APPROACH  move to a waypoint above the object, fingers open
    DESCEND   drop onto the grasp height, fingers still open
    CLOSE     squeeze, and wait for the pads to load up
    LIFT      rise to the hold point and stay there

It is deliberately imperfect. The gains are proportional with a saturating
step, it does not re-plan if the box is knocked sideways during the descent,
and it does not check whether the grasp actually took before lifting. On the
nominal world it succeeds on the large majority of episodes; under heavy
randomisation it degrades, which is the honest thing for demonstrations to do.
A behaviour-cloned copy of a flawless expert would tell you nothing about
whether imitation is working.

The expert consumes the same 32-dimensional observation the policies see, so
it can also label states visited by a learner: that is what makes DAgger
possible here.
"""

from __future__ import annotations

import numpy as np

# Observation slices, mirroring the table in envs/mujoco/grasp_env.py
GRIP_POS = slice(0, 3)
GRIP_WIDTH = 6
OBJ_POS = slice(8, 11)
OBJ_ROT_X = slice(14, 17)
GOAL_POS = slice(26, 29)
# Only present on the wrist variant of the task, where the observation carries
# the hand's own yaw as a sin/cos pair on the end.
WRIST_SIN, WRIST_COS = 32, 33
# Matches WRIST_STEP in envs/mujoco/grasp_env.py: the yaw commanded by a unit
# action in one control step.
WRIST_STEP = np.deg2rad(9.0)

APPROACH, DESCEND, CLOSE, LIFT = range(4)

# Vertical offset from the object centre to the grip site for a good grasp. The
# grip site sits at the centre of the finger pads, so this is close to zero: a
# few millimetres high, which keeps the finger tips clear of the table without
# giving up pad contact area on the box.
GRASP_Z_OFFSET = 0.004
# How far above the object the approach waypoint sits.
APPROACH_HEIGHT = 0.085
# Largest commanded displacement per phase, as a fraction of the 2 cm step.
SPEED_CAP = {APPROACH: 1.0, DESCEND: 0.35, CLOSE: 0.20, LIFT: 0.55}


class ScriptedExpert:
    """Phase-based expert. One instance per episode; call :meth:`reset` between."""

    def __init__(
        self,
        kp: float = 8.0,
        lift_kp: float = 5.0,
        close_steps: int = 6,
        approach_tol: float = 0.020,
        descend_tol: float = 0.012,
        filter_alpha: float = 0.5,
        phase_timeout: tuple = (45, 30),
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
        wrist: bool = False,
        align_tol: float = np.deg2rad(6.0),
        grasp_offset: float = 0.0,
        grasp_yaw_offset: float = 0.0,
    ) -> None:
        # Extra yaw, in radians, between the object's frame and the direction
        # the pads should close along.
        #
        # Zero for a box, whose faces repeat every 90 degrees, so "square to the
        # nearest face" is unambiguous. A *bar* is not like that: it repeats
        # every 180 degrees and there is exactly one right answer, across its
        # thin axis rather than along its length. Folding a bar's yaw into
        # +/-45 degrees the way a box's is folded picks the wrong one half the
        # time, which is why the handled shape needs this and the others do not.
        self.grasp_yaw_offset = float(grasp_yaw_offset)
        # Where along the object's own x axis to grasp, in metres.
        #
        # Zero for every shape whose reported pose *is* a graspable point, which
        # is all of them except the handled one. There the body frame sits on a
        # part wider than the pads can open, so an expert that aims at the
        # reported position closes on nothing -- which is the point of that
        # shape, and is measured rather than asserted in
        # experiments/grasp_point.py.
        #
        # The offset is applied in the object's frame, so it rotates with the
        # object and the expert has to read the orientation to use it. That is
        # the same information a policy would need.
        self.grasp_offset = float(grasp_offset)
        self.wrist = bool(wrist)
        self.align_tol = float(align_tol)
        self.kp = kp
        self.lift_kp = lift_kp
        self.close_steps = close_steps
        self.approach_tol = approach_tol
        self.descend_tol = descend_tol
        self.filter_alpha = filter_alpha
        self.phase_timeout = phase_timeout
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.phase = APPROACH
        self._close_counter = 0
        self._phase_steps = 0
        self._filtered: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _advance(self, phase: int) -> None:
        self.phase = phase
        self._phase_steps = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)

        # Low-pass the pose channels. Under the noisier worlds a raw reading
        # jitters by more than the phase tolerances, and an unfiltered state
        # machine simply never leaves APPROACH. Any real system filters its
        # pose estimate; so does this one.
        if self._filtered is None:
            self._filtered = obs.copy()
        else:
            a = self.filter_alpha
            self._filtered = a * obs + (1.0 - a) * self._filtered
        filtered = self._filtered

        grip = filtered[GRIP_POS]
        obj = filtered[OBJ_POS]
        if self.grasp_offset:
            obj = obj + self.grasp_offset * filtered[OBJ_ROT_X]
        goal = obs[GOAL_POS]  # the hold point is commanded, not sensed
        self._phase_steps += 1

        # Wrist yaw, when the hand has one. The pads close along the hand's own
        # x axis, so the grasp wants that axis square to a face of the box. A
        # box repeats every 90 degrees, so the target is the object's yaw folded
        # into +/-45 degrees: there is never a reason to turn further, and
        # folding keeps the command inside the joint's range.
        #
        # This is computed *before* the phase machine because the descent has to
        # wait for it. Descending onto a big box while the wrist is still
        # turning sweeps it off the table, and at 35 mm half-size that is every
        # episode -- the first version of this method computed the yaw after the
        # transitions and the alignment gate was dead code.
        wrist_cmd = None
        aligned = True
        if self.wrist:
            obj_x = filtered[OBJ_ROT_X]
            object_yaw = float(np.arctan2(obj_x[1], obj_x[0]))
            if self.grasp_yaw_offset:
                # A bar: one correct closing direction, and a 180-degree
                # symmetry rather than a 90-degree one.
                target_yaw = object_yaw + self.grasp_yaw_offset
                folded = (target_yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
            else:
                folded = (object_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
            error = folded - float(np.arctan2(obs[WRIST_SIN], obs[WRIST_COS]))
            wrist_cmd = float(np.clip(error / WRIST_STEP, -1.0, 1.0))
            aligned = abs(error) <= self.align_tol

        if self.phase == APPROACH:
            target = np.array([obj[0], obj[1], obj[2] + APPROACH_HEIGHT])
            grip_cmd = -1.0
            timed_out = self._phase_steps > self.phase_timeout[0]
            in_place = np.linalg.norm(target - grip) < self.approach_tol
            if (in_place and aligned) or timed_out:
                self._advance(DESCEND)
        elif self.phase == DESCEND:
            target = np.array([obj[0], obj[1], obj[2] + GRASP_Z_OFFSET])
            grip_cmd = -1.0
            timed_out = self._phase_steps > self.phase_timeout[1]
            if abs(target[2] - grip[2]) < self.descend_tol or timed_out:
                self._advance(CLOSE)
        elif self.phase == CLOSE:
            target = np.array([obj[0], obj[1], obj[2] + GRASP_Z_OFFSET])
            grip_cmd = 1.0
            self._close_counter += 1
            if self._close_counter >= self.close_steps:
                self._advance(LIFT)
        else:  # LIFT
            # Close the loop on the *object*, not on the hand. The weld that
            # drags the hand to the mocap target is compliant, so under load
            # the hand hangs below where it was told to go; a heavy box and a
            # soft weld together put that sag well outside the goal tolerance.
            # Commanding "move by whatever error the object still has" cancels
            # the sag without needing to know the compliance.
            target = grip + (goal - obj)
            grip_cmd = 1.0

        # Proportional controller on the mocap displacement. The env scales the
        # action by pos_step (2 cm), so a unit action is a 2 cm command. The
        # lift gain is lower: the hand is carrying a load by then, and a
        # saturating command plus command latency overshoots the hold point.
        # Per-phase speed caps. Approaching fast is free; descending fast is
        # not. With command latency in the world, a saturated lateral command
        # during the descent sweeps the box off the table before the fingers
        # ever close on it, which is exactly what the shifted worlds punish.
        gain = self.lift_kp if self.phase == LIFT else self.kp
        cap = SPEED_CAP[self.phase]
        delta = np.clip(gain * (target - grip), -cap, cap)
        if wrist_cmd is None:
            action = np.concatenate([delta, [grip_cmd]])
        else:
            action = np.concatenate([delta, [wrist_cmd], [grip_cmd]])

        if self.noise > 0.0:
            action = action + self.rng.normal(0.0, self.noise, size=action.shape)

        return np.clip(action, -1.0, 1.0).astype(np.float32)


def rollout(env, expert: ScriptedExpert | None = None, seed: int | None = None):
    """Run one expert episode. Returns a dict of stacked arrays plus the outcome."""
    expert = expert or ScriptedExpert()
    expert.reset()
    obs, _ = env.reset(seed=seed)

    observations, actions, rewards, phases = [], [], [], []
    success = False
    info: dict = {}
    while True:
        action = expert.act(obs)
        observations.append(obs.copy())
        actions.append(action.copy())
        phases.append(expert.phase)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "phases": np.asarray(phases, dtype=np.int8),
        "success": success,
        "return": float(np.sum(rewards)),
        "info": info,
    }
