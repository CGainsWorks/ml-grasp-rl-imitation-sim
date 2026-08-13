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
    ) -> None:
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
        goal = obs[GOAL_POS]  # the hold point is commanded, not sensed
        self._phase_steps += 1

        if self.phase == APPROACH:
            target = np.array([obj[0], obj[1], obj[2] + APPROACH_HEIGHT])
            grip_cmd = -1.0
            timed_out = self._phase_steps > self.phase_timeout[0]
            if np.linalg.norm(target - grip) < self.approach_tol or timed_out:
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
        action = np.concatenate([delta, [grip_cmd]])

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
