"""A scripted expert for the pick-and-place task.

The first three phases are the lift expert's, inherited rather than copied --
approaching and grasping a box is the same problem whatever you then do with it,
and two copies of that code would drift. What is new is everything after the
grasp:

    LIFT      raise the object to transport height, straight up
    TRAVERSE  fly it across to above the target, at that height
    LOWER     set it down on the target
    RELEASE   open the fingers and wait for the pads to clear
    RETREAT   rise clear of the object and stay there

Two things about this sequence are decisions rather than details.

**Lift, then traverse, rather than both at once.** A proportional controller
driving the hand straight at "above the target, 10 cm up" moves diagonally, and
for the first part of that diagonal the box is still on the table being dragged.
It arrives, but it arrives having scraped, and under randomised friction it
arrives having tipped over. Two axis-aligned phases cost about eight extra steps
and remove the failure mode.

**Release, then retreat, as separate phases.** Opening the fingers and
retreating in the same step lifts the box a few millimetres on the pads as they
part -- enough to leave it rocking, and the success check reads the object's
speed. Waiting for the gripper to clear before moving is what makes the placed
box actually settle.

Like the lift expert this is deliberately imperfect: proportional gains, no
re-planning if the box slips in the fingers during the traverse, and no check
that the grasp took before it commits to carrying. A behaviour-cloned copy of a
flawless expert would tell you nothing about whether imitation is working.
"""

from __future__ import annotations

import numpy as np

from src.policies.scripted_expert import (
    APPROACH,
    CLOSE,
    DESCEND,
    GRIP_POS,
    OBJ_POS,
    ScriptedExpert,
)

# The goal site sits at the object's resting height, so every target height
# below is expressed relative to it and nothing needs to know the table's
# thickness or the box's size.
GOAL_POS = slice(26, 29)

LIFT, TRAVERSE, LOWER, RELEASE, RETREAT = range(3, 8)

# Height the object is carried at. High enough to clear a box of the largest
# randomised size sitting in the way, low enough that a slip during the traverse
# does not turn into a drop test.
CARRY_HEIGHT = 0.10
# Where the object is let go from. Not zero: releasing with the box pressed onto
# the table loads the pads against it, and the pads then flick it sideways as
# they part.
PLACE_CLEARANCE = 0.006
RETREAT_HEIGHT = 0.14

SPEED_CAP = {LIFT: 0.55, TRAVERSE: 0.70, LOWER: 0.30, RELEASE: 0.10, RETREAT: 0.60}


class ScriptedPlaceExpert(ScriptedExpert):
    """Phase-based expert for ``task="place"``. One instance per episode."""

    def __init__(self, *args, release_steps: int = 6, carry_tol: float = 0.020,
                 lower_tol: float = 0.010, **kwargs) -> None:
        self.release_steps = int(release_steps)
        self.carry_tol = float(carry_tol)
        self.lower_tol = float(lower_tol)
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._release_counter = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        # The grasp phases are the parent's, unchanged, including its low-pass
        # filter and its wrist handling.
        # The parent's LIFT constant is this module's LIFT, so its transition
        # out of CLOSE lands directly in the sequence below.
        if self.phase in (APPROACH, DESCEND, CLOSE):
            return super().act(obs)

        obs = np.asarray(obs, dtype=np.float64)
        a = self.filter_alpha
        self._filtered = a * obs + (1.0 - a) * self._filtered
        filtered = self._filtered
        grip = filtered[GRIP_POS]
        obj = filtered[OBJ_POS]
        goal = obs[GOAL_POS]  # the target is commanded, not sensed
        rest_z = goal[2]
        self._phase_steps += 1

        # Captured before the transitions below, so a phase that advances this
        # step still executes its own action rather than half of the next one's.
        phase = self.phase
        grip_cmd = 1.0
        if phase == LIFT:
            want = np.array([obj[0], obj[1], rest_z + CARRY_HEIGHT])
            if obj[2] - rest_z > CARRY_HEIGHT - 0.02 or self._phase_steps > 40:
                self._advance(TRAVERSE)
        elif phase == TRAVERSE:
            want = np.array([goal[0], goal[1], rest_z + CARRY_HEIGHT])
            if (np.linalg.norm(obj[:2] - goal[:2]) < self.carry_tol
                    or self._phase_steps > 60):
                self._advance(LOWER)
        elif phase == LOWER:
            want = np.array([goal[0], goal[1], rest_z + PLACE_CLEARANCE])
            if (abs(obj[2] - rest_z - PLACE_CLEARANCE) < self.lower_tol
                    or self._phase_steps > 40):
                self._advance(RELEASE)
        elif phase == RELEASE:
            # Hold still. The only thing happening this phase is the fingers
            # opening, and any hand motion while they are in contact drags what
            # was just placed.
            want = grip
            grip_cmd = -1.0
            self._release_counter += 1
            if self._release_counter >= self.release_steps:
                self._advance(RETREAT)
        else:  # RETREAT
            want = np.array([goal[0], goal[1], rest_z + RETREAT_HEIGHT])
            grip_cmd = -1.0

        # Closed loop on the *object* while it is in the hand, on the hand once
        # it is not. The weld dragging the hand is compliant, so under load the
        # hand hangs below where it was told to go; commanding "move by the
        # error the object still has" cancels that sag without modelling it.
        # After release there is no object to close the loop on -- doing it
        # anyway would command the hand towards a box it is supposed to be
        # leaving alone.
        if phase in (RELEASE, RETREAT):
            target = want if phase == RETREAT else grip
        else:
            target = grip + (want - obj)

        delta = np.clip(self.kp * (target - grip), -SPEED_CAP[phase],
                        SPEED_CAP[phase])
        if self.wrist:
            action = np.concatenate([delta, [0.0], [grip_cmd]])
        else:
            action = np.concatenate([delta, [grip_cmd]])
        if self.noise > 0.0:
            action = action + self.rng.normal(0.0, self.noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
