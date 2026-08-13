"""Reward and termination definitions for the lift-and-hold grasp task.

Why this file is backend-agnostic
---------------------------------
The MuJoCo environment steps one world at a time and hands us ``numpy`` arrays.
The Isaac Lab port steps many worlds at a time and hands us ``torch`` tensors on
the GPU. Both call the functions below. That is the only way the claim "the
Isaac task is the same task" means anything: if the reward lived in two places
it would drift apart within a week and the numbers from the two simulators
would not be comparable.

Everything here is written with plain arithmetic plus a few operations
(``clip``, ``exp``, ``sqrt``, ``sum``) resolved against whichever array library
the caller used. Inputs are batched: ``(N, 3)`` for positions, ``(N,)`` for
scalars, ``(N, A)`` for actions. The MuJoCo env passes ``N = 1``.

The rationale for each term is in ``docs/reward-design.md``. Keep the two in
step when changing weights.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional

import numpy as np


# --------------------------------------------------------------------------
# Backend shim
# --------------------------------------------------------------------------
def _xp(x: Any):
    """Return the array module that owns ``x`` (numpy, or torch if installed)."""
    if type(x).__module__.startswith("torch"):
        import torch  # local import: torch is optional for pure-numpy users

        return torch
    return np


def _norm(x, xp):
    """Euclidean norm over the last axis, avoiding linalg API differences."""
    return xp.sqrt(xp.sum(x * x, axis=-1) + 1e-12)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GraspRewardConfig:
    """Weights and thresholds for :func:`grasp_reward`.

    The defaults are the ones every number in the README was produced with.
    They were chosen by hand, not searched; ``docs/reward-design.md`` records
    what each one is meant to buy and what happened when it was removed.
    """

    # Dense terms
    w_reach: float = 1.0          # pull the pads onto the object
    w_align: float = 0.3          # penalise lateral offset harder than vertical
    w_grasp: float = 0.5          # both pads in contact with the object
    w_lift: float = 4.0           # height gained above the resting height
    w_place: float = 6.0          # closing distance to the hold point, once grasped
    w_hold: float = 3.0           # smooth bonus for sitting *at* the hold point
    # Events
    w_success: float = 5.0        # paid every step the success condition holds
    w_drop: float = 5.0           # paid once, negatively, if the object leaves the table
    # Regularisers
    w_action: float = 0.02        # squared action penalty, discourages bang-bang
    w_time: float = 0.0           # per-step cost; zero by default, the horizon is fixed

    # Thresholds
    lift_target: float = 0.12     # metres of clearance that counts as fully lifted
    goal_tolerance: float = 0.05  # metres, radius of the hold point
    reach_scale: float = 0.10     # metres, distance at which the reach term saturates
    hold_scale: float = 0.05      # metres, width of the hold bonus

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


def load_reward_config(path: Optional[str]) -> GraspRewardConfig:
    """Load weights from a JSON file, falling back to the documented defaults."""
    if not path:
        return GraspRewardConfig()
    if not os.path.exists(path):
        raise FileNotFoundError("reward config not found: " + str(path))
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(GraspRewardConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError("unknown reward keys in {}: {}".format(path, sorted(unknown)))
    return GraspRewardConfig(**raw)


@dataclasses.dataclass
class RewardTerms:
    """Per-term breakdown, returned alongside the scalar so plots can show it."""

    reach: Any
    align: Any
    grasp: Any
    lift: Any
    place: Any
    hold: Any
    success: Any
    drop: Any
    action: Any
    time: Any

    def total(self):
        return (
            self.reach + self.align + self.grasp + self.lift + self.place
            + self.hold + self.success + self.drop + self.action + self.time
        )

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def names():
        return [f.name for f in dataclasses.fields(RewardTerms)]


# --------------------------------------------------------------------------
# Success / failure
# --------------------------------------------------------------------------
def success_condition(object_pos, goal_pos, grasped, cfg: GraspRewardConfig):
    """Task success: object held inside the goal sphere, with both pads on it.

    Requiring ``grasped`` is what stops the classic exploit where the policy
    flicks the object upward and collects the goal bonus mid-flight. Evaluation
    reads this at the final step of the episode, so a policy that throws the
    object and lets go scores zero.
    """
    xp = _xp(object_pos)
    dist = _norm(object_pos - goal_pos, xp)
    return (dist < cfg.goal_tolerance) & (grasped > 0.5)


def dropped_condition(object_pos, table_height: float, margin: float = 0.06):
    """Failure: the object has fallen off (or through) the table."""
    return object_pos[..., 2] < (table_height - margin)


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------
def grasp_reward(
    grip_pos,
    object_pos,
    goal_pos,
    object_rest_z,
    grasped,
    dropped,
    action,
    cfg: GraspRewardConfig,
):
    """Dense staged reward for lift-and-hold grasping.

    The shape of the reward, in words:

    1. ``reach``   rises as the pads close on the object and saturates at
       ``reach_scale``, so a distant object does not dominate the sum.
    2. ``align``   an extra penalty on horizontal offset only. Vertical error
       is recoverable; lateral error knocks the object over.
    3. ``grasp``   a flat bonus while both pads touch the object. Deliberately
       small: it is a means, not the goal. Too large and the policy learns to
       hold the object against the table forever.
    4. ``lift``    height gained over the resting height, clipped at
       ``lift_target``. This is the term that actually creates the behaviour.
    5. ``place``   progress from the object's resting place towards the hold
       point, gated on ``grasped`` so the policy is not paid for shoving the
       box across the table.
    6. ``hold``    a smooth bump centred on the hold point, width
       ``hold_scale``. Without it the only thing marking the goal is the
       success bonus, which is a cliff: a policy that flies the box past the
       hold point on its way to the ceiling collects that bonus for one or two
       steps out of a hundred, the critic almost never sees it, and the run
       settles into hoisting the box as high as it will go. This term turns
       that cliff into a hill the policy can climb. It cost one wasted set of
       training runs to find out.
    7. ``success`` paid on every step the success condition holds, which
       rewards holding rather than touching and letting go.
    8. ``drop``    one-off penalty when the object leaves the table.
    9. ``action``  squared-action penalty for smoothness.

    Returns ``(reward, terms)``.
    """
    xp = _xp(grip_pos)

    reach_dist = _norm(object_pos - grip_pos, xp)
    reach = -cfg.w_reach * (1.0 - xp.exp(-reach_dist / cfg.reach_scale))

    lateral = _norm((object_pos - grip_pos)[..., :2], xp)
    align = -cfg.w_align * lateral

    grasp = cfg.w_grasp * grasped

    height_gain = xp.clip(object_pos[..., 2] - object_rest_z, 0.0, cfg.lift_target)
    lift = cfg.w_lift * height_gain

    goal_dist = _norm(object_pos - goal_pos, xp)
    # Progress towards the hold point, measured from where the object started
    # rather than as an absolute distance. Written as a raw distance penalty
    # this term charges the policy for picking the box up at all: on the table,
    # grasped, it is a flat -0.77 per step, which is more than the grasp bonus
    # pays, so the only way to profit is to discover grasping and lifting in
    # one move. Measured as progress it is zero at rest, rises as the box comes
    # up, and falls again past the hold point.
    rest_distance = goal_pos[..., 2] - object_rest_z
    place = cfg.w_place * (rest_distance - goal_dist) * grasped
    hold = cfg.w_hold * xp.exp(-goal_dist / cfg.hold_scale) * grasped

    success = cfg.w_success * (success_condition(object_pos, goal_pos, grasped, cfg) * 1.0)

    drop = -cfg.w_drop * dropped

    action_cost = -cfg.w_action * xp.sum(action * action, axis=-1)
    time_cost = -cfg.w_time * (grasp * 0.0 + 1.0)

    terms = RewardTerms(
        reach=reach,
        align=align,
        grasp=grasp,
        lift=lift,
        place=place,
        hold=hold,
        success=success,
        drop=drop,
        action=action_cost,
        time=time_cost,
    )
    return terms.total(), terms
