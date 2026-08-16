"""Reward and termination for the pick-and-place task.

Why a second task exists
------------------------
Every number in this repository comes from one task: lift the box to a point
above where it started and still be holding it at the end. A reward design that
works on exactly one task is not evidence that the design *method* works, and
``docs/limitations.md`` said so. This is the second task, and it was chosen to
break the first one's assumptions rather than to be easy:

* the goal is **somewhere else on the table**, so the object has to travel
  laterally, which the lift task never asks for;
* success requires the hand to have **let go**, which is the exact opposite of
  the lift task's "still holding at the final step";
* success requires the object to have been **picked up on the way**, so sliding
  it across the table does not count.

The observation is unchanged -- the hold point already lives at indices 26:29
and the goal-relative offset at 29:32, so moving the goal onto the table needs
no new channel and every network, demonstration format and training script
carries over untouched. That is a property of the original observation design
and it is the one thing here that came for free.

Backend-agnostic in the same way as :mod:`src.rewards.grasp_reward`, so an
Isaac port of this task would share the definition rather than re-implement it.
No such port exists yet; ``docs/limitations.md`` says so plainly.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional

from src.rewards.grasp_reward import _norm, _xp


@dataclasses.dataclass(frozen=True)
class PlaceRewardConfig:
    """Weights and thresholds for :func:`place_reward`.

    Chosen by the same method as the lift weights, and with the same honesty
    about it: by hand, from the failure modes described below, not by a search.
    """

    # Dense terms
    w_reach: float = 1.0          # pull the pads onto the object
    w_align: float = 0.3          # lateral offset costs more than vertical
    w_grasp: float = 0.5          # both pads in contact
    w_clear: float = 3.0          # lift the object clear of the table before carrying
    w_carry: float = 6.0          # progress across the table, once grasped
    w_settle: float = 3.0         # object down, near the target, hand off it
    # Events
    w_success: float = 8.0        # paid every step the place condition holds
    w_drop: float = 5.0           # paid once if the object leaves the table
    # Regularisers
    w_action: float = 0.02

    # Thresholds
    clear_target: float = 0.06    # metres of clearance that counts as carried
    lift_threshold: float = 0.04  # clearance that latches "this was picked up"
    goal_tolerance: float = 0.05  # metres, radius of the target patch
    place_z_tolerance: float = 0.02   # how close to its resting height it must sit
    speed_tolerance: float = 0.05     # m/s, so a box in flight is not "placed"
    reach_scale: float = 0.10
    settle_scale: float = 0.05

    # How `carry` -- progress across the table -- is gated on having picked the
    # object up. Three settings, and all three were run, because the first two
    # are wrong in instructive ways.
    #
    # "none"   the first design. `carry` is the largest dense term and it was
    #          payable to a policy that closed the pads on the box and *pushed*:
    #          five from-scratch seeds grasped on 63-83% of steps, never lifted
    #          the box above 1.9 cm against a 4 cm latch, and scored 0.002. A
    #          shaping term that can be satisfied without doing the task will be.
    #
    # "latch"  multiply by the binary lift latch. This closes the exploit and
    #          still scores 0.000 across five seeds, because it replaces the
    #          exploit with a cliff: the largest term in the reward is invisible
    #          until the object is 4 cm off the table, so the policy grasps, sits
    #          there (peak lift 0.006-0.016 m) and never discovers transport.
    #          This is the same failure as the lift task's missing `hold` term,
    #          which docs/reward-design.md records, and it was walked into again.
    #
    # "ramp"   multiply by clearance/lift_threshold, clipped to one. Sliding is
    #          still worth exactly nothing -- clearance is zero on the table --
    #          but every millimetre of lift now buys a share of the transport
    #          gradient, so the cliff becomes a hill. This is the default.
    carry_gate: str = "ramp"

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


def load_place_config(path: Optional[str]) -> PlaceRewardConfig:
    """Load weights from JSON, falling back to the documented defaults."""
    if not path:
        return PlaceRewardConfig()
    if not os.path.exists(path):
        raise FileNotFoundError("place reward config not found: " + str(path))
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(PlaceRewardConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError("unknown reward keys in {}: {}".format(path, sorted(unknown)))
    return PlaceRewardConfig(**raw)


@dataclasses.dataclass
class PlaceTerms:
    """Per-term breakdown, so the training plots can show where reward came from."""

    reach: Any
    align: Any
    grasp: Any
    clear: Any
    carry: Any
    settle: Any
    success: Any
    drop: Any
    action: Any

    def total(self):
        return (self.reach + self.align + self.grasp + self.clear + self.carry
                + self.settle + self.success + self.drop + self.action)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def names():
        return [f.name for f in dataclasses.fields(PlaceTerms)]


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------
def place_success_condition(object_pos, goal_pos, grasped, lifted, object_speed,
                            cfg: PlaceRewardConfig):
    """Object resting on the target, hand off it, having been picked up first.

    Five conditions, and each is here because dropping it admits a policy that
    has not done the task:

    ``xy``       within ``goal_tolerance`` of the target patch. Horizontal only:
                 the vertical check is separate and much tighter, because a box
                 hovering 4 cm above the target is not placed and a spherical
                 tolerance would call it placed.
    ``on table`` within ``place_z_tolerance`` of the height it rests at. The
                 goal site is set to exactly that height at reset.
    ``released`` ``grasped`` false. Without it the lift task's answer -- carry
                 it there and keep holding -- would score, and the point of the
                 second task is that it must not.
    ``lifted``   a latch, set when the object cleared the table by
                 ``lift_threshold`` while grasped. Without it the policy can
                 shove the box along the table with two pads touching, which
                 satisfies everything else and never picks anything up. This one
                 is a *task definition*, not a shaping trick: "pick and place"
                 without the pick is a different task.
    ``settled``  ``object_speed`` below tolerance, which rules out the single
                 frame where a box released from height passes through the
                 tolerance band on its way to bouncing.

    Read at the final step of the episode, like the lift task, so a policy that
    places the object and then knocks it off scores zero.
    """
    xp = _xp(object_pos)
    xy = _norm((object_pos - goal_pos)[..., :2], xp)
    dz = xp.abs(object_pos[..., 2] - goal_pos[..., 2])
    return ((xy < cfg.goal_tolerance)
            & (dz < cfg.place_z_tolerance)
            & (grasped < 0.5)
            & (lifted > 0.5)
            & (object_speed < cfg.speed_tolerance))


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------
def place_reward(grip_pos, object_pos, goal_pos, object_start, object_rest_z,
                 grasped, lifted, dropped, object_speed, action,
                 cfg: PlaceRewardConfig):
    """Dense staged reward for pick-and-place.

    The stages, and the reasoning that is *not* shared with the lift reward:

    1. ``reach`` and 2. ``align`` are the lift task's terms verbatim, **gated
       off once the object is placed**. In the lift task the hand ends the
       episode on the object, so a standing pull towards it costs nothing. Here
       the hand has to leave, and an ungated reach term pays the policy to go
       back and touch what it just put down -- which, at these contact forces,
       means nudging it out of tolerance. That gate is the difference between
       the two tasks written in one line.
    3. ``grasp``  small flat bonus for two-pad contact, as before.
    4. ``clear``  height above the resting height, clipped at ``clear_target``
       and gated on ``grasped``. The lift task's version of this term *is* the
       objective; here it is a precondition, so it is worth less and saturates
       sooner.
    5. ``carry``  progress across the table towards the target, measured from
       where the object started rather than as an absolute distance -- the same
       argument as the lift task's ``place`` term, where an absolute distance
       charged the policy a flat penalty for picking the box up at all. Scaled
       by how far the object is off the table, and that scaling is the single
       most important line in this file: ungated the term pays for *sliding*,
       and gated on a binary latch it pays nothing until 4 cm up, which is a
       cliff. See ``carry_gate``; all three settings were trained.
    6. ``settle`` a smooth bump for the object being near the target, on the
       table, with the hand off it. This is the term that pays for *letting go*,
       and it is why the policy does not sit hovering over the target holding
       on: at the target with the box in hand the terms pay about 2.2 a step;
       settled and successful they pay about 11.
    7. ``success`` paid every step the place condition holds, so staying placed
       beats placing and then fiddling.
    8. ``drop``   the object left the table.
    9. ``action`` squared-action penalty.

    Returns ``(reward, terms)``.
    """
    xp = _xp(grip_pos)

    placed = place_success_condition(
        object_pos, goal_pos, grasped, lifted, object_speed, cfg) * 1.0
    engaged = 1.0 - placed

    reach_dist = _norm(object_pos - grip_pos, xp)
    reach = -cfg.w_reach * (1.0 - xp.exp(-reach_dist / cfg.reach_scale)) * engaged

    lateral = _norm((object_pos - grip_pos)[..., :2], xp)
    align = -cfg.w_align * lateral * engaged

    grasp = cfg.w_grasp * grasped

    clearance = xp.clip(object_pos[..., 2] - object_rest_z, 0.0, 1.0)
    clear = cfg.w_clear * xp.clip(clearance, 0.0, cfg.clear_target) * grasped

    start_dist = _norm((object_start - goal_pos)[..., :2], xp)
    goal_xy = _norm((object_pos - goal_pos)[..., :2], xp)
    carry = cfg.w_carry * (start_dist - goal_xy) * grasped
    if cfg.carry_gate == "latch":
        carry = carry * lifted
    elif cfg.carry_gate == "ramp":
        carry = carry * xp.clip(clearance / cfg.lift_threshold, 0.0, 1.0)
    elif cfg.carry_gate != "none":
        raise ValueError("carry_gate must be none, latch or ramp, got "
                         + repr(cfg.carry_gate))

    # On the table, near the target, not in the hand. The height gate is what
    # stops the policy collecting this by releasing from altitude: a box in
    # freefall over the target is near it horizontally and pays nothing.
    on_table = xp.exp(-xp.clip(object_pos[..., 2] - object_rest_z, 0.0, 1.0) / 0.02)
    settle = (cfg.w_settle * xp.exp(-goal_xy / cfg.settle_scale)
              * on_table * (1.0 - grasped))

    success = cfg.w_success * placed
    drop = -cfg.w_drop * dropped
    action_cost = -cfg.w_action * xp.sum(action * action, axis=-1)

    terms = PlaceTerms(
        reach=reach, align=align, grasp=grasp, clear=clear, carry=carry,
        settle=settle, success=success, drop=drop, action=action_cost,
    )
    return terms.total(), terms
