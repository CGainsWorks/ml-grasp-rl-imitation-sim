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
    w_clear: float = 4.0          # lift the object clear of the table before carrying
    w_carry: float = 6.0          # progress across the table, once grasped
    w_settle: float = 3.0         # object down, near the target, hand off it
    # Carrying the object *over* the target, still in the hand and still up.
    #
    # This is the lift task's `hold` term transplanted, and it is here because
    # of a measurement rather than a hunch. Decomposing what the scripted expert
    # actually earns per step:
    #
    #     lift-and-hold    5.948/step positive, 51.8% from terms that only pay
    #                      once the task is complete
    #     pick-and-place   3.202/step positive, 80.7% from such terms
    #
    # The lift task's shaping is worth 2.87 a step on its own -- `hold` alone
    # pays 1.73 -- and it rises continuously all the way to the goal. The place
    # task's shaping was worth 0.62, and nothing in it paid *more* as the policy
    # got closer to finishing. A reward that is 81% terminal is a sparse reward
    # with decorations, which is exactly what four failed designs were saying.
    #
    # `approach` fills the gap between carrying and releasing: it pays a smooth
    # bump for having the object above the target, off the table, in the hand.
    w_approach: float = 3.0
    # Events
    w_success: float = 8.0        # paid every step the place condition holds
    w_drop: float = 5.0           # paid once if the object leaves the table
    # Regularisers
    w_action: float = 0.02

    # Thresholds
    # Matched to the lift task's w_lift and lift_target exactly, and not by
    # coincidence. Getting the object off the table is the same sub-problem
    # here as it is there, and SAC solves it from scratch there at 4.0 x 0.12 =
    # 0.48 a step. At the 3.0 x 0.06 = 0.18 this reward started with, five
    # seeds grasped and sat on the table with a peak lift of 0.009 m -- the
    # gradient existed and was too shallow to follow. Rather than search for a
    # weight, take the one the other task already established for the identical
    # sub-task.
    clear_target: float = 0.12    # metres of clearance that counts as carried
    lift_threshold: float = 0.04  # clearance that latches "this was picked up"
    goal_tolerance: float = 0.05  # metres, radius of the target patch
    place_z_tolerance: float = 0.02   # how close to its resting height it must sit
    speed_tolerance: float = 0.05     # m/s, so a box in flight is not "placed"
    reach_scale: float = 0.10
    settle_scale: float = 0.05
    approach_scale: float = 0.06
    # How `approach` measures "close to finishing". Both were run.
    #
    # "hover"  distance in the horizontal plane only, gated on the clearance
    #          ramp. This was the first version and it half-works: the policies
    #          started lifting -- peak lift went from 0.010 m to 0.10-0.19 m,
    #          which is the whole carry -- and then hovered over the target
    #          holding on, because that is where the term is maximised. Success
    #          stayed at 0.000. It bought four fifths of the behaviour and paid
    #          for stopping just before the end.
    #
    # "goal"   full three-dimensional distance to the target, gated on the lift
    #          latch instead of on current clearance. The maximum is now exactly
    #          where the object should be released -- on the target, in the hand
    #          -- rather than ten centimetres above it, and the term survives the
    #          descent because the latch does not reset. Hovering at carry height
    #          pays 0.56 a step against 3.0 for bringing it down. From scratch
    #          this scores 0.000 too -- and, unlike "hover", the policies stop
    #          lifting again (peak 0.010-0.022 m). The two modes each buy a
    #          different half of the behaviour and neither buys both.
    #
    # "both"   the sum: a half-weight hover bump gated on current clearance, so
    #          it pays during the very first lift the way "hover" did, plus the
    #          full goal bump gated on the latch, so it survives the descent the
    #          way "goal" does. This is the last variant of this shaping family
    #          that is worth trying; if a reward whose dense part rises
    #          monotonically through lift, carry and descent still does not
    #          produce the behaviour, the conclusion is about the family.
    approach_mode: str = "both"

    # Whether success requires the object to have been picked up.
    #
    # True is the task definition and it stays the default: "pick and place"
    # without the pick is a different task, and without this a policy scores by
    # shoving the box along the table.
    #
    # This comment used to say the latch made hindsight experience replay
    # inapplicable: relabelling can move the target to wherever the box ended
    # up, but cannot retroactively pick the box up. That was wrong, and the
    # error is worth leaving on the record because it was confidently argued
    # from a real measurement.
    #
    # The measurement was right -- 16 000 relabelled transitions, zero
    # successes -- and the explanation was not. `train_her.py` stores `lifted`
    # per transition and recomputes this condition with it, so the latch does
    # travel with the relabelled transition. What produced the zero is that a
    # sparse from-scratch policy never lifts the box at all, so every relabelled
    # goal is scored against `lifted = 0`. "The latch is history" was confused
    # with "the latch is unavailable".
    #
    # Give the policy start states where the lift has already happened and
    # relabelling fires on half of all transitions
    # (`scripts/her_relabel_probe.py`), and sparse reward plus hindsight plus an
    # annealed start curriculum reaches 0.944 -- above the demonstration-seeded
    # pipeline, on a task no shaped reward here ever solved from scratch.
    #
    # False exists so that claim can be tested rather than asserted, and it is
    # not a task anyone should report numbers on.
    success_requires_lift: bool = True

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
    approach: Any
    settle: Any
    success: Any
    drop: Any
    action: Any

    def total(self):
        return (self.reach + self.align + self.grasp + self.clear + self.carry
                + self.approach + self.settle + self.success + self.drop
                + self.action)

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
    ok = ((xy < cfg.goal_tolerance)
          & (dz < cfg.place_z_tolerance)
          & (grasped < 0.5)
          & (object_speed < cfg.speed_tolerance))
    if cfg.success_requires_lift:
        ok = ok & (lifted > 0.5)
    return ok


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
       table, with the hand off it, **and having been picked up first**. That
       last gate looks redundant on the full task -- the object starts far from
       the target, so an unlifted object is never near it -- and it is not. The
       travel ladder in ``experiments/place_ladder.py`` shrinks the distance to
       zero to isolate transport from release, and at zero travel an ungated
       ``settle`` pays **+0.96 a step for doing nothing at all**, because the
       object begins on the target. Five seeds duly did nothing, at a grasp rate
       of exactly 0.000. That is the fifth term in this repository to be
       satisfiable without doing the task, and the first found by an experiment
       designed to measure something else. This is the term that pays for
       *letting go*,
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

    # Over the target, up, and still holding it. Gated on the same clearance
    # ramp as `carry`, so it cannot be collected by sliding the box onto the
    # target and sitting on it.
    if cfg.approach_mode == "hover":
        lift_ramp = xp.clip(clearance / cfg.lift_threshold, 0.0, 1.0)
        approach = (cfg.w_approach * xp.exp(-goal_xy / cfg.approach_scale)
                    * grasped * lift_ramp)
    elif cfg.approach_mode == "goal":
        goal_3d = _norm(object_pos - goal_pos, xp)
        approach = (cfg.w_approach * xp.exp(-goal_3d / cfg.approach_scale)
                    * grasped * lifted)
    elif cfg.approach_mode == "both":
        # The hover half is gated on *current* clearance so it pays during the
        # first lift, before any latch has been set; the goal half is gated on
        # the latch so it survives the descent. Sliding earns neither: clearance
        # is zero on the table and the latch has never fired.
        lift_ramp = xp.clip(clearance / cfg.lift_threshold, 0.0, 1.0)
        goal_3d = _norm(object_pos - goal_pos, xp)
        approach = cfg.w_approach * grasped * (
            0.5 * xp.exp(-goal_xy / cfg.approach_scale) * lift_ramp
            + xp.exp(-goal_3d / cfg.approach_scale) * lifted)
    else:
        raise ValueError("approach_mode must be hover, goal or both, got "
                         + repr(cfg.approach_mode))

    # On the table, near the target, not in the hand. The height gate is what
    # stops the policy collecting this by releasing from altitude: a box in
    # freefall over the target is near it horizontally and pays nothing.
    on_table = xp.exp(-xp.clip(object_pos[..., 2] - object_rest_z, 0.0, 1.0) / 0.02)
    settle = (cfg.w_settle * xp.exp(-goal_xy / cfg.settle_scale)
              * on_table * (1.0 - grasped) * lifted)

    success = cfg.w_success * placed
    drop = -cfg.w_drop * dropped
    action_cost = -cfg.w_action * xp.sum(action * action, axis=-1)

    terms = PlaceTerms(
        reach=reach, align=align, grasp=grasp, clear=clear, carry=carry,
        approach=approach, settle=settle, success=success, drop=drop,
        action=action_cost,
    )
    return terms.total(), terms
