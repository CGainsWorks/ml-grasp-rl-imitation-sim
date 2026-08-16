"""Contract tests for the pick-and-place task.

The interesting tests here are the ones that pin down what "placed" means,
because every one of them corresponds to a policy that would otherwise score
without doing the task: keeping hold of the object at the target, sliding it
there without ever picking it up, dropping it from height and catching the
frame where it passes through the tolerance band.

There is also a test that the lift task is bit-identical to what it was, since
the whole argument for adding a second task is that the first one's numbers
stay valid.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.mujoco.grasp_env import OBS_DIM, PLACE_MAX_TRAVEL, PLACE_MIN_TRAVEL, make_env
from src.policies.scripted_expert import ScriptedExpert, rollout
from src.policies.scripted_place_expert import ScriptedPlaceExpert
from src.rewards.grasp_reward import GraspRewardConfig, grasp_reward
from src.rewards.place_reward import (
    PlaceRewardConfig,
    place_reward,
    place_success_condition,
)


@pytest.fixture(scope="module")
def env():
    e = make_env("none", seed=0, task="place")
    yield e
    e.close()


def test_spaces_are_unchanged_by_the_task(env):
    """The second task reuses the observation, which is why nothing else moved."""
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.act_dim == 4


def test_target_is_on_the_table_and_far_enough_away(env):
    for seed in range(25):
        env.reset(seed=seed)
        travel = float(np.linalg.norm((env._goal - env._object_start)[:2]))
        assert PLACE_MIN_TRAVEL - 1e-9 <= travel <= PLACE_MAX_TRAVEL + 1e-9
        # The target sits at the height the object rests at, which is what lets
        # the success check compare z directly.
        assert env._goal[2] == pytest.approx(env._object_rest_z, abs=1e-9)


def test_holding_the_object_at_the_target_is_not_success():
    """The lift task's answer must not score here. This is the whole point."""
    cfg = PlaceRewardConfig()
    at_target = np.zeros((1, 3))
    placed = place_success_condition(
        at_target, at_target, np.array([1.0]), np.array([1.0]), np.array([0.0]), cfg)
    assert not bool(placed[0])
    released = place_success_condition(
        at_target, at_target, np.array([0.0]), np.array([1.0]), np.array([0.0]), cfg)
    assert bool(released[0])


def test_sliding_without_lifting_is_not_success():
    cfg = PlaceRewardConfig()
    at_target = np.zeros((1, 3))
    slid = place_success_condition(
        at_target, at_target, np.array([0.0]), np.array([0.0]), np.array([0.0]), cfg)
    assert not bool(slid[0])


def test_an_object_in_flight_is_not_placed():
    cfg = PlaceRewardConfig()
    at_target = np.zeros((1, 3))
    falling = place_success_condition(
        at_target, at_target, np.array([0.0]), np.array([1.0]),
        np.array([cfg.speed_tolerance * 5.0]), cfg)
    assert not bool(falling[0])


def test_hovering_over_the_target_pays_less_than_placing():
    """The local optimum the reward is shaped to avoid, priced explicitly."""
    cfg = PlaceRewardConfig()
    goal = np.array([[0.10, 0.0, 0.423]])
    start = np.array([[-0.10, 0.0, 0.423]])
    rest = np.array([0.423])
    action = np.zeros((1, 4))

    hovering, _ = place_reward(
        grip_pos=goal + np.array([[0.0, 0.0, 0.10]]),
        object_pos=goal + np.array([[0.0, 0.0, 0.10]]),
        goal_pos=goal, object_start=start, object_rest_z=rest,
        grasped=np.array([1.0]), lifted=np.array([1.0]), dropped=np.array([0.0]),
        object_speed=np.array([0.0]), action=action, cfg=cfg)
    done, _ = place_reward(
        grip_pos=goal + np.array([[0.0, 0.0, 0.12]]),
        object_pos=goal, goal_pos=goal, object_start=start, object_rest_z=rest,
        grasped=np.array([0.0]), lifted=np.array([1.0]), dropped=np.array([0.0]),
        object_speed=np.array([0.0]), action=action, cfg=cfg)
    assert float(done[0]) > float(hovering[0]) + 5.0


def test_releasing_from_height_over_the_target_pays_almost_nothing():
    """`settle` is gated on height, so a drop test is not a placement."""
    cfg = PlaceRewardConfig()
    goal = np.array([[0.10, 0.0, 0.423]])
    kwargs = dict(goal_pos=goal, object_start=np.array([[-0.10, 0.0, 0.423]]),
                  object_rest_z=np.array([0.423]), grasped=np.array([0.0]),
                  lifted=np.array([1.0]), dropped=np.array([0.0]),
                  object_speed=np.array([0.0]), action=np.zeros((1, 4)), cfg=cfg)
    _, midair = place_reward(grip_pos=goal, object_pos=goal + np.array([[0, 0, 0.12]]),
                             **kwargs)
    _, down = place_reward(grip_pos=goal + np.array([[0, 0, 0.12]]), object_pos=goal,
                           **kwargs)
    assert float(np.asarray(midair.settle)[0]) < 0.01
    assert float(np.asarray(down.settle)[0]) > 2.0


def test_reach_stops_pulling_once_the_object_is_placed():
    """Otherwise the policy is paid to go back and nudge what it put down."""
    cfg = PlaceRewardConfig()
    goal = np.array([[0.10, 0.0, 0.423]])
    kwargs = dict(object_pos=goal, goal_pos=goal,
                  object_start=np.array([[-0.10, 0.0, 0.423]]),
                  object_rest_z=np.array([0.423]), dropped=np.array([0.0]),
                  object_speed=np.array([0.0]), action=np.zeros((1, 4)), cfg=cfg)
    far = goal + np.array([[0.0, 0.0, 0.20]])
    _, placed = place_reward(grip_pos=far, grasped=np.array([0.0]),
                             lifted=np.array([1.0]), **kwargs)
    _, not_yet = place_reward(grip_pos=far, grasped=np.array([0.0]),
                              lifted=np.array([0.0]), **kwargs)
    assert float(np.asarray(placed.reach)[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(np.asarray(not_yet.reach)[0]) < -0.8


def test_expert_places_the_object_on_the_nominal_world(env):
    hits = sum(rollout(env, ScriptedPlaceExpert(), seed=900 + i)["success"]
               for i in range(12))
    assert hits >= 10


def test_expert_actually_picks_the_object_up(env):
    """A scripted expert that slid the box would pass the success test and
    quietly make the lift latch untested."""
    info = rollout(env, ScriptedPlaceExpert(), seed=42)["info"]
    assert info["lifted"] == 1.0


def test_the_place_task_degrades_under_a_shifted_world():
    """Not a performance claim -- a check that the level is doing something,
    since a randomisation config that silently does nothing has happened here
    before."""
    env = make_env("shifted", seed=0, task="place")
    hits = sum(rollout(env, ScriptedPlaceExpert(), seed=800 + i)["success"]
               for i in range(15))
    env.close()
    assert hits < 12


def test_lift_task_is_untouched():
    """Every headline number in the repository depends on this."""
    env = make_env("none", seed=0)
    assert isinstance(env.reward_cfg, GraspRewardConfig)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,)
    # The lift goal is still directly above where the object started.
    assert np.allclose(env._goal[:2], env._object_start[:2], atol=1e-9)
    hits = sum(rollout(env, ScriptedExpert(), seed=700 + i)["success"]
               for i in range(8))
    env.close()
    assert hits >= 7


def test_lift_reward_is_bit_identical_to_its_own_definition():
    """A regression guard on the shared helpers the place reward imports."""
    cfg = GraspRewardConfig()
    r, _ = grasp_reward(
        np.array([[0.0, 0.0, 0.5]]), np.array([[0.0, 0.0, 0.45]]),
        np.array([[0.0, 0.0, 0.55]]), np.array([0.423]), np.array([1.0]),
        np.array([0.0]), np.zeros((1, 4)), cfg)
    assert float(r[0]) == pytest.approx(0.7825362093, abs=1e-9)


def test_place_config_rejects_unknown_keys(tmp_path):
    import json

    from src.rewards.place_reward import load_place_config

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"w_teleport": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_place_config(str(path))


def test_place_task_refuses_a_lift_reward_config():
    with pytest.raises(TypeError):
        make_env("none", seed=0, task="place", reward_cfg=GraspRewardConfig())


def test_the_carry_gate_prices_sliding_at_nothing_and_lifting_at_something():
    """The three gate settings, and why only one of them is the default.

    Sliding means clearance zero, which the ramp and the latch both price at
    zero and the ungated version pays in full. Half-lifted means the ramp pays
    half while the latch still pays nothing -- that is the difference between a
    hill and a cliff, and it is the difference between 0.000 and a working
    policy."""
    from src.rewards.place_reward import PlaceRewardConfig, place_reward

    goal = np.array([[0.10, 0.0, 0.423]])

    def carry(gate, clearance, lifted):
        _, terms = place_reward(
            grip_pos=goal, object_pos=goal + np.array([[0.10, 0.0, clearance]]),
            goal_pos=goal, object_start=np.array([[-0.10, 0.0, 0.423]]),
            object_rest_z=np.array([0.423]), grasped=np.array([1.0]),
            lifted=np.array([lifted]), dropped=np.array([0.0]),
            object_speed=np.array([0.0]), action=np.zeros((1, 4)),
            cfg=PlaceRewardConfig(carry_gate=gate))
        return float(np.asarray(terms.carry)[0])

    assert carry("none", 0.0, 0.0) > 0.5          # sliding pays, which is the bug
    assert carry("latch", 0.0, 0.0) == 0.0
    assert carry("ramp", 0.0, 0.0) == 0.0
    half = PlaceRewardConfig().lift_threshold / 2.0
    assert carry("latch", half, 0.0) == 0.0       # the cliff
    assert carry("ramp", half, 0.0) == pytest.approx(carry("none", half, 0.0) / 2.0)


def test_an_unknown_carry_gate_is_rejected():
    from src.rewards.place_reward import PlaceRewardConfig, place_reward

    with pytest.raises(ValueError):
        place_reward(
            grip_pos=np.zeros((1, 3)), object_pos=np.zeros((1, 3)),
            goal_pos=np.zeros((1, 3)), object_start=np.zeros((1, 3)),
            object_rest_z=np.array([0.0]), grasped=np.array([1.0]),
            lifted=np.array([0.0]), dropped=np.array([0.0]),
            object_speed=np.array([0.0]), action=np.zeros((1, 4)),
            cfg=PlaceRewardConfig(carry_gate="teleport"))


def test_the_travel_ladder_samples_the_range_it_is_given():
    """The rungs exist to decompose the task; a rung that quietly sampled the
    default range would make the decomposition meaningless."""
    from envs.mujoco.grasp_env import PLACE_TRAVEL_LADDER

    for rung, (lo, hi) in PLACE_TRAVEL_LADDER.items():
        e = make_env("none", seed=0, task="place", travel_range=(lo, hi))
        for seed in range(15):
            e.reset(seed=seed)
            travel = float(np.linalg.norm((e._goal - e._object_start)[:2]))
            assert lo - 1e-9 <= travel <= hi + 1e-9, (rung, travel)
        e.close()


def test_a_zero_travel_target_still_requires_a_pick_and_a_release():
    """The `none` rung removes transport, not the task. If the object counted as
    placed where it already sits, the rung would measure nothing."""
    from envs.mujoco.grasp_env import PLACE_TRAVEL_LADDER

    e = make_env("none", seed=0, task="place",
                 travel_range=PLACE_TRAVEL_LADDER["none"])
    obs, info = e.reset(seed=1)
    assert not info["is_success"]      # sitting on the target is not placed
    assert info["lifted"] == 0.0
    e.close()


def test_settle_pays_nothing_for_an_object_that_was_never_picked_up():
    """Found by the travel ladder rather than by inspection: at zero travel the
    object starts on the target, and an ungated `settle` paid +0.96 a step for
    doing nothing. Five seeds learned to do nothing, at a grasp rate of 0.000."""
    cfg = PlaceRewardConfig()
    goal = np.array([[0.10, 0.0, 0.423]])
    kwargs = dict(grip_pos=goal + np.array([[0.0, 0.0, 0.2]]), object_pos=goal,
                  goal_pos=goal, object_start=goal,
                  object_rest_z=np.array([0.423]), grasped=np.array([0.0]),
                  dropped=np.array([0.0]), object_speed=np.array([0.0]),
                  action=np.zeros((1, 4)), cfg=cfg)
    _, untouched = place_reward(lifted=np.array([0.0]), **kwargs)
    _, placed = place_reward(lifted=np.array([1.0]), **kwargs)
    assert float(np.asarray(untouched.settle)[0]) == 0.0
    assert float(np.asarray(placed.settle)[0]) > 2.0


def test_approach_pays_for_carrying_over_the_target_but_not_for_sliding():
    """The lift task's `hold` term transplanted. It has to be unreachable by
    shoving the box onto the target, or it reintroduces the sliding exploit."""
    cfg = PlaceRewardConfig()
    goal = np.array([[0.10, 0.0, 0.423]])
    kwargs = dict(grip_pos=goal, goal_pos=goal,
                  object_start=np.array([[-0.10, 0.0, 0.423]]),
                  object_rest_z=np.array([0.423]), lifted=np.array([1.0]),
                  dropped=np.array([0.0]), object_speed=np.array([0.0]),
                  action=np.zeros((1, 4)), cfg=cfg)
    _, carried = place_reward(object_pos=goal + np.array([[0, 0, 0.08]]),
                              grasped=np.array([1.0]), **kwargs)
    _, slid = place_reward(object_pos=goal, grasped=np.array([1.0]), **kwargs)
    _, elsewhere = place_reward(object_pos=goal + np.array([[0.2, 0, 0.08]]),
                                grasped=np.array([1.0]), **kwargs)
    assert float(np.asarray(carried.approach)[0]) > 2.5
    assert float(np.asarray(slid.approach)[0]) == 0.0
    # 20 cm away is nearly four scale lengths, so the bump has decayed to a few
    # percent -- present, so there is still a gradient to follow, but nowhere
    # near enough to be worth collecting instead of finishing.
    assert float(np.asarray(elsewhere.approach)[0]) < 0.15
    assert (float(np.asarray(elsewhere.approach)[0])
            < 0.05 * float(np.asarray(carried.approach)[0]))


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        make_env("none", seed=0, task="stack")
