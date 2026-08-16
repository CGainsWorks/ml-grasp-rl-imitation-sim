"""The reward must be identical under numpy and torch.

This is the test that gives the Isaac Lab port its meaning. That port cannot be
run here, so the strongest available claim is "it computes the same reward on
the same state" -- and this test is what makes that claim checkable rather than
aspirational. It runs in CI on random states, both backends, single precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rewards.grasp_reward import (
    GraspRewardConfig,
    RewardTerms,
    dropped_condition,
    grasp_reward,
    success_condition,
)

torch = pytest.importorskip("torch")


def _random_batch(n: int, rng: np.random.Generator):
    grip = rng.uniform(-0.3, 0.3, size=(n, 3)) + np.array([0.0, 0.0, 0.55])
    obj = rng.uniform(-0.3, 0.3, size=(n, 3)) + np.array([0.0, 0.0, 0.45])
    goal = rng.uniform(-0.1, 0.1, size=(n, 3)) + np.array([0.0, 0.0, 0.55])
    rest = np.full(n, 0.422)
    grasped = rng.integers(0, 2, size=n).astype(float)
    dropped = rng.integers(0, 2, size=n).astype(float)
    action = rng.uniform(-1.0, 1.0, size=(n, 4))
    return grip, obj, goal, rest, grasped, dropped, action


def test_reward_matches_between_numpy_and_torch():
    rng = np.random.default_rng(0)
    cfg = GraspRewardConfig()
    args = _random_batch(256, rng)

    np_reward, np_terms = grasp_reward(*args, cfg=cfg)
    torch_args = [torch.as_tensor(a, dtype=torch.float64) for a in args]
    torch_reward, torch_terms = grasp_reward(*torch_args, cfg=cfg)

    assert np.allclose(np_reward, torch_reward.numpy(), atol=1e-6)
    for name in RewardTerms.names():
        a = np.asarray(getattr(np_terms, name), dtype=float)
        b = np.asarray(getattr(torch_terms, name).numpy(), dtype=float)
        assert np.allclose(np.broadcast_to(a, b.shape), b, atol=1e-6), name


def test_success_condition_matches_between_backends():
    rng = np.random.default_rng(1)
    cfg = GraspRewardConfig()
    _, obj, goal, _, grasped, _, _ = _random_batch(128, rng)
    np_ok = success_condition(obj, goal, grasped, cfg)
    torch_ok = success_condition(
        torch.as_tensor(obj), torch.as_tensor(goal), torch.as_tensor(grasped), cfg
    )
    assert np.array_equal(np_ok, torch_ok.numpy())


def test_success_needs_both_proximity_and_grasp():
    cfg = GraspRewardConfig()
    at_goal = np.array([[0.0, 0.0, 0.55]])
    goal = np.array([[0.0, 0.0, 0.55]])
    assert success_condition(at_goal, goal, np.array([1.0]), cfg)[0]
    assert not success_condition(at_goal, goal, np.array([0.0]), cfg)[0]
    far = np.array([[0.0, 0.0, 0.40]])
    assert not success_condition(far, goal, np.array([1.0]), cfg)[0]


def test_lift_term_saturates_at_the_target():
    cfg = GraspRewardConfig()
    grip = np.array([[0.0, 0.0, 0.55]])
    goal = np.array([[0.0, 0.0, 0.55]])
    rest = np.array([0.42])
    action = np.zeros((1, 4))

    def lift_at(height):
        _, terms = grasp_reward(
            grip, np.array([[0.0, 0.0, height]]), goal, rest,
            np.array([1.0]), np.array([0.0]), action, cfg,
        )
        return float(np.asarray(terms.lift).reshape(-1)[0])

    assert lift_at(0.42) == pytest.approx(0.0)
    assert lift_at(0.42 + cfg.lift_target) == pytest.approx(cfg.w_lift * cfg.lift_target)
    # Beyond the target the term stops paying: flying the box to the ceiling
    # earns nothing extra.
    assert lift_at(0.42 + 2 * cfg.lift_target) == pytest.approx(cfg.w_lift * cfg.lift_target)


def test_drop_penalty_is_negative_and_one_off():
    cfg = GraspRewardConfig()
    grip = np.array([[0.0, 0.0, 0.5]])
    obj = np.array([[0.0, 0.0, 0.3]])
    goal = np.array([[0.0, 0.0, 0.55]])
    _, terms = grasp_reward(grip, obj, goal, np.array([0.42]), np.array([0.0]),
                            np.array([1.0]), np.zeros((1, 4)), cfg)
    assert float(np.asarray(terms.drop).reshape(-1)[0]) == pytest.approx(-cfg.w_drop)


def test_dropped_condition_uses_the_table_height():
    below = np.array([[0.0, 0.0, 0.30]])
    above = np.array([[0.0, 0.0, 0.45]])
    assert dropped_condition(below, 0.40)[0]
    assert not dropped_condition(above, 0.40)[0]


# --------------------------------------------------------------------------
# The place task shares this contract, because the Isaac port shares the file
# --------------------------------------------------------------------------
def _random_place_batch(n: int, rng: np.random.Generator):
    grip = rng.uniform(-0.3, 0.3, size=(n, 3)) + np.array([0.0, 0.0, 0.55])
    obj = rng.uniform(-0.3, 0.3, size=(n, 3)) + np.array([0.0, 0.0, 0.45])
    goal = rng.uniform(-0.15, 0.15, size=(n, 3)) + np.array([0.0, 0.0, 0.423])
    start = rng.uniform(-0.15, 0.15, size=(n, 3)) + np.array([0.0, 0.0, 0.423])
    rest = np.full(n, 0.423)
    grasped = rng.integers(0, 2, size=n).astype(float)
    lifted = rng.integers(0, 2, size=n).astype(float)
    dropped = rng.integers(0, 2, size=n).astype(float)
    speed = rng.uniform(0.0, 0.3, size=n)
    action = rng.uniform(-1.0, 1.0, size=(n, 4))
    return grip, obj, goal, start, rest, grasped, lifted, dropped, speed, action


@pytest.mark.parametrize("mode", ["hover", "goal", "both"])
def test_place_reward_matches_between_numpy_and_torch(mode):
    """Every `approach_mode` has to agree across backends, not just the default.

    The Isaac port selects the mode from the same config object, so a mode that
    only worked under numpy would make the ported task quietly different from
    the one the MuJoCo numbers came from.
    """
    from src.rewards.place_reward import PlaceRewardConfig, PlaceTerms, place_reward

    rng = np.random.default_rng(7)
    cfg = PlaceRewardConfig(approach_mode=mode)
    args = _random_place_batch(256, rng)

    np_reward, np_terms = place_reward(*args, cfg=cfg)
    torch_args = [torch.as_tensor(a, dtype=torch.float64) for a in args]
    torch_reward, torch_terms = place_reward(*torch_args, cfg=cfg)

    assert np.allclose(np_reward, torch_reward.numpy(), atol=1e-6)
    for name in PlaceTerms.names():
        a = np.asarray(getattr(np_terms, name), dtype=float)
        b = np.asarray(getattr(torch_terms, name).numpy(), dtype=float)
        assert np.allclose(np.broadcast_to(a, b.shape), b, atol=1e-6), name


def test_place_success_condition_matches_between_backends():
    from src.rewards.place_reward import PlaceRewardConfig, place_success_condition

    rng = np.random.default_rng(8)
    cfg = PlaceRewardConfig()
    _, obj, goal, _, _, grasped, lifted, _, speed, _ = _random_place_batch(128, rng)
    np_ok = place_success_condition(obj, goal, grasped, lifted, speed, cfg)
    torch_ok = place_success_condition(
        torch.as_tensor(obj), torch.as_tensor(goal), torch.as_tensor(grasped),
        torch.as_tensor(lifted), torch.as_tensor(speed), cfg,
    )
    assert np.array_equal(np_ok, torch_ok.numpy())


def test_place_carry_gates_match_between_backends():
    """All three `carry_gate` settings, since two of them exist only so the
    failed runs on disk stay reproducible and would otherwise never be tested."""
    from src.rewards.place_reward import PlaceRewardConfig, place_reward

    rng = np.random.default_rng(9)
    args = _random_place_batch(64, rng)
    torch_args = [torch.as_tensor(a, dtype=torch.float64) for a in args]
    for gate in ("none", "latch", "ramp"):
        cfg = PlaceRewardConfig(carry_gate=gate)
        np_reward, _ = place_reward(*args, cfg=cfg)
        torch_reward, _ = place_reward(*torch_args, cfg=cfg)
        assert np.allclose(np_reward, torch_reward.numpy(), atol=1e-6), gate
