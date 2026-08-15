"""Domain randomisation must actually randomise, and must stay inside its ranges.

A randomisation bug is silent: training still runs, curves still look fine, and
the ablation quietly compares four identical conditions. These tests exist to
make that failure loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.mujoco.grasp_env import make_env
from src.randomisation.domain_rand import (
    LEVELS,
    NOMINAL,
    load_randomisation,
    sample_world,
)

TRAINING_LEVELS = ("none", "low", "medium", "high")


def test_every_shipped_level_loads():
    for level in LEVELS:
        cfg = load_randomisation(level)
        assert cfg.name == level


def test_none_level_returns_the_nominal_world():
    cfg = load_randomisation("none")
    rng = np.random.default_rng(0)
    for _ in range(20):
        world = sample_world(cfg, rng)
        assert world.object_mass == NOMINAL.object_mass
        assert world.object_friction == NOMINAL.object_friction
        assert world.action_latency == 0
        assert world.obs_noise_pos == 0.0


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_levels_vary_their_parameters(level):
    cfg = load_randomisation(level)
    rng = np.random.default_rng(1)
    masses = [sample_world(cfg, rng).object_mass for _ in range(50)]
    assert len(set(masses)) > 40, "mass is not being randomised at level " + level


def test_scale_orders_the_levels():
    """A wider level must produce a wider spread. This is the ablation's premise."""
    spreads = {}
    for level in TRAINING_LEVELS:
        cfg = load_randomisation(level)
        rng = np.random.default_rng(2)
        masses = np.asarray([sample_world(cfg, rng).object_mass for _ in range(400)])
        spreads[level] = float(masses.max() - masses.min())
    assert spreads["none"] == 0.0
    assert spreads["low"] < spreads["medium"] < spreads["high"]


def test_object_size_is_capped_to_what_the_gripper_can_hold():
    """The hand has no wrist rotation; oversized boxes are ungraspable at 45 degrees."""
    for level in LEVELS:
        cfg = load_randomisation(level)
        rng = np.random.default_rng(3)
        sizes = [sample_world(cfg, rng).object_half_size for _ in range(200)]
        assert max(sizes) <= 0.024 + 1e-9
        assert min(sizes) >= 0.014 - 1e-9


def test_shifted_is_outside_the_training_ranges():
    """The held-out distribution has to actually be held out.

    Checked on the parameters the sim-to-real story rests on. ``high`` is
    allowed to overlap: covering part of the shift is the point of training
    with wide randomisation, and the ablation would be meaningless if the
    widest training level could never reach it.
    """
    rng = np.random.default_rng(4)
    shifted = [sample_world(load_randomisation("shifted"), rng) for _ in range(200)]
    for level in ("none", "low", "medium"):
        train = [sample_world(load_randomisation(level), rng) for _ in range(400)]
        assert min(w.object_mass for w in shifted) > max(w.object_mass for w in train)
        assert max(w.object_friction for w in shifted) < min(w.object_friction for w in train)
        assert min(w.obs_noise_pos for w in shifted) > max(w.obs_noise_pos for w in train)


def test_randomisation_reaches_the_simulator():
    """Sampling a world is useless if it never lands in the MjModel."""
    env = make_env("high", seed=0)
    masses, frictions = [], []
    for seed in range(30):
        env.reset(seed=seed)
        masses.append(float(env.model.body_mass[env._object_bid]))
        frictions.append(float(env.model.geom_friction[env._object_gid, 0]))
    env.close()
    assert len(set(masses)) > 25
    assert len(set(frictions)) > 25


def test_unknown_parameter_name_is_rejected():
    from src.randomisation.domain_rand import ParamSpec, RandomisationConfig

    cfg = RandomisationConfig("bad", 1.0, {"not_a_parameter": ParamSpec(0.5, 1.5)})
    with pytest.raises(ValueError):
        sample_world(cfg, np.random.default_rng(0))


def test_unknown_level_name_is_rejected():
    with pytest.raises(FileNotFoundError):
        load_randomisation("enormous")


def test_correlated_sensor_noise_is_correlated_and_not_larger():
    """obs_noise_corr must change the error's structure, not its magnitude.

    A correlated error that is also a bigger one would confound every
    comparison against the independent case, which is the whole point of the
    `measured` and `measured_corr` pair.
    """
    from envs.mujoco.grasp_env import make_env

    for rho, expected in ((0.0, 0.0), (0.9, 0.9)):
        env = make_env("none", seed=0)
        env.reset()
        env.world.obs_noise_pos = 0.006
        env.world.obs_noise_corr = rho
        series = np.array([env._sensor_noise("object", 0.006)[0] for _ in range(4000)])
        lag1 = float(np.corrcoef(series[:-1], series[1:])[0, 1])
        assert abs(lag1 - expected) < 0.08, (rho, lag1)
        assert abs(series.std() - 0.006) < 0.001, (rho, series.std())
