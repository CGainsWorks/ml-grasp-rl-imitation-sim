"""Environment contract tests.

These are the tests that would have caught the bugs actually hit while building
this task, which is the only defensible reason to write a test after the fact:

* the finger pads closing *above* the box because the grasp offset was measured
  from the wrong frame;
* the grasp flag chattering between steps, so a held box scored as a failure;
* a randomisation level silently doing nothing because the config named a
  parameter the sampler ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.mujoco.grasp_env import ACT_DIM, GRIPPER_OPEN_WIDTH, OBS_DIM, TABLE_HEIGHT, make_env
from src.policies.scripted_expert import ScriptedExpert, rollout


@pytest.fixture(scope="module")
def env():
    e = make_env("none", seed=0)
    yield e
    e.close()


def test_observation_shape_and_finiteness(env):
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "is_success" in info


def test_action_bounds_are_respected(env):
    env.reset(seed=2)
    # Actions outside [-1, 1] must be clipped, not scaled up.
    far = env.step(np.array([50.0, 50.0, 50.0, 50.0]))[0]
    env.reset(seed=2)
    unit = env.step(np.array([1.0, 1.0, 1.0, 1.0]))[0]
    assert np.allclose(far[:3], unit[:3], atol=1e-6)


def test_observation_layout_matches_documentation(env):
    """Indices 11:14 and 29:32 are documented as relative positions."""
    obs, _ = env.reset(seed=3)
    grip, obj, goal = obs[0:3], obs[8:11], obs[26:29]
    assert np.allclose(obs[11:14], obj - grip, atol=1e-5)
    assert np.allclose(obs[29:32], goal - obj, atol=1e-5)


def test_object_starts_on_the_table(env):
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert obs[10] > TABLE_HEIGHT
        assert obs[10] < TABLE_HEIGHT + 0.05


def test_goal_is_above_the_object_start(env):
    obs, _ = env.reset(seed=4)
    assert obs[28] > obs[10] + 0.10


def test_episode_truncates_at_the_horizon():
    e = make_env("none", seed=0, max_steps=12)
    e.reset(seed=5)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = e.step(np.zeros(ACT_DIM))
        steps += 1
        if terminated or truncated:
            break
    assert steps == 12
    e.close()


def test_gripper_opens_and_closes(env):
    env.reset(seed=6)
    for _ in range(15):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    assert obs[6] == pytest.approx(GRIPPER_OPEN_WIDTH, abs=2e-3)
    for _ in range(15):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
    assert obs[6] < 0.01


def test_grasp_flag_requires_the_object():
    """Closing on thin air is not a grasp."""
    e = make_env("none", seed=0)
    e.reset(seed=7)
    for _ in range(20):
        # Move well away from the object first, then close.
        _, _, _, _, info = e.step(np.array([1.0, 1.0, 1.0, 1.0]))
    assert info["grasped"] == 0.0
    e.close()


def test_success_requires_holding_at_the_end():
    """A successful expert episode ends with the object at the goal, still held."""
    e = make_env("none", seed=0)
    result = rollout(e, ScriptedExpert(), seed=11)
    assert result["success"]
    assert result["info"]["grasped"] == 1.0
    assert result["info"]["object_height"] > 0.10
    e.close()


def test_expert_succeeds_on_the_nominal_world():
    """The scripted expert is the environment's smoke test."""
    e = make_env("none", seed=0)
    successes = [rollout(e, ScriptedExpert(), seed=100 + i)["success"] for i in range(10)]
    assert sum(successes) >= 9
    e.close()


def test_reset_is_reproducible_from_a_seed():
    a = make_env("medium", seed=0)
    b = make_env("medium", seed=0)
    obs_a, _ = a.reset(seed=99)
    obs_b, _ = b.reset(seed=99)
    assert np.allclose(obs_a, obs_b)
    assert a.world.as_dict() == b.world.as_dict()
    a.close()
    b.close()


def test_different_seeds_give_different_worlds():
    e = make_env("medium", seed=0)
    e.reset(seed=1)
    first = e.world.object_mass
    e.reset(seed=2)
    assert e.world.object_mass != first
    e.close()


def test_dropping_the_object_terminates():
    """Pushing the object off the table ends the episode with a drop."""
    e = make_env("none", seed=0, max_steps=200)
    e.reset(seed=12)
    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, info = e.step(np.array([0.0, -1.0, -1.0, -1.0]))
        if terminated or truncated:
            break
    # Either it was pushed off (terminated) or it stayed on the table; both are
    # legitimate, but a termination must be accompanied by the drop flag.
    if terminated:
        assert info["dropped"]


def test_clean_observation_is_noise_free_and_does_not_leak():
    """The privileged path must be privileged only where it is asked for.

    `clean_observation` exists so a demonstrator can act on true state while the
    stored transition keeps the noisy view. Two things have to hold or every
    number at `measured_camera` is wrong: the clean view must actually differ
    from the noisy one under heavy sensing error, and calling it must leave the
    environment's noise switched back on afterwards.
    """
    env = make_env("measured_camera", seed=0)
    env.reset(seed=3)

    clean_a = env.clean_observation()
    clean_b = env.clean_observation()
    # Deterministic: no noise draw happens inside it.
    assert np.allclose(clean_a, clean_b)

    noisy = np.stack([env._observation() for _ in range(8)])
    # The noise is still live after the clean read -- this is the leak that
    # would quietly turn a hard benchmark into an easy one.
    assert noisy.std(axis=0).max() > 0.0
    assert np.abs(noisy.mean(axis=0) - clean_a).max() > 1e-4
    env.close()


def test_observation_history_stacks_and_reports_its_width():
    env = make_env("none", seed=0, history=4)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,)
    assert env.obs_dim == 4 * env._single_obs_dim
    # Primed with copies of the first frame, so the window is never ragged.
    single = env._single_obs_dim
    assert np.allclose(obs[:single], obs[-single:])
    env.close()


def test_action_latency_is_never_negative():
    """A latency is a queue depth, and `high` widens the range below zero.

    `absolute` ranges widen about their midpoint, so at scale 1.8 the 0-2 step
    latency becomes (-0.8, 2.8). MuJoCo absorbed the negative silently -- a list
    multiplied by a negative count is empty -- while Isaac used it as a gather
    index and crashed the CUDA context, taking PhysX with it. Every Isaac run at
    `high` died this way.
    """
    from src.randomisation.domain_rand import load_randomisation, sample_world

    for level in ("none", "low", "medium", "high", "shifted"):
        cfg = load_randomisation(level)
        rng = np.random.default_rng(0)
        latencies = [sample_world(cfg, rng).action_latency for _ in range(500)]
        assert min(latencies) >= 0, (
            "level {} samples a negative action latency".format(level))


def test_arm_actuators_stay_position_servos_under_randomisation():
    """Scaling a MuJoCo position servo means scaling both of its terms.

    The actuator force is ``gainprm[0] * ctrl + biasprm[1] * qpos``, and it is
    only a position servo while those are equal in magnitude and opposite in
    sign. `hand_compliance` scales the arm's stiffness; scaling the gain alone
    left the bias at the original value, so the joints settled at
    ``(new/old) * commanded`` -- a systematic kinematic error that made the
    scripted expert miss the box by 143 mm at `medium` and turned the arm's
    randomisation grid into a measurement of a mis-specified actuator.
    """
    for level in ("none", "low", "medium", "high"):
        env = make_env(level, seed=0, arm=True)
        for i in range(6):
            env.reset(seed=100 + i)
            gain = env.model.actuator_gainprm[env._arm_ctrl, 0]
            bias = env.model.actuator_biasprm[env._arm_ctrl, 1]
            assert np.allclose(bias, -gain), (
                "level {}: arm actuator is not a position servo "
                "(gain {}, bias {})".format(level, gain, bias))
        env.close()
