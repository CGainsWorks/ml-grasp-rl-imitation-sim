"""The Isaac Lab port must fail honestly when Isaac is not installed.

This file cannot test the port's behaviour -- there is no Isaac Sim here. What
it can test is that the module imports for inspection, that it does not
pretend to work, and that it really does share the reward and randomisation
definitions with the MuJoCo task rather than carrying copies of them.
"""

from __future__ import annotations

import inspect

import pytest

import envs.isaac.grasp_task as isaac_task


def test_module_imports_without_isaac_sim():
    """Importing must not explode: the file is meant to be readable anywhere."""
    assert isaac_task.OBS_DIM == 32
    assert isaac_task.ACT_DIM == 4


def test_it_refuses_to_construct_without_isaac():
    """No silent stub. Asking for the environment without Isaac must say so."""
    if isaac_task.ISAAC_AVAILABLE:  # pragma: no cover - not the CI path
        pytest.skip("Isaac Lab is installed; this test covers the absent case")
    with pytest.raises(RuntimeError, match="Isaac Lab is not installed"):
        isaac_task.GraspTask(cfg=None)


def test_reward_is_imported_not_reimplemented():
    """The port must call the shared reward, not carry its own copy."""
    from src.rewards.grasp_reward import grasp_reward, success_condition

    assert isaac_task.grasp_reward is grasp_reward
    assert isaac_task.success_condition is success_condition

    source = inspect.getsource(isaac_task)
    # A second implementation would have to name the weights again.
    assert "w_lift" not in source
    assert "w_place" not in source


def test_randomisation_ranges_come_from_the_shared_configs():
    from src.randomisation.domain_rand import CONFIG_DIR

    assert isaac_task.CONFIG_DIR == CONFIG_DIR
    ranges = isaac_task.load_randomisation_ranges("medium")
    assert ranges["name"] == "medium"
    assert "object_mass" in ranges["params"]


def test_scene_constants_match_the_mujoco_task():
    from envs.mujoco.grasp_env import ACT_DIM, OBS_DIM, TABLE_HEIGHT

    assert isaac_task.TABLE_HEIGHT == TABLE_HEIGHT
    assert isaac_task.OBS_DIM == OBS_DIM
    assert isaac_task.ACT_DIM == ACT_DIM
