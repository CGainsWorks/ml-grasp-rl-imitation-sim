"""The ROS 2 node's replay fixture still describes this environment and policy.

``ros2/grasp_policy_server`` is tested in a ROS container against a recorded
episode. That test proves the node reproduces the recording; this one proves
the recording is still about the code in this repository: the same policy file,
the same observation width and layout and an episode that starts where the
environment starts it today.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from envs.mujoco.grasp_env import ACT_DIM, OBS_DIM, make_env

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(REPO, "ros2", "grasp_policy_server")
STREAM = os.path.join(PACKAGE, "test", "data", "replay_bcrl_medium_s0.json")

sys.path.insert(0, PACKAGE)
from grasp_policy_server.core import PolicyRunner  # noqa: E402


def _stream():
    with open(STREAM, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_fixture_names_the_committed_policy():
    stream = _stream()
    with open(os.path.join(REPO, stream["policy"]), "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == stream["policy_sha256"]
    assert stream["simulated"] is True


def test_fixture_matches_the_observation_interface():
    stream = _stream()
    runner = PolicyRunner(os.path.join(REPO, stream["policy"]))
    assert stream["observation_dim"] == OBS_DIM == runner.observation_dim
    assert stream["action_dim"] == ACT_DIM == runner.action_dim
    assert all(len(o) == OBS_DIM for o in stream["observations"])
    assert all(len(a) == ACT_DIM for a in stream["actions"])


def test_fixture_starts_where_the_environment_does():
    stream = _stream()
    env = make_env(stream["randomisation"], seed=1234)
    obs, _ = env.reset(seed=stream["episode_seed"])
    np.testing.assert_allclose(np.asarray(obs, dtype=np.float32),
                               np.asarray(stream["observations"][0], dtype=np.float32),
                               atol=1e-5)


def test_policy_reproduces_the_recorded_actions():
    stream = _stream()
    runner = PolicyRunner(os.path.join(REPO, stream["policy"]))
    got = np.asarray([runner.act(o) for o in stream["observations"]])
    np.testing.assert_allclose(got, np.asarray(stream["actions"]), atol=1e-5)
