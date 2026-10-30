"""The policy wrapper, without ROS: loading, validation and the recorded stream."""

from __future__ import annotations

import json
import math
import os
import shutil

import pytest

from grasp_policy_server.core import (
    ObservationError,
    PolicyLoadError,
    PolicyRunner,
    default_meta_path,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
POLICY = os.path.join(REPO, "experiments", "policies", "bcrl_medium_s0.ts.pt")
STREAM = os.path.join(HERE, "data", "replay_bcrl_medium_s0.json")


@pytest.fixture(scope="module")
def runner():
    return PolicyRunner(POLICY)


@pytest.fixture(scope="module")
def stream():
    with open(STREAM, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_meta_path_sits_next_to_the_policy():
    assert default_meta_path("a/b/p.ts.pt") == "a/b/p.meta.json"


def test_widths_come_from_the_metadata(runner):
    assert runner.observation_dim == 32
    assert runner.action_dim == 4
    names = [row["name"] for row in runner.layout("observation_layout")]
    assert names[0] == "grip_pos" and names[-1] == "goal_rel_pos"


def test_recorded_stream_reproduces(runner, stream):
    """Every recorded observation maps to its recorded action."""
    assert len(stream["observations"]) == len(stream["actions"]) == stream["steps"]
    worst = 0.0
    for obs, expected in zip(stream["observations"], stream["actions"]):
        got = runner.act(obs)
        worst = max(worst, max(abs(a - b) for a, b in zip(got, expected)))
    assert worst < 1e-5


def test_wrong_width_is_rejected(runner):
    with pytest.raises(ObservationError, match="length 31"):
        runner.act([0.0] * 31)


def test_non_finite_value_is_rejected(runner):
    obs = [0.0] * 32
    obs[7] = math.nan
    with pytest.raises(ObservationError, match="index 7"):
        runner.act(obs)


def test_actions_stay_in_the_declared_range(runner):
    for scale in (1.0, 1e3, -1e3):
        action = runner.act([scale] * 32)
        assert all(-1.0 <= a <= 1.0 for a in action)


def test_missing_policy_fails_loudly(tmp_path):
    with pytest.raises(PolicyLoadError, match="no policy file"):
        PolicyRunner(str(tmp_path / "absent.ts.pt"))


def test_policy_without_metadata_fails_loudly(tmp_path):
    lone = tmp_path / "lone.ts.pt"
    shutil.copy(POLICY, lone)
    with pytest.raises(PolicyLoadError, match="no metadata"):
        PolicyRunner(str(lone))


@pytest.mark.parametrize("key, value, message", [
    ("action_dim", 5, "declares 5"),
    ("observation_dim", 34, "34-D observation"),
])
def test_metadata_that_disagrees_with_the_policy_fails(tmp_path, key, value, message):
    shutil.copy(POLICY, tmp_path / "p.ts.pt")
    with open(default_meta_path(POLICY), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta[key] = value
    with open(tmp_path / "p.meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with pytest.raises(PolicyLoadError, match=message):
        PolicyRunner(str(tmp_path / "p.ts.pt"))
