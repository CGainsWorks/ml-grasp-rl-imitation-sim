"""Committed run records name files by repository-relative POSIX paths only.

Every number in the README traces to a JSON or CSV file under ``experiments/``,
and those files record which run, checkpoint and dataset produced them. Runs
made on Windows used to record ``experiments/runs\\sac_none_s0``; worse, the same
mechanism is how an absolute local path ends up in a public file. This walks
every record and fails on either.
"""

from __future__ import annotations

import os
import re

from src.utils.paths import posix_paths

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = ["experiments", "ros2"]
SUFFIXES = (".json", ".csv")

WINDOWS_SEPARATOR = re.compile(r'"[A-Za-z0-9_.\-/]+(?:\\\\[A-Za-z0-9_.\-/]+)+"')
ABSOLUTE = re.compile(r'(?:\b[A-Za-z]:[\\/]{1,2}[A-Za-z]|/home/|/Users/|/mnt/[a-z]/|/tmp/)')


def _records():
    for root in ROOTS:
        for dirpath, _, names in os.walk(os.path.join(REPO, root)):
            for name in names:
                if name.endswith(SUFFIXES):
                    yield os.path.join(dirpath, name)


def test_there_are_records_to_check():
    assert sum(1 for _ in _records()) > 1000


def test_no_windows_separators_in_records():
    bad = []
    for path in _records():
        with open(path, "r", encoding="utf-8") as fh:
            m = WINDOWS_SEPARATOR.search(fh.read())
        if m:
            bad.append("{}: {}".format(os.path.relpath(path, REPO), m.group(0)))
    assert not bad, "\n".join(bad[:20])


def test_no_absolute_paths_in_records():
    bad = []
    for path in _records():
        with open(path, "r", encoding="utf-8") as fh:
            m = ABSOLUTE.search(fh.read())
        if m:
            bad.append("{}: {}".format(os.path.relpath(path, REPO), m.group(0)))
    assert not bad, "\n".join(bad[:20])


def test_writer_converts_windows_separators():
    blob = {"run": "experiments/runs\\sac_none_s0",
            "runs": ["demonstrations\\expert_low.npz"],
            "note": "a sentence with a \\ in it is left alone"}
    out = posix_paths(blob)
    assert out["run"] == "experiments/runs/sac_none_s0"
    assert out["runs"] == ["demonstrations/expert_low.npz"]
    assert out["note"] == blob["note"]
