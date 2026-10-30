"""Load an exported TorchScript grasp policy and turn observations into actions.

Nothing in this module imports ROS. The node in ``policy_server_node.py`` is a
thin shell around :class:`PolicyRunner`, so everything that can go wrong with
the policy itself (a missing file, a width mismatch, a non-finite input) is
testable with plain pytest and fails the same way inside and outside ROS.

The artefact is what ``src/export_policy.py`` writes: ``<name>.ts.pt`` holds
the actor's deterministic path with the observation normaliser inside it and
``<name>.meta.json`` states the observation and action layouts. The runner
refuses to start if the two disagree, because a policy fed the wrong
observation width does not crash: it fails inside the normaliser, or worse, it
runs on a shifted vector and emits plausible nonsense.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence

import torch


class PolicyLoadError(RuntimeError):
    """The policy artefact is missing or inconsistent with its metadata."""


class ObservationError(ValueError):
    """An observation the policy cannot be run on."""


def default_meta_path(policy_path: str) -> str:
    """``foo.ts.pt`` is described by ``foo.meta.json``, as the exporter writes them."""
    if policy_path.endswith(".ts.pt"):
        return policy_path[: -len(".ts.pt")] + ".meta.json"
    return os.path.splitext(policy_path)[0] + ".meta.json"


class PolicyRunner:
    """A TorchScript policy with its declared observation and action widths."""

    def __init__(self, policy_path: str, meta_path: Optional[str] = None,
                 num_threads: int = 1) -> None:
        if not policy_path or not os.path.isfile(policy_path):
            raise PolicyLoadError("no policy file at '{}'".format(policy_path))
        meta_path = meta_path or default_meta_path(policy_path)
        if not os.path.isfile(meta_path):
            raise PolicyLoadError(
                "no metadata at '{}'; the exporter writes it next to the "
                "policy and it is what states the observation layout".format(meta_path))
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta: Dict = json.load(fh)
        try:
            self.observation_dim = int(self.meta["observation_dim"])
            self.action_dim = int(self.meta["action_dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyLoadError(
                "metadata '{}' has no usable observation_dim/action_dim".format(
                    meta_path)) from exc
        low, high = self.meta.get("action_range", [-1.0, 1.0])
        self.action_low, self.action_high = float(low), float(high)

        # One thread: the network is two layers of a few hundred units and a
        # control loop wants a steady latency more than a parallel one.
        torch.set_num_threads(max(1, int(num_threads)))
        self.module = torch.jit.load(policy_path, map_location="cpu")
        self.module.eval()
        self.policy_path = policy_path
        self.meta_path = meta_path

        # Prove the widths at load time rather than on the first observation.
        try:
            probe = self._forward([0.0] * self.observation_dim)
        except RuntimeError as exc:
            raise PolicyLoadError(
                "policy does not accept the {}-D observation its metadata "
                "declares".format(self.observation_dim)) from exc
        if len(probe) != self.action_dim:
            raise PolicyLoadError(
                "policy returns {} values but the metadata declares {}".format(
                    len(probe), self.action_dim))

    def _forward(self, observation: Sequence[float]) -> List[float]:
        with torch.no_grad():
            x = torch.tensor([list(observation)], dtype=torch.float32)
            y = self.module(x)
        return [float(v) for v in y.reshape(-1)]

    def validate(self, observation: Sequence[float]) -> None:
        if len(observation) != self.observation_dim:
            raise ObservationError(
                "rejected observation of length {}: the policy expects {}".format(
                    len(observation), self.observation_dim))
        for i, value in enumerate(observation):
            if not math.isfinite(float(value)):
                raise ObservationError(
                    "rejected observation with a non-finite value at index {}".format(i))

    def act(self, observation: Sequence[float]) -> List[float]:
        """Deterministic action for one observation, clipped to the declared range."""
        self.validate(observation)
        action = self._forward(observation)
        return [min(self.action_high, max(self.action_low, a)) for a in action]

    def layout(self, key: str) -> List[Dict]:
        """``observation_layout`` or ``action_layout`` from the metadata."""
        return list(self.meta.get(key, []))
