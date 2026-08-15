"""Domain randomisation: what is perturbed, by how much, and when.

The randomiser owns a set of named *parameters*, each with a nominal value and
a multiplicative or additive range. A **level** (``none``, ``low``, ``medium``,
``high``) scales every range by one number, so the ablation varies a single
knob rather than twenty. The exact ranges live in
``src/randomisation/configs/*.json`` and are printed into the results file of
every run, so a result can always be traced back to the distribution it was
trained on.

Three kinds of parameter, deliberately separated because they fail differently:

``dynamics``    mass, friction, object size, gravity. Changing these changes
                what the optimal policy *is*.
``actuation``   gripper gain, weld compliance, action latency. These change how
                a command turns into motion; they are the ones that bite hardest
                on real hardware.
``sensing``     additive noise on the observation. Cheap to randomise and cheap
                to over-do: too much and the policy stops trusting its own eyes.

``shifted`` is not a training level. It is a held-out evaluation distribution
whose parameters sit *outside* the low and medium training ranges, and it is
the stand-in for a real robot in this repository. See ``docs/sim-to-real.md``
for why that stand-in is weaker than a real arm.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict

import numpy as np

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

LEVELS = ("none", "low", "medium", "high", "shifted")


@dataclasses.dataclass
class ParamSpec:
    """One randomised quantity.

    ``low`` and ``high`` are the bounds at scale 1.0. ``mode`` says how the
    bounds are interpreted:

    ``scale``  multiply the nominal model value, bounds are ratios
    ``absolute``  sample the value directly, bounds are in SI units
    """

    low: float
    high: float
    mode: str = "scale"

    def sample(self, rng: np.random.Generator, scale: float, nominal: float) -> float:
        if scale <= 0.0:
            return nominal if self.mode == "scale" else 0.5 * (self.low + self.high)
        if self.mode == "scale":
            lo = 1.0 + (self.low - 1.0) * scale
            hi = 1.0 + (self.high - 1.0) * scale
            return float(nominal * rng.uniform(lo, hi))
        mid = 0.5 * (self.low + self.high)
        half = 0.5 * (self.high - self.low) * scale
        return float(rng.uniform(mid - half, mid + half))


@dataclasses.dataclass
class RandomisationConfig:
    """A named randomisation level: a scale plus the per-parameter ranges."""

    name: str
    scale: float
    params: Dict[str, ParamSpec]

    @staticmethod
    def from_dict(raw: Dict) -> "RandomisationConfig":
        params = {
            key: ParamSpec(**spec) for key, spec in raw.get("params", {}).items()
        }
        return RandomisationConfig(
            name=raw["name"], scale=float(raw["scale"]), params=params
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "scale": self.scale,
            "params": {k: dataclasses.asdict(v) for k, v in self.params.items()},
        }


def load_randomisation(level_or_path: str) -> RandomisationConfig:
    """Load a level by name (``medium``) or by path to a JSON file."""
    if level_or_path is None:
        level_or_path = "none"
    path = level_or_path
    if not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, "{}.json".format(level_or_path))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "unknown randomisation level {!r}; expected one of {} or a path".format(
                level_or_path, ", ".join(LEVELS)
            )
        )
    with open(path, "r", encoding="utf-8") as fh:
        return RandomisationConfig.from_dict(json.load(fh))


@dataclasses.dataclass
class SampledWorld:
    """The concrete parameter values drawn for one episode."""

    object_half_size: float
    object_mass: float
    object_friction: float
    table_friction: float
    gripper_gain: float
    hand_compliance: float
    action_latency: int
    gravity: float
    obs_noise_pos: float
    obs_noise_vel: float
    obs_noise_rot: float
    action_noise: float
    init_xy_jitter: float
    init_yaw_jitter: float

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


NOMINAL = SampledWorld(
    object_half_size=0.022,
    object_mass=0.08,
    object_friction=1.0,
    table_friction=0.8,
    gripper_gain=300.0,
    hand_compliance=0.02,
    action_latency=0,
    gravity=9.81,
    obs_noise_pos=0.0,
    obs_noise_vel=0.0,
    obs_noise_rot=0.0,
    action_noise=0.0,
    init_xy_jitter=0.10,
    init_yaw_jitter=np.pi,
)


def sample_world(cfg: RandomisationConfig, rng: np.random.Generator) -> SampledWorld:
    """Draw one world from ``cfg``.

    Parameters absent from the config keep their nominal value, so a config
    file only has to name what it actually perturbs.
    """
    out = dataclasses.replace(NOMINAL)
    for key, spec in cfg.params.items():
        if not hasattr(out, key):
            raise ValueError("randomisation config names unknown parameter " + key)
        nominal = getattr(NOMINAL, key)
        value = spec.sample(rng, cfg.scale, nominal)
        if key == "action_latency":
            value = int(round(value))
        setattr(out, key, value)
    # The initial-pose jitter is part of the task, not of the randomisation:
    # every level starts the object somewhere different. Only its magnitude is
    # randomisable, and it is clamped so the object always sits on the table.
    out.init_xy_jitter = float(np.clip(out.init_xy_jitter, 0.0, 0.16))
    # Hard cap on object size. The hand has no wrist rotation, so the pads
    # always close along world x. A square box at 45 degrees of yaw presents
    # sqrt(2) times its side to the pads, and the open gap is 78 mm, so any
    # half-size above about 27 mm is ungraspable at the worst yaw no matter
    # what the policy does. The cap sits below that with margin; the missing
    # wrist DoF is recorded in docs/limitations.md.
    out.object_half_size = float(np.clip(out.object_half_size, 0.014, 0.024))
    return out
