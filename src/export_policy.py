"""Export a trained policy to a self-contained artefact for deployment.

    python src/export_policy.py --run experiments/runs/bcrl_medium_s0 \
        --output experiments/policies/bcrl_medium_s0

Writes three things next to each other:

    <name>.ts.pt      TorchScript module: observation (32,) -> action (4,)
    <name>.meta.json  observation layout, action layout, training provenance
    <name>.check.npz  reference input/output pairs

The TorchScript module has the observation normaliser **inside** it. A policy
exported without its normalisation statistics behaves exactly like a badly
trained one, and that is a miserable bug to diagnose on hardware.

The reference pairs exist so that whatever runs the policy on the other side —
a C++ node, a different Python version, a different machine — can prove it
agrees with what was trained here before anything moves. ``--verify`` does that
check locally as part of the export.

What this artefact does *not* include, and what a real deployment must add in
front of it, is in ``docs/limitations.md``: there is no arm in this simulation,
so the Cartesian delta this policy emits needs a reachability and joint-limit
check before it reaches a servo loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import ACT_DIM, OBS_DIM  # noqa: E402
from src.evaluate import load_actor  # noqa: E402

OBS_LAYOUT = [
    ("grip_pos", 0, 3, "grip site position, world frame, metres"),
    ("grip_vel", 3, 6, "grip site linear velocity, m/s"),
    ("gripper_width", 6, 7, "distance between the finger pads, metres"),
    ("gripper_width_rate", 7, 8, "rate of change of that width, m/s"),
    ("object_pos", 8, 11, "object position, world frame, metres"),
    ("object_rel_pos", 11, 14, "object position minus grip position"),
    ("object_rot6", 14, 20, "first two columns of the object rotation matrix"),
    ("object_linvel", 20, 23, "object linear velocity, m/s"),
    ("object_angvel", 23, 26, "object angular velocity, rad/s"),
    ("goal_pos", 26, 29, "hold point, world frame, metres"),
    ("goal_rel_pos", 29, 32, "hold point minus object position"),
]

ACT_LAYOUT = [
    ("delta_xyz", 0, 3, "Cartesian displacement command, multiplied by 0.02 m"),
    ("gripper", 3, 4, "-1 fully open, +1 fully closed"),
]


class DeployablePolicy(torch.nn.Module):
    """Actor reduced to its deterministic path, normalisation included."""

    def __init__(self, actor) -> None:
        super().__init__()
        self.norm_mean = torch.nn.Parameter(actor.norm.mean.clone(), requires_grad=False)
        self.norm_var = torch.nn.Parameter(actor.norm.var.clone(), requires_grad=False)
        self.clip = float(actor.norm.clip)
        self.trunk = actor.trunk
        self.mu_head = actor.mu_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normed = (obs - self.norm_mean) / torch.sqrt(self.norm_var + 1e-8)
        normed = torch.clamp(normed, -self.clip, self.clip)
        return torch.tanh(self.mu_head(self.trunk(normed)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory containing the checkpoint")
    parser.add_argument("--checkpoint", default="policy.pt")
    parser.add_argument("--output", default=None,
                        help="output basename; defaults to experiments/policies/<run name>")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()

    checkpoint = os.path.join(args.run, args.checkpoint)
    actor = load_actor(checkpoint)
    module = DeployablePolicy(actor).eval()

    base = args.output or os.path.join("experiments", "policies", os.path.basename(args.run))
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)

    scripted = torch.jit.script(module)
    ts_path = base + ".ts.pt"
    scripted.save(ts_path)

    rng = np.random.default_rng(0)
    sample_obs = rng.normal(size=(args.samples, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        reference = module(torch.as_tensor(sample_obs)).numpy()
    np.savez(base + ".check.npz", observations=sample_obs, actions=reference)

    config = {}
    config_path = os.path.join(args.run, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    result = {}
    result_path = os.path.join(args.run, "result.json")
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as fh:
            result = json.load(fh)

    meta = {
        "source_run": args.run,
        "checkpoint": args.checkpoint,
        "observation_dim": OBS_DIM,
        "action_dim": ACT_DIM,
        "observation_layout": [
            {"name": n, "start": a, "stop": b, "units": u} for n, a, b, u in OBS_LAYOUT
        ],
        "action_layout": [
            {"name": n, "start": a, "stop": b, "meaning": u} for n, a, b, u in ACT_LAYOUT
        ],
        "action_range": [-1.0, 1.0],
        "control_period_seconds": 0.04,
        "position_step_metres": 0.02,
        "trained_with_randomisation": config.get("randomisation"),
        "training_steps": config.get("steps"),
        "seed": config.get("seed"),
        "final_success_rate_training_distribution": result.get("final_success_rate"),
        "deployment_note": (
            "This policy was trained on a free-floating hand with no arm. A real "
            "deployment must put a reachability and joint-limit check between this "
            "output and any servo loop. See docs/limitations.md."
        ),
    }
    with open(base + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if args.verify:
        loaded = torch.jit.load(ts_path)
        with torch.no_grad():
            replayed = loaded(torch.as_tensor(sample_obs)).numpy()
        max_error = float(np.abs(replayed - reference).max())
        if max_error > 1e-6:
            raise SystemExit("TorchScript round trip differs by {:.2e}".format(max_error))
        print("round trip verified, max error {:.2e}".format(max_error))

    print("wrote {}.ts.pt ({:.0f} kB), {}.meta.json, {}.check.npz".format(
        base, os.path.getsize(ts_path) / 1e3, base, base))


if __name__ == "__main__":
    main()
