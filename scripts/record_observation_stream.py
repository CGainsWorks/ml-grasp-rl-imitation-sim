"""Record one simulated episode as an observation stream for the ROS 2 node.

    python scripts/record_observation_stream.py
    python scripts/record_observation_stream.py --policy experiments/policies/bcrl_medium_s0.ts.pt \
        --randomisation none --seed 900000 \
        --output ros2/grasp_policy_server/test/data/replay_bcrl_medium_s0.json

The exported TorchScript policy drives the MuJoCo environment in closed loop
for one episode. Every observation is stored as the float32 values a
``std_msgs/Float32MultiArray`` would carry, next to the action the policy
returned for exactly those values. The launch test in
``ros2/grasp_policy_server/test`` replays the observations through the node and
requires the node's actions to match the recorded ones.

Everything in the file is simulated: MuJoCo, the scripted world and the
exported policy. No robot produced any of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import mujoco
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from envs.mujoco.grasp_env import make_env  # noqa: E402

DEFAULT_POLICY = "experiments/policies/bcrl_medium_s0.ts.pt"
DEFAULT_OUTPUT = "ros2/grasp_policy_server/test/data/replay_bcrl_medium_s0.json"


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--randomisation", default="none")
    parser.add_argument("--seed", type=int, default=900_000,
                        help="episode seed; 900000 is the first of the final-evaluation block")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    policy_path = os.path.join(REPO, args.policy)
    module = torch.jit.load(policy_path, map_location="cpu").eval()
    env = make_env(args.randomisation, seed=1234, max_steps=args.max_steps)

    observations, actions = [], []
    obs, _ = env.reset(seed=args.seed)
    done, info = False, {}
    while not done:
        x = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            a = module(torch.as_tensor(x[None, :])).numpy()[0].astype(np.float32)
        observations.append([float(v) for v in x])
        actions.append([float(v) for v in a])
        obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated

    blob = {
        "description": "One simulated MuJoCo episode driven in closed loop by the "
                       "exported policy. Observations are float32 as sent over "
                       "std_msgs/Float32MultiArray; each action is the policy's "
                       "output for the observation at the same index.",
        "generator": "scripts/record_observation_stream.py",
        "simulated": True,
        "simulator": "MuJoCo " + mujoco.__version__,
        "torch": torch.__version__.split("+")[0],
        "policy": args.policy.replace(os.sep, "/"),
        "policy_sha256": sha256(policy_path),
        "randomisation": args.randomisation,
        "episode_seed": args.seed,
        "control_period_seconds": 0.04,
        "steps": len(observations),
        "episode_success": bool(info.get("is_success", False)),
        "observation_dim": len(observations[0]),
        "action_dim": len(actions[0]),
        "observations": observations,
        "actions": actions,
    }
    out = os.path.join(REPO, args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # One step per line keeps the file readable and its diffs small.
    head = {k: v for k, v in blob.items() if k not in ("observations", "actions")}
    lines = json.dumps(head, indent=2)[:-2].splitlines()
    lines[-1] += ","
    for key in ("observations", "actions"):
        rows = ",\n".join("    " + json.dumps(r) for r in blob[key])
        lines.append('  "{}": [\n{}\n  ]{}'.format(key, rows, "," if key == "observations" else ""))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n}\n")
    with open(out, "r", encoding="utf-8") as fh:
        json.load(fh)  # the hand-built layout must still parse
    print("wrote {}: {} steps, episode success {}".format(
        args.output, blob["steps"], blob["episode_success"]))


if __name__ == "__main__":
    main()
