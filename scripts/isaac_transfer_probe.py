"""Why don't MuJoCo policies transfer to Isaac? One variable at a time.

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_transfer_probe.py

`experiments/results/cross_sim_ablation.json` establishes *that* transfer fails:
twenty policies, five seeds at each of four randomisation levels, 0.05 to 0.08
in Isaac against 1.000 for the scripted expert in the same environment. The
observation and action layouts match, so what fails is the behaviour. This asks
which difference in behaviour, by changing one thing at a time about how the
policy is applied rather than about the policy.

Each arm is the same exported MuJoCo policy, run in Isaac with one modification:

``baseline``        as-is, the number already in the tables
``scale-0.5``       every action halved
``scale-0.25``      every action quartered
``scale-1.5``       every action half again larger
``scale-2.0``       every action doubled
``clip-xy``         lateral commands halved, vertical left alone
``hold-gripper``    the gripper channel forced closed once a grasp registers

The first three test the simplest hypothesis available: the two simulators
agree on what a command *means* but not on what it *does*, so a policy tuned to
MuJoCo's compliant mocap weld overshoots against Isaac's stiffer IK controller
and knocks the box away before it can close. If halving the action rescues the
policy, the mismatch is control gain and the fix is a calibration constant. If
it does nothing, the mismatch is in contact or dynamics and no rescaling will
help.

``clip-xy`` separates lateral overshoot from vertical: sweeping the box off the
table is a lateral failure, descending too fast is a vertical one.
``hold-gripper`` tests whether the policy is losing grasps it had, which the
per-step contact readings in the cross-sim runs hinted at.

Evaluation only -- nothing is trained -- so this is minutes of GPU rather than
hours.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", default="experiments/policies/bcrl_high_s0.ts.pt")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--randomisation", default="none")
parser.add_argument("--arms", nargs="+",
                    default=["baseline", "scale-0.5", "scale-0.25",
                             "scale-1.5", "scale-2.0", "clip-xy", "hold-gripper"])
parser.add_argument("--output", default="experiments/results/transfer_probe.json")
args = parser.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

# Isaac Sim prompts for the EULA on first launch and reads the answer from
# stdin. A backgrounded run has no stdin, so it hangs on "Do you accept the
# EULA?" and then dies with "EOF when reading a line" -- which looks like a
# crash rather than a prompt. Accepting here keeps unattended runs unattended.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher({"headless": True}).app

from envs.isaac.grasp_task import GraspTask, GraspTaskCfg  # noqa: E402
from src.utils.stats import wilson_interval  # noqa: E402

cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)

# The exported policies are CPU TorchScript; the observations are on the card.
# Move the policy rather than the observations: it is 128x128, the observation
# batch is not, and the cross-sim numbers this probe extends were produced the
# same way.
actor = torch.jit.load(args.policy, map_location=env.device)
actor.eval()


def transform(action: torch.Tensor, arm: str, grasped: torch.Tensor) -> torch.Tensor:
    """Apply one arm's modification to a batch of actions."""
    out = action.clone()
    if arm == "scale-0.5":
        out[:, :3] *= 0.5
    elif arm == "scale-0.25":
        out[:, :3] *= 0.25
    elif arm == "scale-1.5":
        out[:, :3] *= 1.5
    elif arm == "scale-2.0":
        out[:, :3] *= 2.0
    elif arm == "clip-xy":
        out[:, :2] *= 0.5
    elif arm == "hold-gripper":
        out[:, 3] = torch.where(grasped > 0.5, torch.ones_like(out[:, 3]), out[:, 3])
    return out


def run(arm: str) -> dict:
    successes, trials, peaks = 0, 0, []
    for _ in range(args.episodes):
        obs_dict, _ = env.reset()
        peak = torch.zeros(env.num_envs, device=env.device)
        for _ in range(int(env.max_episode_length) - 2):
            with torch.no_grad():
                action = actor(obs_dict["policy"])
            action = transform(action, arm, env._grasped().float())
            obs_dict, _, _, _, _ = env.step(action)
            peak = torch.maximum(peak, env._object_pos()[:, 2] - env.object_rest_z)
        successes += int(env.success().sum())
        trials += env.num_envs
        peaks.append(float(peak.mean()))
    interval = wilson_interval(successes, trials)
    print("{:<16s} success {:.3f}  95% Wilson [{:.3f}, {:.3f}]  peak lift {:.3f} m".format(
        arm, interval.point, interval.low, interval.high, float(np.mean(peaks))), flush=True)
    return {"arm": arm, "successes": successes, "episodes": trials,
            "success_rate": interval.point, "wilson": interval.as_dict(),
            "mean_peak_lift": float(np.mean(peaks))}


arms = args.arms
results = [run(arm) for arm in arms]

blob = {
    "question": "which difference in applying the policy breaks cross-simulator transfer?",
    "policy": args.policy,
    "simulator": "Isaac Sim 5.1.0 / Isaac Lab 2.3.2",
    "randomisation": args.randomisation,
    "num_envs": args.num_envs,
    "episodes_per_arm": args.episodes,
    "arms": results,
    "note": "One modification per arm, applied to the action only. The policy is "
            "unchanged throughout; this asks whether the transfer failure is a "
            "control-gain mismatch that a constant could fix.",
}
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as fh:
    json.dump(blob, fh, indent=2)
print("wrote " + args.output)

env.close()
_app.close()
