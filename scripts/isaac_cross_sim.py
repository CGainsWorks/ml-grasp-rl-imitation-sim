"""Run a MuJoCo-trained policy in Isaac, and measure what transfers.

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_cross_sim.py \\
        --policy experiments/policies/bcrl_high_s0.ts.pt --episodes 4

The two environments present the same 32-dimensional observation and take the
same 4-dimensional action, so a policy exported from the MuJoCo runs can be
dropped into Isaac without adaptation. What differs is everything underneath:
a free-floating hand dragged by a mocap weld against a Franka driven by
differential inverse kinematics, with its own tracking lag, joint limits and
finger geometry.

This is the honest version of a sim-to-sim experiment. It is not a substitute
for hardware, but unlike the ``shifted`` proxy it is a genuinely different
simulator, different contact solver and different embodiment.

The scripted expert is run alongside as a reference, because a low number for
the policy only means something next to what an ideal controller achieves in
the same environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", default="experiments/policies/bcrl_high_s0.ts.pt")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--randomisation", default="none")
parser.add_argument("--output", default="experiments/results/cross_sim.json")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import GraspTask, GraspTaskCfg  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.utils.stats import wilson_interval  # noqa: E402

cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)

policy = torch.jit.load(args.policy, map_location=env.device)
policy.eval()


def run(actor, label):
    """Returns (successes, episodes, mean peak lift)."""
    successes, trials, peaks = 0, 0, []
    for _ in range(args.episodes):
        obs_dict, _ = env.reset()
        experts = [ScriptedExpert() for _ in range(args.num_envs)]
        peak = torch.zeros(args.num_envs, device=env.device)
        # Two steps short of the horizon: the env auto-resets on time-out.
        for _ in range(int(env.max_episode_length) - 2):
            obs = obs_dict["policy"]
            if actor == "expert":
                obs_np = obs.cpu().numpy()
                action = torch.as_tensor(
                    np.stack([e.act(obs_np[i]) for i, e in enumerate(experts)]),
                    device=env.device,
                )
            else:
                with torch.no_grad():
                    action = actor(obs)
            obs_dict, _, _, _, _ = env.step(action)
            peak = torch.maximum(peak, env._object_pos()[:, 2] - env.object_rest_z)
        successes += int(env.success().sum())
        trials += args.num_envs
        peaks.append(float(peak.mean()))
    interval = wilson_interval(successes, trials)
    print("{:<28s} success {:.3f}  95% Wilson [{:.3f}, {:.3f}]  mean peak lift {:.3f} m".format(
        label, interval.point, interval.low, interval.high, float(np.mean(peaks))), flush=True)
    return {
        "label": label, "successes": successes, "episodes": trials,
        "success_rate": interval.point,
        "wilson_low": interval.low, "wilson_high": interval.high,
        "mean_peak_lift": float(np.mean(peaks)),
    }


rows = [
    run("expert", "scripted expert (reference)"),
    run(policy, "MuJoCo policy, run in Isaac"),
]

blob = {
    "policy": args.policy,
    "num_envs": args.num_envs,
    "episodes_per_condition": args.episodes * args.num_envs,
    "randomisation": args.randomisation,
    "simulator": "Isaac Sim 5.1.0 / Isaac Lab 2.3.2",
    "rows": rows,
    "note": "Same observation and action layout, different embodiment and "
            "contact solver. The expert row is the reference: it is the same "
            "state machine that drives the MuJoCo environment.",
}
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as fh:
    json.dump(blob, fh, indent=2)
print("wrote " + args.output)

env.close()
_app.close()
