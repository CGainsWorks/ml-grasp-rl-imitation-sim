"""Is cross-simulator failure a vertical offset? A falsifiable test.

    C:\\isaac\\venv311\\Scripts\\python scripts/isaac_z_probe.py --policy <exported.pt>

What is already known about the transfer failure, from
`scripts/isaac_cross_sim_ablation.py` and the probe series: MuJoCo policies score
0.05-0.08 in Isaac against 1.000 for the scripted expert, and it is **not** a
control-gain constant (seven action scalings, both directions, none clears its
interval), not grip force, and not friction. What the end-state probe found
instead is a *vertical* signature: the policies are still gripping in 55% of
episodes at the horizon, roughly 10 cm below the hold point.

"Vertical positioning" is a description of the symptom, not a cause, and it has
an obvious falsifiable reading: if the policy's commanded height is
systematically short by a constant, adding that constant back should recover
performance. This sweeps exactly that -- a fixed bias added to the z component of
the action, nothing else touched.

The sweep is the point rather than any single value:

* if success rises and falls around some non-zero bias, the failure is a
  vertical calibration offset and the cause is identified;
* if success stays flat and near zero across the whole range while the peak lift
  moves, the height was a symptom, the policy is failing at contact rather than
  at reaching, and "vertical positioning" should stop being offered as the
  explanation.

Both outcomes are worth having. The second is the one this repository should
expect, because raising peak lift without moving success is exactly what the
earlier lateral-scaling probe already did.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", required=True, help="TorchScript actor")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--randomisation", default="none")
parser.add_argument("--biases", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, -0.05])
parser.add_argument("--output", default="experiments/results/isaac_z_probe.json")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import GraspTask, GraspTaskCfg  # noqa: E402
from src.utils.stats import wilson_interval  # noqa: E402

cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)
policy = torch.jit.load(args.policy, map_location=env.device)
policy.eval()


def run(bias: float) -> dict:
    successes, trials, peaks, held = 0, 0, [], 0
    for _ in range(args.episodes):
        obs_dict, _ = env.reset()
        peak = torch.zeros(args.num_envs, device=env.device)
        for _ in range(int(env.max_episode_length) - 2):
            with torch.no_grad():
                action = policy(obs_dict["policy"])
            # The whole intervention: a constant added to the vertical command,
            # clipped back into the action range the environment accepts so the
            # bias cannot smuggle in extra authority the policy never had.
            action = action.clone()
            action[:, 2] = torch.clamp(action[:, 2] + bias, -1.0, 1.0)
            obs_dict, _, _, _, _ = env.step(action)
            peak = torch.maximum(peak, env._object_pos()[:, 2] - env.object_rest_z)
        successes += int(env.success().sum())
        held += int((env._grasped() > 0.5).sum())
        trials += args.num_envs
        peaks.append(float(peak.mean()))
    interval = wilson_interval(successes, trials)
    row = {"bias": bias, "success_rate": interval.point,
           "wilson_low": interval.low, "wilson_high": interval.high,
           "mean_peak_lift": float(np.mean(peaks)),
           "still_holding_at_horizon": held / trials}
    print("bias {:+.2f} m  success {:.3f} [{:.3f}, {:.3f}]  peak lift {:.3f} m  "
          "still holding {:.2f}".format(
              bias, interval.point, interval.low, interval.high,
              row["mean_peak_lift"], row["still_holding_at_horizon"]), flush=True)
    return row


rows = [run(b) for b in args.biases]
best = max(rows, key=lambda r: r["success_rate"])
print("\nbest bias {:+.2f} m at {:.3f}; zero-bias baseline {:.3f}".format(
    best["bias"], best["success_rate"], rows[0]["success_rate"]))
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as fh:
    json.dump({"policy": args.policy, "randomisation": args.randomisation,
               "episodes_per_bias": args.episodes * args.num_envs,
               "note": "a constant added to the vertical action, clipped to the "
                       "action range. Rising-then-falling success identifies a "
                       "calibration offset; flat-and-zero success with moving "
                       "peak lift means height was a symptom",
               "rows": rows, "best": best}, fh, indent=2)
env.close()
_app.close()
