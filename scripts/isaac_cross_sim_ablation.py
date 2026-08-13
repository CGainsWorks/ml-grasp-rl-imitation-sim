"""Does domain randomisation help *across simulators*, not just across worlds?

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_cross_sim_ablation.py

The MuJoCo ablation shows wider randomisation transferring better to the
``shifted`` worlds. But `shifted` is a distribution this repository designed, in
the engine the policies were trained in, so it can only ever be suggestive: it
shares MuJoCo's contact model, its actuator model and its idea of a gripper.

Isaac is not that. Different solver, different contact model, a Franka on
differential IK instead of a floating hand on a weld. Running the four
randomisation levels there asks the question the proxy cannot: does training
with wider randomisation in one simulator produce a policy that survives a
*different simulator*?

All five seeds of every level are evaluated, exported to TorchScript and run
without any adaptation, and the headline per level is the mean across seeds with
a 95% t interval -- the same standard the MuJoCo tables use. A single policy per
level would be an anecdote, and this repository says so about everyone else's
numbers.

The scripted expert is the reference line: it is the same state machine that
drives the MuJoCo environment, and it scores 1.000 here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--levels", nargs="+", default=["none", "low", "medium", "high"])
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--randomisation", default="none",
                    help="the Isaac-side distribution to evaluate on")
parser.add_argument("--output", default="experiments/results/cross_sim_ablation.json")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import GraspTask, GraspTaskCfg  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.utils.stats import summarise_seeds, wilson_interval  # noqa: E402

cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)


def run(actor, label):
    successes = trials = 0
    peaks = []
    for _ in range(args.episodes):
        obs_dict, _ = env.reset()
        experts = [ScriptedExpert() for _ in range(args.num_envs)]
        peak = torch.zeros(env.num_envs, device=env.device)
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
        trials += env.num_envs
        peaks.append(float(peak.mean()))
    interval = wilson_interval(successes, trials)
    print("{:<34s} success {:.3f}  95% Wilson [{:.3f}, {:.3f}]  peak lift {:.3f} m".format(
        label, interval.point, interval.low, interval.high, float(np.mean(peaks))), flush=True)
    return {
        "label": label, "successes": successes, "episodes": trials,
        "success_rate": interval.point,
        "wilson_low": interval.low, "wilson_high": interval.high,
        "mean_peak_lift": float(np.mean(peaks)),
    }


rows = [run("expert", "scripted expert (reference)")]
levels_summary = {}
for level in args.levels:
    per_seed = []
    for seed in args.seeds:
        path = os.path.join(
            "experiments", "policies", "bcrl_{}_s{}.ts.pt".format(level, seed)
        )
        if not os.path.exists(path):
            print("skip {} seed {}: not exported".format(level, seed), flush=True)
            continue
        policy = torch.jit.load(path, map_location=env.device)
        policy.eval()
        row = run(policy, "  '{}' DR, seed {}".format(level, seed))
        row["train_randomisation"] = level
        row["seed"] = seed
        rows.append(row)
        per_seed.append(row)
    if per_seed:
        summary = summarise_seeds(
            [r["successes"] for r in per_seed], [r["episodes"] for r in per_seed]
        )
        levels_summary[level] = summary
        across = summary["across_seeds"]
        print("{:<34s} {:.3f}  95% t across {} seeds [{:.3f}, {:.3f}]".format(
            "'{}' DR, across seeds".format(level), across["point"], summary["n_seeds"],
            across["low"], across["high"]), flush=True)

blob = {
    "simulator": "Isaac Sim 5.1.0 / Isaac Lab 2.3.2",
    "isaac_randomisation": args.randomisation,
    "episodes_per_condition": args.episodes * args.num_envs,
    "rows": rows,
    "across_seeds": levels_summary,
    "seeds": args.seeds,
    "note": "Policies trained in MuJoCo at each randomisation level, run in Isaac "
            "without adaptation. Unlike the 'shifted' proxy, this distribution "
            "was not designed by this repository and does not share MuJoCo's "
            "contact or actuator model.",
}
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as fh:
    json.dump(blob, fh, indent=2)
print("wrote " + args.output)

env.close()
_app.close()
