"""Where the Isaac training time actually goes, on this card.

    C:\\isaac\\venv311\\Scripts\\python scripts/isaac_profile.py --num-envs 256 --device cuda

A budget-matched Isaac grid is the last thing this repository is missing, and at
the settings the existing runs use it is GPU-days. Before buying that with
patience it is worth knowing which part is slow, because the two candidates want
opposite fixes:

* if **simulation** dominates, more parallel environments is the answer, and the
  only real limit is the card's memory;
* if the **learner** dominates, more environments makes it worse -- each step
  costs more and produces transitions the optimiser cannot keep up with -- and
  the fix is to move the networks onto the GPU beside the simulator.

So this times the two separately rather than guessing. It reports transitions
per second, which is the number that decides how long a budget-matched grid
takes, and peak GPU memory, which is what decides whether the next step up in
environment count fits on an 8 GB card at all.

The learner is timed on exactly the batch size and update ratio the training
script uses, so the numbers compose: a run's wall-clock is
``steps x (sim + learner)`` and nothing else of consequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--device", default="cuda")
parser.add_argument("--task", default="place", choices=("lift", "place"))
parser.add_argument("--randomisation", default="none")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--output", default="experiments/results/isaac_profile.json")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import ACT_DIM, OBS_DIM, GraspTask, GraspTaskCfg  # noqa: E402
from src.policies.sac import SAC, SACConfig  # noqa: E402

cfg = GraspTaskCfg()
cfg.task = args.task
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)
obs_dict, _ = env.reset()
obs = obs_dict["policy"].cpu().numpy()

sac_cfg = SACConfig(hidden=(args.hidden, args.hidden), batch_size=args.batch_size)
agent = SAC(OBS_DIM, ACT_DIM, sac_cfg, seed=0, device=args.device)
rng = np.random.default_rng(0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- simulation
# Random actions, so the actor is not in this measurement at all.
for _ in range(10):  # warm up: the first steps pay for lazy CUDA initialisation
    env.step(torch.as_tensor(
        rng.uniform(-1, 1, (env.num_envs, ACT_DIM)).astype(np.float32),
        device=env.device))
sync()
t0 = time.time()
for _ in range(args.steps):
    out = env.step(torch.as_tensor(
        rng.uniform(-1, 1, (env.num_envs, ACT_DIM)).astype(np.float32),
        device=env.device))
    out[0]["policy"].cpu().numpy()   # the transfer the training loop really does
sync()
sim_per_step = (time.time() - t0) / args.steps

# ---------------------------------------------------------------- learner
for _ in range(4):
    agent.buffer.add_batch(
        obs, rng.uniform(-1, 1, (env.num_envs, ACT_DIM)).astype(np.float32),
        rng.normal(size=env.num_envs).astype(np.float32), obs,
        np.zeros(env.num_envs, dtype=np.float32))
for _ in range(10):
    agent.update(1)
sync()
t0 = time.time()
for i in range(args.steps):
    agent.update(i + 1)
sync()
update_per_call = (time.time() - t0) / args.steps

# The training loop performs `update_every` updates every `update_every` steps,
# i.e. one update per environment step on average.
learner_per_step = update_per_call
total_per_step = sim_per_step + learner_per_step
peak_gb = (torch.cuda.max_memory_allocated() / 1e9
           if torch.cuda.is_available() else 0.0)

row = {
    "num_envs": args.num_envs, "device": args.device, "task": args.task,
    "sim_ms_per_step": round(sim_per_step * 1e3, 2),
    "learner_ms_per_update": round(update_per_call * 1e3, 2),
    "total_ms_per_step": round(total_per_step * 1e3, 2),
    "transitions_per_second": round(env.num_envs / total_per_step, 1),
    "sim_share": round(sim_per_step / total_per_step, 3),
    "torch_peak_gb": round(peak_gb, 2),
}
print(json.dumps(row, indent=2), flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
rows = []
if os.path.exists(args.output):
    with open(args.output, "r", encoding="utf-8") as fh:
        rows = json.load(fh).get("rows", [])
rows = [r for r in rows
        if not (r["num_envs"] == args.num_envs and r["device"] == args.device
                and r["task"] == args.task)]
rows.append(row)
with open(args.output, "w", encoding="utf-8") as fh:
    json.dump({"note": "wall-clock of a run is steps x total_ms_per_step; "
                       "transitions_per_second is what decides how long a "
                       "budget-matched grid takes",
               "rows": sorted(rows, key=lambda r: (r["device"], r["num_envs"]))},
              fh, indent=2)
env.close()
_app.close()
