"""Record scripted-expert demonstrations inside Isaac.

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_record_demos.py \\
        --episodes 200 --randomisation low --output demonstrations/isaac_expert_low.npz

Writes the same npz layout as ``src/record_demos.py``, so the same replay-buffer
seeding and behaviour-cloning code reads it without knowing which simulator it
came from.

Recording here rather than reusing the MuJoCo demonstrations is deliberate. The
two environments share an observation and action layout but not an embodiment,
so a MuJoCo demonstration is a slightly wrong label for an Isaac state: the arm
lags its setpoint differently, and the expert's corrections differ with it.
Cloning across that gap is a separate experiment; ``scripts/isaac_cross_sim.py``
is the one that measures it.

Episodes run to two steps short of the horizon, because DirectRLEnv auto-resets
the instant the time-out fires and the last transition would otherwise straddle
two episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--randomisation", default="low")
parser.add_argument("--expert-noise", type=float, default=0.02)
parser.add_argument("--keep-failures", action="store_true")
parser.add_argument("--output", default="demonstrations/isaac_expert_low.npz")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import GraspTask, GraspTaskCfg  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402

cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)

rng = np.random.default_rng(7)
obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
episode_starts, episode_lengths, episode_success = [], [], []
attempted = kept = 0
t0 = time.time()
horizon = int(env.max_episode_length) - 2

while kept < args.episodes:
    obs_dict, _ = env.reset()
    experts = [ScriptedExpert(noise=args.expert_noise, rng=rng) for _ in range(args.num_envs)]
    # Per-environment rollout buffers; each environment is one episode.
    ep_obs = [[] for _ in range(args.num_envs)]
    ep_act = [[] for _ in range(args.num_envs)]
    ep_rew = [[] for _ in range(args.num_envs)]
    ep_next = [[] for _ in range(args.num_envs)]
    ep_done = [[] for _ in range(args.num_envs)]

    for _ in range(horizon):
        obs_np = obs_dict["policy"].cpu().numpy()
        actions = np.stack([e.act(obs_np[i]) for i, e in enumerate(experts)])
        obs_dict, reward, terminated, _, _ = env.step(
            torch.as_tensor(actions, device=env.device)
        )
        next_np = obs_dict["policy"].cpu().numpy()
        rew_np = reward.cpu().numpy()
        term_np = terminated.cpu().numpy()
        for i in range(args.num_envs):
            ep_obs[i].append(obs_np[i])
            ep_act[i].append(actions[i])
            ep_rew[i].append(rew_np[i])
            ep_next[i].append(next_np[i])
            ep_done[i].append(float(term_np[i]))

    success = env.success().cpu().numpy()
    for i in range(args.num_envs):
        attempted += 1
        if kept >= args.episodes:
            break
        if not success[i] and not args.keep_failures:
            continue
        episode_starts.append(len(obs_list))
        obs_list.extend(ep_obs[i])
        act_list.extend(ep_act[i])
        rew_list.extend(ep_rew[i])
        next_list.extend(ep_next[i])
        done_list.extend(ep_done[i])
        episode_lengths.append(len(ep_obs[i]))
        episode_success.append(int(success[i]))
        kept += 1
    print("  kept {}/{} after {} attempts".format(kept, args.episodes, attempted), flush=True)

data = {
    "observations": np.asarray(obs_list, dtype=np.float32),
    "actions": np.asarray(act_list, dtype=np.float32),
    "rewards": np.asarray(rew_list, dtype=np.float32),
    "next_observations": np.asarray(next_list, dtype=np.float32),
    "dones": np.asarray(done_list, dtype=np.float32),
    "episode_starts": np.asarray(episode_starts, dtype=np.int64),
    "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
    "episode_success": np.asarray(episode_success, dtype=np.int64),
    "meta": json.dumps({
        "simulator": "isaac-sim-5.1.0",
        "episodes_requested": args.episodes,
        "episodes_attempted": attempted,
        "episodes_kept": kept,
        "expert_success_rate": kept / max(1, attempted) if not args.keep_failures else None,
        "randomisation": args.randomisation,
        "expert_noise": args.expert_noise,
        "horizon": horizon,
        "wall_seconds": round(time.time() - t0, 1),
        "source": "src/policies/scripted_expert.py",
    }),
}

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
np.savez_compressed(args.output, **data)
print("wrote {} - {} episodes, {} transitions, {:.2f} MB".format(
    args.output, kept, len(obs_list), os.path.getsize(args.output) / 1e6))

env.close()
_app.close()
