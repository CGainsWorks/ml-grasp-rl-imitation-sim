"""Bring up and check the Isaac Lab port.

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_bringup.py --num-envs 4

Runs the checklist from ``envs/isaac/README.md`` in order, and prints a pass or
fail line for each step:

1. the environment constructs and resets;
2. zero actions for 100 steps leave the box on the table;
3. the observation layout matches the MuJoCo table, index by index;
4. the reward computed inside Isaac matches the shared numpy implementation on
   the same state, which is the point of sharing the module;
5. the scripted expert -- the same one that drives the MuJoCo environment --
   can grasp and lift.

Step 5 is the real test. The expert reads the observation vector and nothing
else, so if it cannot grasp here, the environment is wrong rather than the
policy.

The simulation app has to be launched before ``isaaclab`` and the task module
are imported, which is why the imports below are not all at the top.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--randomisation", default="none")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--gui", dest="headless", action="store_false")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=args.headless).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import (  # noqa: E402
    HOLD_HEIGHT,
    TABLE_HEIGHT,
    GraspTask,
    GraspTaskCfg,
)
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402
from src.rewards.grasp_reward import GraspRewardConfig, grasp_reward  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print("[{}] {}{}".format(PASS if ok else FAIL, name, "  " + detail if detail else ""),
          flush=True)


# ---------------------------------------------------------------- 1. construct
cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)
obs_dict, _ = env.reset()
obs = obs_dict["policy"]
check("environment constructs and resets", obs.shape == (args.num_envs, 32),
      "observation {}".format(tuple(obs.shape)))

# ---------------------------------------------------------------- 2. idle
zero = torch.zeros((args.num_envs, 4), device=env.device)
for _ in range(100):
    obs_dict, _, _, _, _ = env.step(zero)
obj_z = env._object_pos()[:, 2]
check("box stays on the table under zero actions",
      bool(((obj_z > TABLE_HEIGHT - 0.02) & (obj_z < TABLE_HEIGHT + 0.10)).all()),
      "object z {}".format(np.round(obj_z.cpu().numpy(), 3)))

# ---------------------------------------------------------------- 3. layout
obs = obs_dict["policy"]
grip, obj, goal = obs[:, 0:3], obs[:, 8:11], obs[:, 26:29]
layout_ok = (
    torch.allclose(obs[:, 11:14], obj - grip, atol=1e-4)
    and torch.allclose(obs[:, 29:32], goal - obj, atol=1e-4)
    and bool((goal[:, 2] > TABLE_HEIGHT + HOLD_HEIGHT - 1e-3).all())
)
check("observation layout matches the MuJoCo table", layout_ok)

# ---------------------------------------------------------------- 4. reward parity
torch_reward = env._get_rewards()
np_reward, _ = grasp_reward(
    env._grip_pos().cpu().numpy().astype(np.float64),
    env._object_pos().cpu().numpy().astype(np.float64),
    env.goal_pos.cpu().numpy().astype(np.float64),
    env.object_rest_z.cpu().numpy().astype(np.float64),
    env._grasped().cpu().numpy().astype(np.float64),
    np.zeros(args.num_envs),
    env.last_action.cpu().numpy().astype(np.float64),
    GraspRewardConfig(),
)
max_diff = float(np.abs(torch_reward.cpu().numpy() - np_reward).max())
check("reward matches the shared numpy implementation", max_diff < 1e-4,
      "max difference {:.2e}".format(max_diff))

# ---------------------------------------------------------------- 5. expert
successes, lifted = 0, 0
for episode in range(args.episodes):
    obs_dict, _ = env.reset()
    experts = [ScriptedExpert() for _ in range(args.num_envs)]
    peak = torch.zeros(args.num_envs, device=env.device)
    # Stop two steps short of the horizon. DirectRLEnv auto-resets the instant
    # the time-out fires, so reading the state after the last step reports the
    # *next* episode's freshly placed box -- which looks exactly like a policy
    # that dropped it.
    for _ in range(int(env.max_episode_length) - 2):
        obs_np = obs_dict["policy"].cpu().numpy()
        actions = np.stack([e.act(obs_np[i]) for i, e in enumerate(experts)])
        obs_dict, _, _, _, _ = env.step(torch.as_tensor(actions, device=env.device))
        peak = torch.maximum(peak, env._object_pos()[:, 2] - env.object_rest_z)
    successes += int(env.success().sum())
    lifted += int((peak > 0.05).sum())
    print("   episode {}: {} of {} held at the goal, {} lifted above 5 cm".format(
        episode, int(env.success().sum()), args.num_envs, int((peak > 0.05).sum())), flush=True)
    print("      peak lift per env: {}".format(np.round(peak.cpu().numpy(), 3)), flush=True)
    print("      final object z:    {}".format(
        np.round(env._object_pos()[:, 2].cpu().numpy(), 3)), flush=True)
    print("      goal z:            {}".format(
        np.round(env.goal_pos[:, 2].cpu().numpy(), 3)), flush=True)
    print("      grasped flag:      {}".format(
        env._grasped().cpu().numpy()), flush=True)

total = args.episodes * args.num_envs
check("scripted expert lifts the box", lifted > 0,
      "{}/{} lifted, {}/{} held at the hold point".format(lifted, total, successes, total))
check("scripted expert holds it at the hold point", successes == total,
      "{}/{} held".format(successes, total))

# ------------------------------------------------- 6. randomisation is not inert
# A randomisation config that silently does nothing is the failure this repo
# guards against in MuJoCo too: training still runs, curves still look fine, and
# the ablation quietly compares identical conditions.
masses, frictions = [], []
for _ in range(6):
    env.reset()
    masses.append(env._object.root_physx_view.get_masses().clone().cpu().numpy().ravel())
    frictions.append(
        env._object.root_physx_view.get_material_properties().clone().cpu().numpy()[..., 0].ravel()
    )
mass_spread = float(np.ptp(np.concatenate(masses)))
friction_spread = float(np.ptp(np.concatenate(frictions)))
if args.randomisation == "none":
    check("randomisation is inert at level 'none'",
          mass_spread < 1e-6 and friction_spread < 1e-6,
          "mass spread {:.4f}, friction spread {:.4f}".format(mass_spread, friction_spread))
else:
    check("randomisation actually varies the world",
          mass_spread > 1e-3 and friction_spread > 1e-3,
          "mass spread {:.4f} kg, friction spread {:.4f}".format(mass_spread, friction_spread))

print("\n{} of {} checks passed".format(sum(1 for _, ok in results if ok), len(results)))
env.close()
_app.close()
sys.exit(0 if all(ok for _, ok in results) else 1)
