"""Train a policy in Isaac, with this repository's own SAC.

    C:\\isaac\\venv311\\Scripts\\python.exe scripts/isaac_train.py \\
        --num-envs 64 --steps 300000 --randomisation none \\
        --output experiments/runs/isaac_sac_none

Deliberately the same `src/policies/sac.py` that produced every MuJoCo number,
not a framework wrapper. The point of the port is that the two simulators run
the same task; running them with the same learner as well keeps the comparison
honest, and it is the only way a difference in the curves can be attributed to
the simulator rather than to the algorithm.

The one structural difference is throughput: Isaac steps `num_envs` worlds at
once, so each environment step contributes `num_envs` transitions to a single
shared replay buffer. Gradient steps are counted per environment step, so a
run of N steps here does the same number of updates as a run of N steps in
MuJoCo but sees `num_envs` times as much data.

Writes the same `progress.csv` / `result.json` layout as `src/train_rl.py`, so
`analysis/plots.py` can read it without special-casing.
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
parser.add_argument("--steps", type=int, default=200_000,
                    help="environment steps (each contributes num_envs transitions)")
parser.add_argument("--randomisation", default="none")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--start-steps", type=int, default=500)
parser.add_argument("--eval-every", type=int, default=2_000)
parser.add_argument("--eval-episodes", type=int, default=2)
parser.add_argument("--demos", default=None,
                    help="npz of Isaac demonstrations; pins them in the replay buffer "
                         "and behaviour-clones the actor before RL starts")
parser.add_argument("--bc-epochs", type=int, default=40)
parser.add_argument("--bc-coef", type=float, default=50.0)
parser.add_argument("--bc-decay-steps", type=int, default=0,
                    help="defaults to half the run")
parser.add_argument("--alpha-floor", type=float, default=0.0,
                    help="lower clamp on the entropy coefficient; the fix for the "
                         "grasp-and-hold local optimum, measured in docs/exploration.md")
parser.add_argument("--output", default="experiments/runs/isaac_sac")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.isaac.grasp_task import ACT_DIM, OBS_DIM, GraspTask, GraspTaskCfg  # noqa: E402
from src.policies.behaviour_cloning import fit  # noqa: E402
from src.policies.sac import SAC, SACConfig  # noqa: E402

os.makedirs(args.output, exist_ok=True)
torch.manual_seed(args.seed)

cfg = GraspTaskCfg()
cfg.scene.num_envs = args.num_envs
cfg.randomisation_level = args.randomisation
env = GraspTask(cfg)

sac_cfg = SACConfig(
    hidden=(args.hidden, args.hidden),
    start_steps=0 if args.demos else args.start_steps,
    buffer_size=600_000,
    bc_coef=args.bc_coef if args.demos else 0.0,
    bc_decay_steps=(args.bc_decay_steps or args.steps // 2) if args.demos else 0,
    demo_sample_fraction=0.25 if args.demos else 0.0,
    critic_warmup_updates=3_000 if args.demos else 0,
    target_entropy_scale=2.0 if args.demos else 1.0,
    init_alpha=0.02 if args.demos else 0.1,
    alpha_floor=args.alpha_floor,
)
agent = SAC(OBS_DIM, ACT_DIM, sac_cfg, seed=args.seed)

if args.demos:
    # Same recipe the MuJoCo runs use: pin the demonstrations so the ring never
    # overwrites them, clone the actor first, then let RL improve on it.
    with np.load(args.demos, allow_pickle=True) as archive:
        demos = {k: archive[k] for k in archive.files}
    n_demo = agent.buffer.add_demonstrations(
        demos["observations"], demos["actions"], demos["rewards"],
        demos["next_observations"], demos["dones"],
    )
    agent.observe_normalisation(demos["observations"])
    print("pinned {} demonstration transitions".format(n_demo), flush=True)

    train_curve, val_curve = fit(
        agent.actor, demos["observations"], demos["actions"],
        epochs=args.bc_epochs, batch_size=256, lr=1e-3, weight_decay=1e-5,
        rng=np.random.default_rng(args.seed),
    )
    print("behaviour cloning done: train mse {:.5f}, val mse {:.5f}".format(
        train_curve[-1], val_curve[-1]), flush=True)

with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "algorithm": "sac",
        "simulator": "isaac-sim-5.1.0",
        "sac": sac_cfg.to_dict(),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "randomisation": args.randomisation,
    }, fh, indent=2)

progress_path = os.path.join(args.output, "progress.csv")
with open(progress_path, "w", encoding="utf-8") as fh:
    fh.write("step,success_rate,grasp_rate,mean_return,mean_max_lift,"
             "train_return,alpha,critic_loss,actor_loss,bc_coef,wall_seconds\n")


def evaluate(episodes: int) -> dict:
    """Deterministic rollouts across all envs, stopping short of the auto-reset."""
    successes = trials = 0
    peaks, returns = [], []
    for _ in range(episodes):
        obs_dict, _ = env.reset()
        peak = torch.zeros(env.num_envs, device=env.device)
        total = torch.zeros(env.num_envs, device=env.device)
        for _ in range(int(env.max_episode_length) - 2):
            with torch.no_grad():
                action, _ = agent.actor(
                    obs_dict["policy"].cpu(), deterministic=True, with_logprob=False
                )
            obs_dict, reward, _, _, _ = env.step(action.to(env.device))
            total += reward
            peak = torch.maximum(peak, env._object_pos()[:, 2] - env.object_rest_z)
        successes += int(env.success().sum())
        trials += env.num_envs
        peaks.append(float(peak.mean()))
        returns.append(float(total.mean()))
    return {
        "success_rate": successes / max(1, trials),
        "grasp_rate": float(env._grasped().mean()),
        "mean_return": float(np.mean(returns)),
        "mean_max_lift": float(np.mean(peaks)),
    }


rng = np.random.default_rng(args.seed)
obs_dict, _ = env.reset()
obs = obs_dict["policy"].cpu().numpy()
episode_return = np.zeros(env.num_envs, dtype=np.float64)
recent_returns: list = []
best_success = -1.0
last_metrics: dict = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": sac_cfg.init_alpha}
t0 = time.time()

for step in range(1, args.steps + 1):
    if step <= sac_cfg.start_steps:
        action_np = rng.uniform(-1.0, 1.0, (env.num_envs, ACT_DIM)).astype(np.float32)
    else:
        with torch.no_grad():
            act_t, _ = agent.actor(
                torch.as_tensor(obs, dtype=torch.float32), deterministic=False,
                with_logprob=False,
            )
        action_np = act_t.numpy()

    obs_dict, reward, terminated, truncated, _ = env.step(
        torch.as_tensor(action_np, device=env.device)
    )
    next_obs = obs_dict["policy"].cpu().numpy()
    reward_np = reward.cpu().numpy()
    term_np = terminated.cpu().numpy()
    done_np = np.logical_or(term_np, truncated.cpu().numpy())

    for i in range(env.num_envs):
        agent.buffer.add(obs[i], action_np[i], reward_np[i], next_obs[i], float(term_np[i]))

    episode_return += reward_np
    if done_np.any():
        recent_returns.extend(episode_return[done_np].tolist())
        recent_returns = recent_returns[-40:]
        episode_return[done_np] = 0.0
    obs = next_obs

    if step % sac_cfg.update_every == 0 and agent.buffer.size >= sac_cfg.batch_size:
        agent.observe_normalisation(obs)
        for _ in range(sac_cfg.update_every):
            last_metrics = agent.update(step)

    if step % args.eval_every == 0 or step == args.steps:
        result = evaluate(args.eval_episodes)
        with open(progress_path, "a", encoding="utf-8") as fh:
            row = ("{},{:.4f},{:.4f},{:.3f},{:.4f},{:.3f},{:.4f},{:.4f},"
                   "{:.4f},0.0,{:.1f}\n")
            fh.write(row.format(
                step, result["success_rate"], result["grasp_rate"], result["mean_return"],
                result["mean_max_lift"],
                float(np.mean(recent_returns)) if recent_returns else float("nan"),
                last_metrics.get("alpha", 0.0), last_metrics.get("critic_loss", 0.0),
                last_metrics.get("actor_loss", 0.0), time.time() - t0))
        if result["success_rate"] > best_success:
            best_success = result["success_rate"]
            torch.save(agent.state_dict(), os.path.join(args.output, "best.pt"))
        print("step {:>7d}  success {:.3f}  grasp {:.3f}  lift {:.3f}  "
              "return {:8.1f}  {:.0f}s".format(
                  step, result["success_rate"], result["grasp_rate"],
                  result["mean_max_lift"], result["mean_return"],
                  time.time() - t0), flush=True)
        # Training resumes from a fresh reset after evaluation.
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"].cpu().numpy()
        episode_return[:] = 0.0

torch.save(agent.state_dict(), os.path.join(args.output, "policy.pt"))
final = evaluate(args.eval_episodes)
with open(os.path.join(args.output, "result.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "final_success_rate": final["success_rate"],
        "best_success_rate": best_success,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "transitions": args.steps * args.num_envs,
        "seed": args.seed,
        "randomisation": args.randomisation,
        "simulator": "isaac-sim-5.1.0",
        "wall_seconds": round(time.time() - t0, 1),
    }, fh, indent=2)
print("final success {:.3f}, best {:.3f}".format(final["success_rate"], best_success))

env.close()
_app.close()
