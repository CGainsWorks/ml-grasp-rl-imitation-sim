"""Run the policies on estimated pose instead of ground truth.

    python experiments/perception_eval.py

The pose estimator trained by `scripts/train_pose_cnn.py` replaces the object
entries of the observation with its own prediction, rendered from the fixed
front camera at every step. Everything else is unchanged, so the difference
between the two columns is the cost of not being told where the object is.

This is a *sim-only* perception loop and the number it produces should be read
with that in mind: one camera, one lighting condition, one box texture, no
domain gap. It is not evidence that any of these policies would work from real
images. What it does test is the thing the noise model was standing in for --
whether a policy trained on ground-truth pose survives a real estimator's error
structure, which is correlated in time and shaped by the viewpoint rather than
drawn independently each step.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.evaluate import load_actor  # noqa: E402
from src.perception.pose_cnn import PoseCNN  # noqa: E402
from src.utils.stats import summarise_seeds  # noqa: E402

OBJ = slice(8, 11)
OBJ_MINUS_GRIP = slice(11, 14)
GOAL = slice(26, 29)
GOAL_MINUS_OBJ = slice(29, 32)
GRIP = slice(0, 3)


def run_episode(env, actor, model, use_perception, seed):
    obs, _ = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        if use_perception:
            frame = env.render()
            with torch.no_grad():
                x = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
                pred = model(x)[0].numpy()
            # Substitute the estimate everywhere the observation derives from the
            # object's position, so the policy sees one coherent estimate rather
            # than a mixture of estimated and true quantities.
            obs = obs.copy()
            obs[OBJ] = pred
            obs[OBJ_MINUS_GRIP] = pred - obs[GRIP]
            obs[GOAL_MINUS_OBJ] = obs[GOAL] - pred
        with torch.no_grad():
            action, _ = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                              deterministic=True, with_logprob=False)
        obs, _, term, trunc, info = env.step(action.squeeze(0).numpy())
        done = term or trunc
    return int(bool(info.get("is_success", False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="experiments/runs/bcrl_medium_s*")
    parser.add_argument("--level", default="none")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--checkpoint", default="policy.pt")
    parser.add_argument("--model", default="experiments/perception/pose_cnn.pt")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "perception_eval.json"))
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    model = PoseCNN()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    runs = sorted(r for r in glob.glob(args.pattern)
                  if os.path.exists(os.path.join(r, args.checkpoint)))
    print("{} policies, {} episodes each, level {}".format(
        len(runs), args.episodes, args.level))

    results = {"ground_truth": [], "perception": []}
    for run in runs:
        actor = load_actor(os.path.join(run, args.checkpoint))
        for key, use in (("ground_truth", False), ("perception", True)):
            env = make_env(args.level, seed=7, render_mode="rgb_array",
                           width=64, height=64, camera="front_cam")
            hits = sum(run_episode(env, actor, model, use, 5000 + i)
                       for i in range(args.episodes))
            env.close()
            results[key].append(hits)
        print("  {}: ground truth {}/{}, perception {}/{}".format(
            os.path.basename(run), results["ground_truth"][-1], args.episodes,
            results["perception"][-1], args.episodes), flush=True)

    trials = [args.episodes] * len(runs)
    blob = {
        "policies": [os.path.basename(r) for r in runs],
        "episodes_per_policy": args.episodes,
        "level": args.level,
        "ground_truth": summarise_seeds(results["ground_truth"], trials),
        "perception": summarise_seeds(results["perception"], trials),
        "note": "Sim-only perception: one camera, one lighting condition, one "
                "box texture, no domain gap. Tests whether a policy trained on "
                "ground-truth pose survives a real estimator's error structure, "
                "not whether it would work from real images.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    for key in ("ground_truth", "perception"):
        across = blob[key]["across_seeds"]
        print("{:<14s} {:.3f} [{:.3f}, {:.3f}]".format(
            key, across["point"], across["low"], across["high"]))
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
