"""Render a dataset for the pose estimator.

    python scripts/collect_pose_data.py --episodes 200

`docs/limitations.md` has said from the beginning that the policy is handed the
object pose, and that the Gaussian noise standing in for a pose estimator is a
weak model of one: a real estimator's error is correlated across frames, biased
by viewpoint, and worst exactly when the gripper occludes the object.

That claim is checkable. This renders the fixed front camera during rollouts and
stores the image alongside the true object pose, so an estimator can be trained
and its *actual* error structure measured against the model in
`src/randomisation/configs/measured.json`.

Two things about the data collection matter more than the network.

The camera is fixed and the arm moves in front of it, so occlusion happens
naturally rather than being simulated -- the frames where the hand is over the
box are the ones a real estimator finds hardest, and they are in the set in
proportion to how often they occur.

And the policy generating the frames is deliberately mixed: the scripted expert
for the trajectories a working system produces, and a random policy for the rest
of the state space. An estimator trained only on expert frames is evaluated on a
distribution it will not see once a *learned* policy is driving.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mujoco.grasp_env import make_env  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--levels", nargs="+", default=["none", "low", "medium"])
    parser.add_argument("--camera", default="front_cam",
                        help="front_cam is fixed; wrist_cam rides the palm")
    parser.add_argument("--clutter", type=int, default=0,
                        help="distractor objects on the table, up to 3")
    parser.add_argument("--expert-fraction", type=float, default=0.7)
    parser.add_argument("--output", default="experiments/perception/pose_data.npz")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rng = np.random.default_rng(0)
    images, poses, occluded = [], [], []

    for ep in range(args.episodes):
        level = args.levels[ep % len(args.levels)]
        env = make_env(level, seed=1000 + ep, render_mode="rgb_array",
                       width=args.resolution, height=args.resolution,
                       camera=args.camera, clutter=args.clutter)
        expert = ScriptedExpert()
        obs, _ = env.reset()
        expert.reset()
        use_expert = rng.random() < args.expert_fraction
        done = False
        while not done:
            frame = env.render()
            obj = env._object_pos().copy()
            grip = env._grip_pos().copy()
            images.append(frame.astype(np.uint8))
            poses.append(obj)
            # A crude but honest occlusion flag. For the fixed camera it is
            # geometric: the hand sits between the camera (at -y) and the box.
            # A wrist camera cannot be occluded by the hand it is bolted to, so
            # for that view the flag records the case that actually degrades it
            # -- the object outside the frustum, because the camera is looking
            # somewhere else entirely. Calling both "occlusion" would compare two
            # different quantities and report it as one.
            if args.camera == "wrist_cam":
                offset = obj - grip
                occluded.append(float(np.linalg.norm(offset[:2]) > 0.10
                                      or offset[2] > 0.02))
            else:
                occluded.append(float(grip[1] < obj[1] and
                                      np.linalg.norm(grip[:2] - obj[:2]) < 0.05))
            action = (expert.act(obs) if use_expert
                      else rng.uniform(-1, 1, env.act_dim).astype(np.float32))
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        env.close()
        if (ep + 1) % 25 == 0:
            print("{} episodes, {} frames".format(ep + 1, len(images)), flush=True)

    images = np.asarray(images, dtype=np.uint8)
    poses = np.asarray(poses, dtype=np.float32)
    occluded = np.asarray(occluded, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, images=images, poses=poses, occluded=occluded)
    print("wrote {}: {} frames, {:.0f} MB, {:.1%} flagged hard ({}, clutter {})".format(
        args.output, len(images), os.path.getsize(args.output) / 1e6,
        occluded.mean(), args.camera, args.clutter))


if __name__ == "__main__":
    main()
