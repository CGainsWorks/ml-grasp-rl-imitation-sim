"""Train the pose estimator and measure the error it actually makes.

    python scripts/train_pose_cnn.py

The training is routine. The measurement is the point, and it is reported
against the model `src/randomisation/configs/measured.json` assumes:

* **magnitude** -- the sourced range is 4-10 mm of position error, taken from
  published YCB-Video results. Does a small estimator on this task land there?
* **occlusion dependence** -- `docs/limitations.md` claims a real estimator is
  "worst when the gripper occludes the object", which the noise model does not
  represent at all. The dataset carries an occlusion flag, so this is a
  two-sample comparison rather than an assertion.
* **temporal correlation** -- `obs_noise_corr = 0.9` was chosen for plausibility
  and never checked. Frames arrive in episode order, so the lag-1
  autocorrelation of the error is directly measurable.

The split is by *episode*, not by frame. Frames within an episode are nearly
duplicates, so a random frame split leaks the answer across it and reports a
validation error several times better than the truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.pose_cnn import PoseCNN  # noqa: E402

EPISODE_LENGTH = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="experiments/perception/pose_data.npz")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output", default="experiments/perception/pose_cnn.pt")
    parser.add_argument("--label", default="fixed camera, no clutter")
    parser.add_argument("--report",
                        default="experiments/results/perception_error.json")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    blob = np.load(args.data)
    images = blob["images"]
    poses = blob["poses"]
    occluded = blob["occluded"]

    # Split by episode. Neighbouring frames are near-duplicates; a frame-level
    # split lets the model memorise an episode and calls it generalisation.
    n_episodes = len(images) // EPISODE_LENGTH
    rng = np.random.default_rng(0)
    order = rng.permutation(n_episodes)
    n_val = max(1, int(n_episodes * args.val_fraction))
    val_eps, train_eps = order[:n_val], order[n_val:]

    def frames_of(eps):
        return np.concatenate([np.arange(e * EPISODE_LENGTH, (e + 1) * EPISODE_LENGTH)
                               for e in eps])

    tr, va = frames_of(train_eps), frames_of(val_eps)
    print("{} episodes: {} train frames, {} validation frames".format(
        n_episodes, len(tr), len(va)))

    def batch(idx):
        x = torch.as_tensor(images[idx], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y = torch.as_tensor(poses[idx], dtype=torch.float32)
        return x, y

    model = PoseCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        total = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = np.sort(tr[perm[start:start + args.batch_size]])
            x, y = batch(idx)
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            errs = []
            for start in range(0, len(va), 512):
                idx = va[start:start + 512]
                x, y = batch(idx)
                errs.append((model(x) - y).numpy())
            err = np.concatenate(errs)
        print("epoch {:>2d}  train mse {:.5f}  val mean |error| {:.4f} m".format(
            epoch + 1, total / len(tr), float(np.abs(err).mean())), flush=True)

    # --- the measurement
    dist = np.linalg.norm(err, axis=-1)
    occ = occluded[va] > 0.5
    # Frames are in episode order within `va`, so consecutive entries of the
    # same episode are consecutive in time; comparing across an episode boundary
    # would invent a correlation, so those pairs are dropped.
    same_episode = (va[1:] // EPISODE_LENGTH) == (va[:-1] // EPISODE_LENGTH)
    signed = err[:, :2].reshape(-1, 2)
    lag1 = []
    for axis in range(2):
        a = signed[:-1, axis][same_episode]
        b = signed[1:, axis][same_episode]
        lag1.append(float(np.corrcoef(a, b)[0, 1]))

    report = {
        "frames_validation": int(len(va)),
        "mean_position_error_m": float(dist.mean()),
        "median_position_error_m": float(np.median(dist)),
        "error_visible_m": float(dist[~occ].mean()),
        "error_occluded_m": float(dist[occ].mean()),
        "occluded_fraction": float(occ.mean()),
        "lag1_autocorrelation_xy": lag1,
        "label": args.label,
        "modelled": {
            "obs_noise_pos_m": [0.004, 0.010],
            "obs_noise_corr": 0.9,
            "occlusion_dependence": "not modelled",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    torch.save(model.state_dict(), args.output)

    print("\nmeasured error structure")
    print("  mean position error   {:.4f} m   (modelled 0.004-0.010)".format(
        report["mean_position_error_m"]))
    print("  visible / occluded    {:.4f} / {:.4f} m   (occlusion not modelled)".format(
        report["error_visible_m"], report["error_occluded_m"]))
    print("  lag-1 autocorrelation {:.3f}, {:.3f}   (modelled 0.9)".format(*lag1))
    print("wrote " + args.report)


if __name__ == "__main__":
    main()
