"""Where should the arm's base go? Measured, not guessed.

    python experiments/arm_base_search.py

The redesigned chain reaches the workspace and starts cleanly, and still cannot
grasp: it stalls about 14 cm above the box because its forearm fouls the table
on the way down. Adjusting the base by hand is how the previous version of this
arm consumed an afternoon, so this searches instead.

The method follows the standard reachability-map approach to fixed-base
placement -- see Wang et al., [*B*: Efficient and Optimal Base Placement for
Fixed-Base Manipulators*](https://arxiv.org/pdf/2504.12719) for a recent
treatment, and the wheelchair-mounted-arm literature for the inverse-reachability
form. The usual rule of thumb from that work is that the target should sit at
roughly 70-80% of maximum reach; this arm is UR5-proportioned, so about 0.85 m
of reach and 0.60-0.68 m of standoff.

Standoff alone is not the problem here -- the current base is already at 0.62 m.
What matters is the *approach*: from a base at table height the arm reaches
nearly horizontally, so the forearm sweeps low and hits the table before the
fingers reach the box. Mounting higher converts that into a downward approach.

So the score is not "can the flange be placed there" but "how much of the region
where grasping actually happens can be reached with the pads facing down and
nothing intersecting the table". Cells are counted, not samples, so a placement
cannot win by being easy to reach in one spot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

SCENE = os.path.join("envs", "mujoco", "assets", "grasp_scene_arm.xml")

# The region grasping happens in: the table centre, from just above the box to
# the height the hand starts at.
CELLS_XY = np.linspace(-0.14, 0.14, 4)
CELLS_Z = np.array([0.43, 0.47, 0.52, 0.60, 0.66])
ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")


def score_placement(env, base_id, base_pos, rng, restarts=12):
    """Fraction of grasp-region cells the arm can reach cleanly.

    Asked with IK rather than by sampling joint angles. Uniform sampling of six
    joints puts almost nothing inside a 0.32 m box with the pads facing down --
    40 000 samples produced single-digit hits and scored every placement at zero,
    which measured the sampler rather than the arm.
    """
    env.model.body_pos[base_id] = base_pos
    lo = np.array([env.model.joint(n).range[0] for n in ARM_JOINTS])
    hi = np.array([env.model.joint(n).range[1] for n in ARM_JOINTS])

    reached, low_reached = 0, 0
    total, low_total = 0, 0
    for iz, z in enumerate(CELLS_Z):
        for x in CELLS_XY:
            for y in CELLS_XY:
                target = np.array([x, y, z])
                total += 1
                low_total += int(iz == 0)
                ok = False
                for attempt in range(restarts):
                    seed_q = env._arm_home if attempt == 0 else rng.uniform(lo, hi)
                    env.data.qpos[env._arm_qpos] = seed_q
                    mujoco.mj_kinematics(env.model, env.data)
                    env._solve_ik(target, 0.0, iters=150, hold=False)
                    mujoco.mj_forward(env.model, env.data)
                    if np.linalg.norm(env._grip_pos() - target) > 0.015:
                        continue
                    if any(env.data.contact[i].dist < -0.001
                           for i in range(env.data.ncon)):
                        continue
                    ok = True
                    break
                reached += ok
                low_reached += int(ok and iz == 0)
    return reached / total, low_reached / max(1, low_total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "arm_base_search.json"))
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from envs.mujoco.grasp_env import GraspEnv
    env = GraspEnv(arm=True, seed=0)
    base_id = env.model.body("base").id
    rng = np.random.default_rng(0)

    rows = []
    print("{:>7s} {:>7s}  {:>9s}  {:>9s}".format("base y", "base z", "coverage", "low layer"))
    for y in (-0.72, -0.66, -0.60, -0.54, -0.48):
        for z in (0.40, 0.55, 0.70, 0.85, 1.00):
            cov, low = score_placement(env, base_id, np.array([0.0, y, z]), rng)
            rows.append({"y": y, "z": z, "coverage": cov, "low_layer": low})
            print("{:>7.2f} {:>7.2f}  {:>9.3f}  {:>9.3f}".format(y, z, cov, low), flush=True)

    best = max(rows, key=lambda r: (r["low_layer"], r["coverage"]))
    print("\nbest: base at y={y:.2f} z={z:.2f}  coverage {coverage:.3f}  "
          "low layer {low_layer:.3f}".format(**best))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "best": best,
                   "note": "coverage is the fraction of grasp-region cells reached "
                           "with the pads facing down and no link intersecting the "
                           "table; low layer is the same at the grasp height alone, "
                           "which is what decides whether the task is possible."},
                  fh, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
