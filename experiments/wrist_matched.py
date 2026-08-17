"""Does the wrist help, when the comparison is actually matched?

    python experiments/wrist_matched.py --jobs 6

Three attempts were needed to answer this, and the first two were wrong in
opposite directions. The cause was one line in ``GraspEnv``: the object size cap
is chosen *by the wrist flag*, 0.034 m with a wrist and 0.024 m without, unless
a caller overrides it. So turning the wrist on also makes the boxes bigger, and
a naive ``--wrist`` versus no-``--wrist`` comparison is between two different
tasks. The comment above that line has always said so. Until this experiment
existed, nothing ever passed the override.

The cap matters because of the pad gap. At 78 mm, a square box of half-size *h*
fits aligned when ``2h < 0.078`` and does not fit at 45 degrees when
``2h*sqrt(2) > 0.078``, so yaw *binds* only for *h* between 27.6 and 39 mm. The
default 24 mm cap sits below that band on purpose: without a wrist the pads
always close along world x, so a box that cannot be rotated into the gap would
be unsolvable rather than hard. That is a sensible default and a terrible thing
to vary silently between the two arms of an ablation.

This runs the full 2x2 -- wrist and no wrist, at both caps -- with everything
else held still. Each cell gets its own demonstrations, recorded by the same
scripted expert under the conditions that cell is evaluated in.

``bc``       behaviour cloning from 200 demonstrations
``hold``     demonstration-seeded SAC with the anchor held
             (``--bc-decay-steps 0``), the configuration that works everywhere
             else in this repository
``scratch``  from scratch with the entropy floor. This is the arm the inherited
             "0.000 with the wrist against 0.122 without" claim came from, which
             carries the same confound and is re-run here matched.

Measured on the first four cells: small deltas straddling zero, negative for
cloning (-0.034, -0.046) and positive for held-anchor RL (+0.014, +0.074),
rather than the two-to-one gap the unmatched comparison produced.

Those numbers were produced by the equivalent runs before this driver existed,
under the ad-hoc names below. Re-running this script reproduces them under the
systematic names; the old directories are kept so the published figures stay
traceable to the runs that made them.

    wristbig     bc -> wristbc        hold -> wristhold        scratch -> fswb
    nowristbig   bc -> nowristbig     hold -> nowristbighold   scratch -> fsnb
    wristsmall   bc -> wristsmall     hold -> wristsmallhold   scratch -> fsws
    nowristsmall bc -> nowristbc      hold -> nowristhold      scratch -> fsns
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import HIDDEN, REPO, RUNS, run_batch  # noqa: E402

LEVEL = "wrist_bench"
ALPHA_FLOOR = "0.15"
STEPS = 200_000
# (cell name, wrist, cap). Both caps for both hands: the whole point.
CELLS = [
    ("wristbig", True, "0.034"),
    ("nowristbig", False, "0.034"),
    ("wristsmall", True, "0.024"),
    ("nowristsmall", False, "0.024"),
]


def demo_path(cell: str) -> str:
    return os.path.join("demonstrations", "expert_{}.npz".format(cell))


def job_demos(cell: str, wrist: bool, cap: str, episodes: int) -> Dict:
    out = demo_path(cell)
    cmd = [sys.executable, "src/record_demos.py", "--episodes", str(episodes),
           "--randomisation", LEVEL, "--seed", "7", "--max-half-size", cap,
           "--output", out]
    if wrist:
        cmd.append("--wrist")
    return {"name": "demos:" + cell, "output": out, "cmd": cmd}


def job_bc(cell: str, wrist: bool, cap: str, seed: int) -> Dict:
    out = os.path.join(RUNS, "{}bc_s{}".format(cell, seed))
    cmd = [sys.executable, "src/train_il.py", "--demos", demo_path(cell),
           "--seed", str(seed), "--epochs", "60", "--randomisation", LEVEL,
           "--max-half-size", cap, "--hidden", str(HIDDEN),
           "--eval-episodes", "50", "--quiet", "--output", out]
    if wrist:
        cmd.append("--wrist")
    return {"name": os.path.basename(out), "output": out, "cmd": cmd}


def job_hold(cell: str, wrist: bool, cap: str, seed: int) -> Dict:
    out = os.path.join(RUNS, "{}hold_s{}".format(cell, seed))
    init = os.path.join(RUNS, "{}bc_s{}".format(cell, seed), "policy.pt")
    cmd = [sys.executable, "src/train_rl.py", "--steps", str(STEPS),
           "--seed", str(seed), "--randomisation", LEVEL,
           "--max-half-size", cap, "--hidden", str(HIDDEN),
           "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
           "--demos", demo_path(cell), "--demo-fraction", "0.25",
           "--bc-coef", "50.0", "--bc-decay-steps", "0",
           "--critic-warmup", "3000", "--target-entropy-scale", "2.0",
           "--init-alpha", "0.02", "--init-actor", init, "--output", out]
    if wrist:
        cmd.append("--wrist")
    return {"name": os.path.basename(out), "output": out, "needs": init,
            "cmd": cmd}


def job_scratch(cell: str, wrist: bool, cap: str, seed: int) -> Dict:
    out = os.path.join(RUNS, "{}scratch_s{}".format(cell, seed))
    cmd = [sys.executable, "src/train_rl.py", "--steps", str(STEPS),
           "--seed", str(seed), "--randomisation", LEVEL,
           "--max-half-size", cap, "--hidden", str(HIDDEN),
           "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
           "--alpha-floor", ALPHA_FLOOR, "--output", out]
    if wrist:
        cmd.append("--wrist")
    return {"name": os.path.basename(out), "output": out, "cmd": cmd}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--demo-episodes", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--arms", nargs="+", default=["bc", "hold", "scratch"])
    args = parser.parse_args()
    os.chdir(REPO)

    run_batch([job_demos(c, w, cap, args.demo_episodes)
               for c, w, cap in CELLS], args.jobs)
    if "bc" in args.arms:
        run_batch([job_bc(c, w, cap, s)
                   for c, w, cap in CELLS for s in args.seeds], args.jobs)

    later: List[Dict] = []
    if "hold" in args.arms:
        later += [job_hold(c, w, cap, s)
                  for c, w, cap in CELLS for s in args.seeds]
    if "scratch" in args.arms:
        later += [job_scratch(c, w, cap, s)
                  for c, w, cap in CELLS for s in args.seeds]
    if later:
        run_batch(later, args.jobs)

    results = {}
    for cell, _, _ in CELLS:
        for arm in args.arms:
            label = cell + arm
            out = os.path.join("experiments", "results", label + "_eval.json")
            subprocess.run(
                [sys.executable, "src/evaluate.py", "--runs",
                 "experiments/runs/{}_s*".format(label), "--eval-levels", LEVEL,
                 "--episodes", str(args.episodes), "--label", label,
                 "--output", out], cwd=REPO, check=False)
            if os.path.exists(out):
                with open(out, "r", encoding="utf-8") as fh:
                    results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "wrist_matched.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": STEPS, "cells": CELLS,
                   "note": "the object size cap is chosen by the wrist flag "
                           "unless overridden, so every arm here passes "
                           "--max-half-size explicitly. Comparing across caps "
                           "compares two tasks, which is the error this "
                           "experiment exists to prevent",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
