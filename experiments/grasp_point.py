"""Can a policy learn *where* to grasp, not just where the object is?

    python experiments/grasp_point.py --jobs 5

Every other shape in this repository has its graspable point at its reported
pose, so "go to the object position and close" is a complete strategy. The
handled shape breaks that on purpose: a 96 mm cube, ungraspable along every axis
against a 78 mm pad gap, with a 20 mm handle offset 118 mm out and 34 mm up. The
observation still reports the **body frame**, which sits on the cube.

The environment is measured, not asserted, and the separation is total:

    scripted expert aiming at the reported pose    0/30
    scripted expert aiming at the handle          30/30

Same hand, same episodes, same everything else. The only difference is knowing
where to grasp, which is what makes this a test of selection.

So the question this asks is narrow and answerable: given demonstrations that
*do* select correctly, does a cloned policy learn the selection from an
observation that never states it? The handle's direction is recoverable -- it
lies along the object's own x axis, and the observation carries the orientation
-- so the information is present. Whether a 128x128 network trained on 200
episodes extracts it is a different question.

Three arms, five seeds, the settings the rest of the repository uses:

``handled_bc``    behaviour cloning from 200 expert demonstrations
``handled_bcrl``  demonstration-seeded SAC
``handled_sac``   from scratch, with the entropy floor. Expected to score zero:
                  the reward's reach term pulls towards the reported pose, which
                  is precisely the wrong place, so this arm is closer to a
                  control than to a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import (  # noqa: E402
    ALPHA_FLOOR,
    HIDDEN,
    LEVEL,
    REPO,
    RUNS,
    run_batch,
)

DEMOS = os.path.join("demonstrations", "expert_handled_low.npz")


def job_bc(seed: int) -> Dict:
    out = os.path.join(RUNS, "handled_bc_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_il.py", "--demos", DEMOS,
                "--seed", str(seed), "--epochs", "60", "--randomisation", "low",
                "--handled", "--hidden", str(HIDDEN), "--eval-episodes", "50",
                "--quiet", "--output", out],
    }


def job_sac(seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "handled_sac_s{}".format(seed))
    return {
        "name": os.path.basename(out), "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL, "--handled",
                "--hidden", str(HIDDEN), "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--alpha-floor", str(ALPHA_FLOOR), "--output", out],
    }


def job_bcrl(seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "handled_bcrl_s{}".format(seed))
    init = os.path.join(RUNS, "handled_bc_s{}".format(seed), "policy.pt")
    return {
        "name": os.path.basename(out), "output": out, "needs": init,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", LEVEL, "--handled",
                "--hidden", str(HIDDEN), "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--demos", DEMOS, "--demo-fraction", "0.25",
                "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
                "--critic-warmup", "3000", "--target-entropy-scale", "2.0",
                "--init-alpha", "0.02", "--init-actor", init, "--output", out],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    os.chdir(REPO)

    run_batch([job_bc(s) for s in args.seeds], args.jobs)
    run_batch([job_sac(s, args.steps) for s in args.seeds]
              + [job_bcrl(s, args.steps) for s in args.seeds], args.jobs)

    results = {}
    for label in ("handled_bc", "handled_bcrl", "handled_sac"):
        out = os.path.join("experiments", "results", label + "_eval.json")
        print("evaluating " + label, flush=True)
        subprocess.run([sys.executable, "src/evaluate.py", "--runs",
                        "experiments/runs/{}_s*".format(label), "--handled",
                        "--eval-levels", "none", "shifted", "--episodes",
                        str(args.episodes), "--label", label, "--output", out],
                       cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "grasp_point.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": args.steps,
                   "expert_reference": {"aiming at the reported pose": 0.0,
                                        "aiming at the handle": 1.0},
                   "note": "the observation reports the body frame; the handle "
                           "direction is recoverable from the reported "
                           "orientation, so the information is present and the "
                           "question is whether the network extracts it",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
