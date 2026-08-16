"""Why RL fine-tuning destroys a working clone on the arm, and whether it has to.

    python experiments/arm_finetune.py --jobs 5

Through the mocap weld, demonstration-seeded SAC improves on its own clone.
Through the six-jointed arm it does the opposite, and the nominal-world
demonstrations make the effect impossible to miss: clones at 0.24-0.62, and the
same clones after 200 000 steps of fine-tuning at 0.000-0.067.

That is a *new* limitation, created by this repository's own arm variant, and it
has three candidate mechanisms. Each is a one-flag change from the run that
failed, so each gets a controlled arm rather than an argument.

``hold``    ``--bc-decay-steps 0``: the cloning term never decays, so the actor
            is leashed to the demonstrations for the whole run while the critic
            still learns. If the clone survives at its original score, RL's
            contribution here is purely negative and the decay schedule is the
            mechanism. If it beats the clone, RL helps when it is not allowed to
            take over.

``warmup``  ``--critic-warmup 20000`` instead of 3 000: the actor sees no policy
            gradient until the critic has had seven times as long. If this
            rescues it, the problem is that the actor was following a critic
            that had not yet learned anything about the arm's dynamics -- which
            is the more forgivable version of the failure.

``entropy`` ``--target-entropy-scale 0.5`` and ``--alpha-floor 0``: the standard
            settings deliberately keep exploration alive, which is what rescues
            *from-scratch* runs. On a good clone in a chain with joint limits and
            self-collision, injected exploration may simply be a way of walking
            off a narrow manifold. This turns it down.

All three start from the same nominal-world clones and run the same 200 000
steps as the arm that failed, so the comparison is one flag wide.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.arm_task import DEMO_SETS, HIDDEN, LEVEL  # noqa: E402
from experiments.place_task import REPO, RUNS, run_batch  # noqa: E402

DEMOS = DEMO_SETS["none"]

# arm -> the flags that differ from the run that failed
VARIANTS = {
    "hold": ["--bc-decay-steps", "0"],
    "warmup": ["--critic-warmup", "20000"],
    "entropy": ["--target-entropy-scale", "0.5", "--alpha-floor", "0.0"],
}


def job(variant: str, seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "arm_ft{}_s{}".format(variant, seed))
    init = os.path.join(RUNS, "arm_bcnom_s{}".format(seed), "policy.pt")
    cmd = [sys.executable, "src/train_rl.py", "--steps", str(steps),
           "--seed", str(seed), "--randomisation", LEVEL, "--arm",
           "--hidden", HIDDEN, "--eval-every", "25000",
           "--eval-episodes", "30", "--quiet",
           "--demos", DEMOS, "--demo-fraction", "0.25",
           "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
           "--critic-warmup", "3000", "--target-entropy-scale", "2.0",
           "--init-alpha", "0.02", "--init-actor", init, "--output", out]
    # The variant's flags go last so argparse takes them over the defaults above.
    return {"name": os.path.basename(out), "output": out, "needs": init,
            "cmd": cmd + VARIANTS[variant]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shifted"])
    args = parser.parse_args()
    os.chdir(REPO)

    jobs: List[Dict] = [job(v, s, args.steps)
                        for v in args.variants for s in args.seeds]
    run_batch(jobs, args.jobs)

    results = {}
    for variant in args.variants:
        label = "arm_ft" + variant
        out = os.path.join("experiments", "results", label + "_eval.json")
        print("evaluating " + label, flush=True)
        subprocess.run([sys.executable, "src/evaluate.py", "--runs",
                        "experiments/runs/{}_s*".format(label), "--arm",
                        "--eval-levels", *args.eval_levels, "--episodes",
                        str(args.episodes), "--label", label, "--output", out],
                       cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "arm_finetune.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"level": LEVEL, "steps": args.steps, "variants": VARIANTS,
                   "control": "arm_bcrlnom (the run that destroyed the clone) "
                              "and arm_bcnom (the clone itself)",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
