"""Ten seeds for the shape-variety comparison, because three could not settle it.

    python experiments/shapes_seeds.py --jobs 5

`docs/limitations.md` reported mixed-shape training at 0.167 [0.000, 0.884]
across three seeds -- 0.50, 0.00, 0.00 -- against 0.407 for boxes alone across
five, and said outright that three seeds with that spread cannot establish
anything. The spread is not noise around a mean either; it is bimodal, which is
the signature of the entropy-collapse failure this repository already documents.
Bimodal outcomes are exactly the case where a handful of seeds misleads, so this
takes both arms to ten.

**Matched to the runs already on disk, deliberately including the parts that are
no longer the default.** `shapes_s0-2` and `sacfloor_medium_s0-4` were trained at
an update-to-data ratio of 1.0; the default is now 0.5, because
`experiments/compute_ablation.py` found 1.0 bought nothing on the nominal
benchmark. Reusing the old seeds while training the new ones at 0.5 would put
half of each arm on a different learning rule and quietly make the comparison
about that instead. So the flag is passed explicitly here, and the value is the
old one.

Both arms are then evaluated on **both** distributions -- boxes only and mixed
shapes -- because "does shape variety in training cost anything" and "does it buy
anything on shapes" are different questions and the earlier note conflated them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import REPO, RUNS, run_batch  # noqa: E402

# What sacfloor_medium_s0-4 and shapes_s0-2 were trained with. Not the current
# defaults; see the module docstring.
UPDATES_PER_STEP = "1.0"
ALPHA_FLOOR = "0.15"
HIDDEN = "128"


def job(level: str, seed: int, steps: int) -> Dict:
    name = "shapes_s{}".format(seed) if level == "shapes" \
        else "sacfloor_medium_s{}".format(seed)
    out = os.path.join(RUNS, name)
    return {
        "name": name, "output": out,
        "cmd": [sys.executable, "src/train_rl.py", "--steps", str(steps),
                "--seed", str(seed), "--randomisation", level,
                "--hidden", HIDDEN, "--eval-every", "25000",
                "--eval-episodes", "30", "--quiet",
                "--updates-per-step", UPDATES_PER_STEP,
                "--alpha-floor", ALPHA_FLOOR, "--output", out],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-levels", nargs="+", default=["none", "shapes"])
    args = parser.parse_args()
    os.chdir(REPO)

    jobs: List[Dict] = [job("shapes", s, args.steps) for s in args.seeds]
    jobs += [job("medium", s, args.steps) for s in args.seeds]
    run_batch(jobs, args.jobs)

    results = {}
    for label, pattern in (("shapes", "experiments/runs/shapes_s*"),
                           ("boxes", "experiments/runs/sacfloor_medium_s*")):
        out = os.path.join("experiments", "results",
                           "shapes_seeds_{}.json".format(label))
        cmd = [sys.executable, "src/evaluate.py", "--runs", pattern,
               "--eval-levels", *args.eval_levels, "--episodes",
               str(args.episodes), "--label", label, "--output", out]
        print("evaluating " + label, flush=True)
        subprocess.run(cmd, cwd=REPO, check=False)
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as fh:
                results[label] = json.load(fh)

    summary = os.path.join("experiments", "results", "shapes_seeds.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({"seeds": args.seeds, "steps": args.steps,
                   "updates_per_step": float(UPDATES_PER_STEP),
                   "note": "both arms trained at the update-to-data ratio the "
                           "runs already on disk used (1.0), not the current "
                           "default of 0.5, so the reused seeds and the new "
                           "ones share a learning rule",
                   "results": results}, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
