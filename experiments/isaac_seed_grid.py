"""Five seeds per configuration in Isaac, so its numbers meet the same standard.

    C:\\isaac\\venv311\\Scripts\\python.exe experiments/isaac_seed_grid.py

Every Isaac result before this was one seed, and this repository says elsewhere
-- repeatedly, and about its own MuJoCo runs -- that one seed is an anecdote.
The cross-simulator transfer number proved the point the hard way: a single seed
suggested wide randomisation transferred at 0.41, and the five-seed version put
it at 0.081 with an interval containing zero.

Two arms, five seeds each:

``scratch``   SAC from scratch. Expected to sit at 0.000 in the
              grasp-and-hold-on-the-table optimum; this measures how reliably.
``bcrl``      Demonstration-seeded, with the behaviour-cloning coefficient held
              rather than decayed -- the configuration that reached 0.938 on one
              seed.

Runs are sequential. Isaac hangs if a second environment is constructed in the
same process, and two simulator instances on one 8 GB card is asking for
trouble.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join("experiments", "runs")
LOGS = os.path.join("experiments", "logs")
DEMOS = os.path.join("demonstrations", "isaac_expert_low.npz")


def job(arm: str, seed: int, steps: int, num_envs: int) -> Dict:
    name = "isaac_{}_s{}".format(arm, seed)
    out = os.path.join(RUNS, name)
    cmd = [
        sys.executable, "scripts/isaac_train.py",
        "--num-envs", str(num_envs), "--steps", str(steps),
        "--eval-every", str(max(1, steps // 4)), "--eval-episodes", "1",
        "--randomisation", "none", "--seed", str(seed), "--output", out,
    ]
    if arm == "bcrl":
        # Hold the anchor rather than decaying it: the decaying schedule
        # collapses at the decay point, which is measured in the Isaac README.
        cmd += ["--demos", DEMOS, "--bc-decay-steps", "1000000"]
    return {"name": name, "output": out, "cmd": cmd}


def summarise(arms: List[str], seeds: List[int], output: str) -> None:
    from src.utils.stats import t_interval

    rows = []
    for arm in arms:
        finals, bests = [], []
        for seed in seeds:
            path = os.path.join(RUNS, "isaac_{}_s{}".format(arm, seed), "result.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            finals.append(result["final_success_rate"])
            bests.append(result["best_success_rate"])
        if not finals:
            continue
        interval = t_interval(finals)
        rows.append({
            "arm": arm,
            "final_per_seed": finals,
            "best_per_seed": bests,
            "across_seeds": interval.as_dict(),
        })
        print("{:<8s} final {}  mean {:.3f}  95% t [{:.3f}, {:.3f}]".format(
            arm, [round(f, 3) for f in finals], interval.point,
            interval.low, interval.high), flush=True)

    blob = {
        "simulator": "Isaac Sim 5.1.0 / Isaac Lab 2.3.2",
        "seeds": seeds,
        "arms": rows,
        "note": "Across-seed t intervals, the same standard the MuJoCo tables use.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["scratch", "bcrl"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "isaac_seed_grid.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)

    if not args.summarise_only:
        os.makedirs(LOGS, exist_ok=True)
        jobs = [job(arm, seed, args.steps, args.num_envs)
                for arm in args.arms for seed in args.seeds]
        t0 = time.time()
        for spec in jobs:
            if os.path.exists(os.path.join(spec["output"], "result.json")):
                print("skip {} (finished)".format(spec["name"]), flush=True)
                continue
            print("[{:>5.0f}s] start {}".format(time.time() - t0, spec["name"]), flush=True)
            with open(os.path.join(LOGS, spec["name"] + ".log"), "w", encoding="utf-8") as log:
                subprocess.call(spec["cmd"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            print("[{:>5.0f}s] done  {}".format(time.time() - t0, spec["name"]), flush=True)

    summarise(args.arms, args.seeds, args.output)


if __name__ == "__main__":
    main()
