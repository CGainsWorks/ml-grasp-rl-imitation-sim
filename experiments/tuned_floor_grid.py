"""The headline grid again, from scratch, with each level's own entropy floor.

    python experiments/tuned_floor_grid.py --jobs 8

Every from-scratch row in the README was trained before the entropy collapse
was diagnosed, so it records what a standard SAC configuration does here rather
than what the task allows. That is a fair record and it is kept. This is the
same grid with the fix applied, so the two can be read side by side.

The floor is **per level**, taken from the matrix in
`experiments/results/floor_by_level.json`, because that matrix is precisely the
finding that one value does not work everywhere:

    none    0.15      (0.05 fails there)
    low     0.05      (0.15 scores 0.000 there)
    medium  0.15      (0.05 is within noise of it)
    high    0.05      (0.15 is within noise of it)

**200 000 steps, matching the original grid exactly.** The tuned-floor runs
already on disk are at 100 000 and 300 000 steps and are not reused here, which
costs twenty runs of CPU and buys the only comparison worth publishing: the
mistake this repository made twice today was reading a difference between an
intervention and a control that had different budgets. Doing it again in the
headline table would be worse than not doing it at all.

Runs land in `experiments/runs/sacfloor_<level>_s<seed>`, which
`experiments/summarise.py` and `experiments/ablation.py` pick up by pattern.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join("experiments", "runs")
LOGS = os.path.join("experiments", "logs")

# Best floor per level, from experiments/results/floor_by_level.json.
FLOORS = {"none": 0.15, "low": 0.05, "medium": 0.15, "high": 0.05}


def job(level: str, seed: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "sacfloor_{}_s{}".format(level, seed))
    return {
        "name": os.path.basename(out),
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", level, "--hidden", "128",
            "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
            "--alpha-floor", str(FLOORS[level]), "--output", out,
        ],
    }


def run_batch(jobs: List[Dict], parallel: int) -> None:
    os.makedirs(LOGS, exist_ok=True)
    queue = [j for j in jobs if not os.path.exists(os.path.join(j["output"], "result.json"))]
    print("{} to train, {} reused".format(len(queue), len(jobs) - len(queue)), flush=True)
    running: List[Dict] = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < parallel:
            spec = queue.pop(0)
            log = open(os.path.join(LOGS, spec["name"] + ".log"), "w", encoding="utf-8")
            proc = subprocess.Popen(spec["cmd"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            running.append({"spec": spec, "proc": proc, "log": log})
            print("[{:>5.0f}s] start {}".format(time.time() - t0, spec["name"]), flush=True)
        time.sleep(2.0)
        for entry in list(running):
            if entry["proc"].poll() is None:
                continue
            entry["log"].close()
            running.remove(entry)
            print("[{:>5.0f}s] done  {}".format(time.time() - t0, entry["spec"]["name"]),
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["none", "low", "medium", "high"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    os.chdir(REPO)
    jobs = [job(level, seed, args.steps)
            for level in args.levels for seed in args.seeds]
    run_batch(jobs, args.jobs)
    print("now rerun: experiments/summarise.py and experiments/ablation.py "
          "--prefix sacfloor, then analysis/readme_tables.py", flush=True)


if __name__ == "__main__":
    main()
