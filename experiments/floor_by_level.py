"""Does the best entropy floor move as the randomisation widens?

    python experiments/floor_by_level.py --jobs 6

`experiments/low_anomaly.py` answered its question and raised a bigger one. At
`low` randomisation, five seeds each, 300 000 steps:

    floor 0.00   0.113 [0.000, 0.428]
    floor 0.05   0.587 [0.310, 0.864]     t = 3.1 against no floor
    floor 0.15   0.000 [0.000, 0.000]
    floor 0.30   0.000

So the floor is not harmful at `low` after all -- 0.15 is simply the wrong
value there, and the right one is five times smaller. On the *nominal* world the
sweep in `docs/exploration.md` says the opposite: 0.05 was the only value that
failed, and everything from 0.10 to 0.50 rescued every seed.

That suggests the useful floor shrinks as the environment supplies more
stochasticity of its own, which would be a satisfying story and is exactly the
kind of story worth trying to break. It already has a hole: `medium` and `high`
are *wider* than `low` and they do fine at 0.15. Either the trend is not
monotone, or 0.05 would be better still at those levels and nobody has looked.

This looks. It trains `medium` and `high` at floor 0.05, five seeds each, and
reads them against every floor value already on disk:

    level    floor 0.00              floor 0.05      floor 0.15
    none     sac_none_s* (200k)      explore_floor005_s* (100k, 3 seeds)
    low      nofloor300_low_s*       lowfloor005_s*  floorgrid_low_s*
    medium   nofloor300_medium_s*    <trained here>  floorgrid_medium_s*
    high     nofloor300_high_s*      <trained here>  floorgrid_high_s*

Budgets differ *between* levels (100 000 at `none`, 300 000 under
randomisation) and match *within* each level, which is what the comparison
needs: the question is which floor wins at a given level, not how the levels
compare to each other.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join("experiments", "runs")
LOGS = os.path.join("experiments", "logs")

# (level, floor) -> run directory template. Anything absent is trained here.
EXISTING = {
    ("none", 0.0): "sac_none_s{}",
    ("none", 0.05): "explore_floor005_s{}",
    ("none", 0.15): "explore_alphafloor_s{}",
    ("none", 0.30): "explore_floor030_s{}",
    ("low", 0.0): "nofloor300_low_s{}",
    ("low", 0.05): "lowfloor005_s{}",
    ("low", 0.15): "floorgrid_low_s{}",
    ("low", 0.30): "lowfloor030_s{}",
    ("medium", 0.0): "nofloor300_medium_s{}",
    ("medium", 0.15): "floorgrid_medium_s{}",
    ("high", 0.0): "nofloor300_high_s{}",
    ("high", 0.15): "floorgrid_high_s{}",
}

# `medium` seeds 0, 2 and 4 at floor 0.15 were trained by the exploration
# follow-up before the grid existed, under a different name.
ALIASES = {
    ("medium", 0.15, 0): "explore_medium300_s0",
    ("medium", 0.15, 2): "explore_medium300_s2",
    ("medium", 0.15, 4): "explore_medium300_s4",
}


def run_dir(level: str, floor: float, seed: int) -> str:
    if (level, floor, seed) in ALIASES:
        return os.path.join(RUNS, ALIASES[(level, floor, seed)])
    if (level, floor) in EXISTING:
        return os.path.join(RUNS, EXISTING[(level, floor)].format(seed))
    return os.path.join(RUNS, "floorlvl_{}_{:03d}_s{}".format(
        level, int(round(floor * 100)), seed))


def job(level: str, floor: float, seed: int, steps: int) -> Dict:
    out = run_dir(level, floor, seed)
    return {
        "name": os.path.basename(out),
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", level, "--hidden", "128",
            "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
            "--alpha-floor", str(floor), "--output", out,
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


def read(run: str) -> Optional[float]:
    path = os.path.join(run, "result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["final_success_rate"]


def summarise(levels: List[str], floors: List[float], seeds: List[int], output: str) -> None:
    sys.path.insert(0, REPO)
    from src.utils.stats import t_interval

    cells = []
    print("{:<8s}".format("level") + "".join("{:>22s}".format("floor " + str(f))
                                             for f in floors))
    for level in levels:
        line = "{:<8s}".format(level)
        for floor in floors:
            values = [v for v in (read(run_dir(level, floor, s)) for s in seeds)
                      if v is not None]
            if not values:
                line += "{:>22s}".format("-")
                continue
            interval = t_interval(values)
            cells.append({
                "level": level,
                "alpha_floor": floor,
                "per_seed": values,
                **interval.as_dict(),
            })
            line += "{:>22s}".format("{:.3f} [{:.2f},{:.2f}] n{}".format(
                interval.point, interval.low, interval.high, len(values)))
        print(line, flush=True)

    blob = {
        "question": "does the useful entropy floor shrink as randomisation widens?",
        "levels": levels,
        "floors": floors,
        "seeds": seeds,
        "cells": cells,
        "note": "Budgets match within a level and differ between levels: 100 000 "
                "steps at `none`, 300 000 under randomisation. The comparison is "
                "down each column, not across rows.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["none", "low", "medium", "high"])
    parser.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.05, 0.15, 0.30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train", nargs="+", default=["medium", "high"],
                        help="levels to train at floor 0.05; everything else is reused")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "floor_by_level.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    if not args.summarise_only:
        jobs = [job(level, 0.05, seed, args.steps)
                for level in args.train for seed in args.seeds]
        run_batch(jobs, args.jobs)
    summarise(args.levels, args.floors, args.seeds, args.output)


if __name__ == "__main__":
    main()
