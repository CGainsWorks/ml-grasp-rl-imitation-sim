"""The matched-budget control for the floor grid, and the test of one anomaly.

    python experiments/floor_control.py --jobs 8

`experiments/results/floor_grid.json` compares the entropy floor at 300 000
steps against the original grid at 200 000. Two things are wrong with reading
that as the effect of the floor.

**The budgets differ.** Half again as much training is not nothing, and the
floor rows would look better than they deserve if some of the gain is simply
the extra 100 000 steps. This runs the same three randomised levels, the same
five seeds, the same 300 000 steps, with ``--alpha-floor 0`` — the only
difference from the floor arm is the one line under test.

**One row is not monotone.** With the floor, `medium` reaches 0.680 and `high`
0.407, but `low` scores exactly 0.000 on all five seeds, and its grasp rate sits
flat around 0.3 for the whole run against 0.9 for the levels that work. `low` is
a *milder* randomisation than `medium` — the same parameter ranges at 0.4 of the
width — so a level that fails while a harder one succeeds is either a bug or a
mechanism, and it should not be written up as either until the control says
which. If the control also fails at `low`, the floor is not the cause and the
level is genuinely awkward; if the control succeeds, the floor is actively
harmful there and the recommendation has to say so.

`low` is queued first for that reason.
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


def run_dir(level: str, seed: int) -> str:
    return os.path.join(RUNS, "nofloor300_{}_s{}".format(level, seed))


def job(level: str, seed: int, steps: int) -> Dict:
    out = run_dir(level, seed)
    return {
        "name": os.path.basename(out),
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", level, "--hidden", "128",
            "--eval-every", "25000", "--eval-episodes", "30", "--quiet",
            "--alpha-floor", "0.0", "--output", out,
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


def read(path: str) -> List[float]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.load(fh)["final_success_rate"]]


def grasp_rate(run: str) -> float:
    """Final grasp rate from the progress log, for diagnosing the `low` row."""
    path = os.path.join(run, "progress.csv")
    if not os.path.exists(path):
        return float("nan")
    with open(path, "r", encoding="utf-8") as fh:
        rows = fh.read().strip().splitlines()
    return float(rows[-1].split(",")[2])


def summarise(levels: List[str], seeds: List[int], output: str) -> None:
    sys.path.insert(0, REPO)
    from src.utils.stats import t_interval
    from experiments.floor_grid import run_dir as floor_run_dir

    rows = []
    for level in levels:
        control, floor, control_grasp, floor_grasp = [], [], [], []
        for seed in seeds:
            control += read(os.path.join(run_dir(level, seed), "result.json"))
            floor += read(os.path.join(floor_run_dir(level, seed), "result.json"))
            control_grasp.append(grasp_rate(run_dir(level, seed)))
            floor_grasp.append(grasp_rate(floor_run_dir(level, seed)))
        if not control:
            continue
        ci, fi = t_interval(control), t_interval(floor) if floor else None
        rows.append({
            "level": level,
            "control_300k": {"per_seed": control, "grasp_per_seed": control_grasp,
                             **ci.as_dict()},
            "with_floor_300k": {"per_seed": floor, "grasp_per_seed": floor_grasp,
                                **(fi.as_dict() if fi else {})},
        })
        print("{:<7s} control {:.3f} [{:.3f}, {:.3f}]  grasp {:.2f}   "
              "floor {:.3f}  grasp {:.2f}".format(
                  level, ci.point, ci.low, ci.high,
                  sum(control_grasp) / len(control_grasp),
                  fi.point if fi else float("nan"),
                  sum(floor_grasp) / len(floor_grasp)), flush=True)

    blob = {
        "question": "how much of the floor grid is the floor, and why does the "
                    "`low` row fail while `medium` and `high` succeed?",
        "seeds": seeds,
        "steps": 300_000,
        "rows": rows,
        "note": "Both columns are 300 000 steps, five seeds, identical except "
                "for --alpha-floor. Grasp rate is the final evaluation's, and "
                "separates 'never learned to pick the box up' from 'picks it up "
                "and will not lift it', which are different failures.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # `low` first: it is the row the grid cannot explain.
    parser.add_argument("--levels", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "floor_control.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    if not args.summarise_only:
        jobs = [job(level, seed, args.steps)
                for level in args.levels for seed in args.seeds]
        run_batch(jobs, args.jobs)
    summarise(args.levels, args.seeds, args.output)


if __name__ == "__main__":
    main()
