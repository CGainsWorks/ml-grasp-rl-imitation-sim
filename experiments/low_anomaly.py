"""Why does the entropy floor break `low` randomisation and nothing else?

    python experiments/low_anomaly.py --jobs 8

The matched-budget control (``experiments/results/floor_control.json``) leaves
one thing unexplained. At 300 000 steps, five seeds, everything else identical:

    level    no floor        floor 0.15      final grasp rate
    low      0.113           0.000           0.91 -> 0.33
    medium   0.460           0.680           0.85 -> 0.96
    high     0.160           0.407           0.75 -> 0.85

The success difference at `low` is within noise (t = -1.0). The grasp rate is
not: the floor arm ends up closing on the box in a third of episodes against
nine in ten without it (t = -6.0, dof 4.9). So at `low` the floor is not failing
to escape a local optimum, it is preventing the policy from learning the first
step of the task at all. At `medium` and `high` the same floor raises the grasp
rate instead. And at `none` -- no randomisation whatever -- the floor takes
success from 0.400 to 0.993. Fails in the middle, works at both ends.

Two candidate explanations, and they make opposite predictions:

**The floor value is wrong for this level.** Exploration a policy has to supply
itself is not the only exploration available; a randomised environment supplies
some. If 0.15 is tuned for the nominal world, then a level that adds a little
environment stochasticity might need less, not more. Predicts: a *smaller*
floor works at `low`.

**Wide randomisation carries a friendly tail.** `medium` and `high` draw high
friction and light objects often enough that a still-exploring policy stumbles
into a successful lift and can learn from it; `low` is too narrow to contain
those draws but wide enough to add noise. Predicts: a smaller floor does *not*
help at `low`, because the problem is the draws, not the coefficient.

Five seeds at floor 0.05 and 0.30, against the 0.15 already measured. The 0.05
arm is the discriminating one: on the *nominal* world 0.05 was the one value in
the sweep that failed, so if it succeeds here the direction of the effect has
reversed with the level, which is the first explanation and nothing else.
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

# floor 0.15 at `low` is experiments/runs/floorgrid_low_s*, and floor 0.0 is
# experiments/runs/nofloor300_low_s*. Neither is retrained.
EXISTING = {
    0.0: os.path.join(RUNS, "nofloor300_low_s{}"),
    0.15: os.path.join(RUNS, "floorgrid_low_s{}"),
}


def run_dir(floor: float, seed: int) -> str:
    if floor in EXISTING:
        return EXISTING[floor].format(seed)
    return os.path.join(RUNS, "lowfloor{:03d}_s{}".format(int(round(floor * 100)), seed))


def job(floor: float, seed: int, steps: int) -> Dict:
    out = run_dir(floor, seed)
    return {
        "name": os.path.basename(out),
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", "low", "--hidden", "128",
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


def final(run: str) -> Dict:
    path = os.path.join(run, "result.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    grasp = float("nan")
    progress = os.path.join(run, "progress.csv")
    if os.path.exists(progress):
        with open(progress, "r", encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        if len(rows) > 1:
            grasp = float(rows[-1].split(",")[2])
    return {"success": result["final_success_rate"], "grasp": grasp}


def summarise(floors: List[float], seeds: List[int], output: str) -> None:
    sys.path.insert(0, REPO)
    from src.utils.stats import t_interval

    rows = []
    for floor in floors:
        successes, grasps = [], []
        for seed in seeds:
            got = final(run_dir(floor, seed))
            if got:
                successes.append(got["success"])
                grasps.append(got["grasp"])
        if not successes:
            continue
        interval = t_interval(successes)
        rows.append({
            "alpha_floor": floor,
            "success_per_seed": successes,
            "grasp_per_seed": grasps,
            "across_seeds": interval.as_dict(),
            "mean_grasp": sum(grasps) / len(grasps),
        })
        print("floor {:<5.2f} success {:.3f} [{:.3f}, {:.3f}]  {}  grasp {:.2f}".format(
            floor, interval.point, interval.low, interval.high,
            [round(s, 2) for s in successes], rows[-1]["mean_grasp"]), flush=True)

    blob = {
        "question": "is the floor's failure at `low` the value, or the level?",
        "randomisation": "low",
        "steps": 300_000,
        "seeds": seeds,
        "rows": rows,
        "note": "floor 0.00 is experiments/runs/nofloor300_low_s*, floor 0.15 is "
                "experiments/runs/floorgrid_low_s*; neither was retrained. Grasp "
                "rate is the discriminating measurement, not success: at `low` "
                "the floor arm's failure is that it never closes on the box.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.05, 0.15, 0.30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "low_anomaly.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    if not args.summarise_only:
        jobs = [job(floor, seed, args.steps)
                for floor in args.floors for seed in args.seeds]
        run_batch(jobs, args.jobs)
    summarise(args.floors, args.seeds, args.output)


if __name__ == "__main__":
    main()
