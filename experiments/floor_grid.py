"""The randomisation ablation, rerun with the entropy floor.

    python experiments/floor_grid.py --jobs 8

The headline grid in the README was trained before the entropy collapse was
diagnosed, so its from-scratch rows record what a standard SAC configuration
does rather than what this task allows. That is a fair record and it is kept.
This is the same grid with the one-line fix applied, so the two can be compared
directly and the claim "demonstrations are the difference between working and
not working" can be checked rather than assumed.

Budgets differ by level, deliberately, and the reason is measured rather than
guessed. On the nominal world the floor solves the task by about 40 000 steps,
so 100 000 is ample. Under randomisation it needs roughly three times that: at
100 000 steps the floor looks inert at `medium` (0.160 against a 0.120
baseline), and at 300 000 the same seeds reach 0.667. Giving the randomised
levels 200 000 would understate the fix and repeat the mistake that produced the
wrong first conclusion.

Runs already on disk are reused, so this only trains what is missing:

    nominal   experiments/runs/explore_alphafloor_s*      (100k, five seeds)
    medium    experiments/runs/explore_medium300_s{0,2,4} (300k, three seeds)
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

# Where an already-trained run lives, per level and seed. Anything not listed
# here is trained by this script.
EXISTING = {
    ("none", s): os.path.join(RUNS, "explore_alphafloor_s{}".format(s)) for s in range(5)
}
EXISTING.update(
    {("medium", s): os.path.join(RUNS, "explore_medium300_s{}".format(s)) for s in (0, 2, 4)}
)


def run_dir(level: str, seed: int) -> str:
    if (level, seed) in EXISTING:
        return EXISTING[(level, seed)]
    return os.path.join(RUNS, "floorgrid_{}_s{}".format(level, seed))


def job(level: str, seed: int, floor: float, steps: int) -> Dict:
    out = run_dir(level, seed)
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


def summarise(levels: List[str], seeds: List[int], floor: float, output: str) -> None:
    sys.path.insert(0, REPO)
    from src.utils.stats import t_interval

    rows = []
    for level in levels:
        fixed, base = [], []
        for seed in seeds:
            path = os.path.join(run_dir(level, seed), "result.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    fixed.append(json.load(fh)["final_success_rate"])
            baseline = os.path.join(RUNS, "sac_{}_s{}".format(level, seed), "result.json")
            if os.path.exists(baseline):
                with open(baseline, "r", encoding="utf-8") as fh:
                    base.append(json.load(fh)["final_success_rate"])
        if not fixed:
            continue
        fi, bi = t_interval(fixed), t_interval(base) if base else None
        rows.append({
            "level": level,
            "with_floor": {"per_seed": fixed, **fi.as_dict()},
            "baseline": {"per_seed": base, **(bi.as_dict() if bi else {})},
        })
        print("{:<7s} floor {} -> {:.3f} [{:.3f}, {:.3f}]   baseline {:.3f}".format(
            level, [round(f, 2) for f in fixed], fi.point, fi.low, fi.high,
            bi.point if bi else float("nan")), flush=True)

    blob = {
        "alpha_floor": floor,
        "seeds": seeds,
        "rows": rows,
        "note": "Nominal level trained for 100k steps, randomised levels for 300k: "
                "the floor needs about three times the budget under randomisation. "
                "Baseline columns are the original grid at 200k without the floor.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["none", "low", "medium", "high"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--floor", type=float, default=0.15)
    parser.add_argument("--nominal-steps", type=int, default=100_000)
    parser.add_argument("--randomised-steps", type=int, default=300_000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "floor_grid.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    if not args.summarise_only:
        jobs = [
            job(level, seed, args.floor,
                args.nominal_steps if level == "none" else args.randomised_steps)
            for level in args.levels for seed in args.seeds
        ]
        run_batch(jobs, args.jobs)
    summarise(args.levels, args.seeds, args.floor, args.output)


if __name__ == "__main__":
    main()
