"""How much cheaper can a run get without changing what it learns?

    python experiments/compute_ablation.py --jobs 6

About 90% of a training run here is SAC gradient updates and about 8% is
physics, measured: 0.54 ms per environment step against roughly 10 ms per
update. So the levers worth pulling are all about doing fewer or cheaper
updates, and every one of them trades compute for learning.

Whether that trade is free is an empirical question, and this repository has
twice today published a conclusion that a control run then contradicted. So each
arm changes exactly one knob against the same baseline, on the nominal world
with the entropy floor -- the one configuration that reliably reaches 1.000 on
five seeds out of five, which makes a regression obvious rather than arguable:

``baseline``    128x128, batch 256, one update per step
``narrow``      64x64 -- the observation is 32-D and the action 4-D, so the
                network may simply be oversized
``batch128``    half the batch
``utd0.5``      one update per two environment steps
``narrow+utd``  both of the above, if they are individually free

Wall-clock is reported alongside success, because an arm that is 2x faster and
0.2 worse is not a saving. Three seeds: enough to catch a regression against a
condition whose baseline is 1.000, not enough to rank two arms that both work.
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

ARMS = {
    "baseline": ["--hidden", "128", "--batch-size", "256", "--updates-per-step", "1.0"],
    "narrow": ["--hidden", "64", "--batch-size", "256", "--updates-per-step", "1.0"],
    "batch128": ["--hidden", "128", "--batch-size", "128", "--updates-per-step", "1.0"],
    "utd0.5": ["--hidden", "128", "--batch-size", "256", "--updates-per-step", "0.5"],
    "narrowutd": ["--hidden", "64", "--batch-size", "256", "--updates-per-step", "0.5"],
}


def job(arm: str, seed: int, steps: int) -> Dict:
    name = "compute_{}_s{}".format(arm, seed)
    out = os.path.join(RUNS, name)
    return {
        "name": name,
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", "none",
            "--alpha-floor", "0.15", "--eval-every", "25000",
            "--eval-episodes", "30", "--quiet", *ARMS[arm], "--output", out,
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


def summarise(arms: List[str], seeds: List[int], output: str) -> None:
    sys.path.insert(0, REPO)
    from src.utils.stats import t_interval

    rows = []
    base_wall = None
    for arm in arms:
        scores, walls = [], []
        for seed in seeds:
            path = os.path.join(RUNS, "compute_{}_s{}".format(arm, seed), "result.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            scores.append(result["final_success_rate"])
            walls.append(result.get("wall_seconds", float("nan")))
        if not scores:
            continue
        interval = t_interval(scores)
        wall = sum(walls) / len(walls)
        if arm == "baseline":
            base_wall = wall
        rows.append({
            "arm": arm, "per_seed": scores, "mean_wall_seconds": wall,
            "speedup": (base_wall / wall) if base_wall else 1.0,
            **interval.as_dict(),
        })
        print("{:<11s} success {:.3f} [{:.3f}, {:.3f}]  {}  wall {:.0f}s  speedup {:.2f}x".format(
            arm, interval.point, interval.low, interval.high,
            [round(v, 2) for v in scores], wall, rows[-1]["speedup"]), flush=True)

    blob = {
        "question": "which compute reductions are free on a condition that "
                    "reliably reaches 1.000?",
        "level": "none", "alpha_floor": 0.15, "seeds": seeds,
        "arms": rows,
        "note": "Wall-clock is per run and depends on how many ran at once, so "
                "the speedup column is only meaningful between arms in the same "
                "batch. Success is the number that decides whether a saving is "
                "free.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "compute_ablation.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    if not args.summarise_only:
        run_batch([job(a, s, args.steps) for a in args.arms for s in args.seeds], args.jobs)
    summarise(args.arms, args.seeds, args.output)


if __name__ == "__main__":
    main()
