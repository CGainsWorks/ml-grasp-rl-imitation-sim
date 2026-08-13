"""Run the whole experiment grid, in parallel, and aggregate it.

    python experiments/run_all.py --jobs 6 --steps 200000 --seeds 0 1 2 3 4

The grid:

    sac_<level>_s<seed>    SAC from scratch at each randomisation level
    bc_s<seed>             behaviour cloning from the recorded demonstrations
    bcrl_<level>_s<seed>   BC initialisation, demonstrations pinned in the
                           replay buffer, then SAC
    dagger_s<seed>         DAgger, evaluated on the shifted worlds

Each run is a separate process with a single torch thread: these are small
networks and the parallelism that pays is across runs, not inside them. On an
eight-core machine ``--jobs 6`` keeps a core free for the operating system and
finishes the default grid in roughly two hours.

Runs that already have a ``result.json`` are skipped, so the grid resumes after
an interruption instead of starting again.
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
DEMOS = os.path.join("demonstrations", "expert_low.npz")


def job_rl(level: str, seed: int, steps: int, hidden: int) -> Dict:
    out = os.path.join(RUNS, "sac_{}_s{}".format(level, seed))
    return {
        "name": "sac_{}_s{}".format(level, seed),
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", level, "--hidden", str(hidden),
            "--eval-every", "10000", "--eval-episodes", "30", "--quiet",
            "--output", out,
        ],
    }


def job_bc(seed: int, hidden: int) -> Dict:
    out = os.path.join(RUNS, "bc_s{}".format(seed))
    return {
        "name": "bc_s{}".format(seed),
        "output": out,
        "cmd": [
            sys.executable, "src/train_il.py", "--demos", DEMOS, "--seed", str(seed),
            "--epochs", "60", "--randomisation", "low", "--hidden", str(hidden),
            "--eval-episodes", "50", "--quiet", "--output", out,
        ],
    }


def job_dagger(seed: int, hidden: int) -> Dict:
    out = os.path.join(RUNS, "dagger_s{}".format(seed))
    return {
        "name": "dagger_s{}".format(seed),
        "output": out,
        "cmd": [
            sys.executable, "src/train_il.py", "--demos", DEMOS, "--seed", str(seed),
            "--epochs", "60", "--dagger", "--dagger-rounds", "5",
            "--randomisation", "shifted", "--hidden", str(hidden),
            "--eval-episodes", "50", "--quiet", "--output", out,
        ],
    }


def job_bcrl(level: str, seed: int, steps: int, hidden: int) -> Dict:
    out = os.path.join(RUNS, "bcrl_{}_s{}".format(level, seed))
    return {
        "name": "bcrl_{}_s{}".format(level, seed),
        "output": out,
        "needs": os.path.join(RUNS, "bc_s{}".format(seed), "policy.pt"),
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", level, "--hidden", str(hidden),
            "--eval-every", "10000", "--eval-episodes", "30", "--quiet",
            "--demos", DEMOS, "--demo-fraction", "0.25",
            # A large coefficient, deliberately: see docs/imitation.md. The BC
            # term is an MSE of order 0.01 competing with a normalised Q term,
            # and anything below about 20 lets the first few thousand actor
            # updates undo the clone before the critic is worth listening to.
            "--bc-coef", "50.0", "--bc-decay-steps", str(steps // 2),
            "--critic-warmup", "3000",
            "--target-entropy-scale", "2.0", "--init-alpha", "0.02",
            "--init-actor", os.path.join(RUNS, "bc_s{}".format(seed), "policy.pt"),
            "--output", out,
        ],
    }


def already_done(job: Dict) -> bool:
    return os.path.exists(os.path.join(job["output"], "result.json"))


LIVE_SECONDS = 15 * 60


def in_flight(job: Dict, now: float) -> bool:
    """True if another process looks to be running this job right now.

    ``result.json`` only appears when a run finishes, so it cannot distinguish
    "not started" from "started ten minutes ago by the other worker pool". A
    live run touches ``progress.csv`` at every evaluation, a few minutes apart,
    so a recently modified output directory means hands off. Without this,
    restarting the driver while runs are in flight gives two processes writing
    one directory, and a progress.csv that interleaves two training curves.
    """
    path = os.path.join(job["output"], "progress.csv")
    if not os.path.exists(path):
        return False
    return (now - os.path.getmtime(path)) < LIVE_SECONDS


def run_batch(jobs: List[Dict], parallel: int, dry_run: bool) -> None:
    os.makedirs(LOGS, exist_ok=True)
    now = time.time()
    pending = [j for j in jobs if not already_done(j) and not in_flight(j, now)]
    skipped = len(jobs) - len(pending)
    if skipped:
        print("skipping {} finished run(s)".format(skipped))
    if dry_run:
        for job in pending:
            print(" ".join(job["cmd"]))
        return

    running: List[Dict] = []
    queue = list(pending)
    t0 = time.time()
    while queue or running:
        while queue and len(running) < parallel:
            job = queue.pop(0)
            # Re-check at dequeue rather than trusting the list built at the
            # start. A second worker pool (or a previous invocation still
            # finishing) may have completed this run in the meantime, and two
            # processes writing the same output directory produce a run whose
            # progress.csv is a mixture of both -- which is not obvious from
            # the file and quietly corrupts a training curve.
            if already_done(job):
                print("[{:>6.0f}s] skip {} (finished elsewhere)".format(
                    time.time() - t0, job["name"]), flush=True)
                continue
            if in_flight(job, time.time()):
                print("[{:>6.0f}s] skip {} (another process is running it)".format(
                    time.time() - t0, job["name"]), flush=True)
                continue
            log_path = os.path.join(LOGS, job["name"] + ".log")
            log = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            running.append({"job": job, "proc": proc, "log": log, "start": time.time()})
            print("[{:>6.0f}s] start {}".format(time.time() - t0, job["name"]), flush=True)

        time.sleep(2.0)
        for entry in list(running):
            code = entry["proc"].poll()
            if code is None:
                continue
            entry["log"].close()
            running.remove(entry)
            status = "done" if code == 0 else "FAILED ({})".format(code)
            print("[{:>6.0f}s] {} {} after {:.0f}s".format(
                time.time() - t0, status, entry["job"]["name"],
                time.time() - entry["start"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--levels", nargs="+", default=["none", "low", "medium", "high"])
    parser.add_argument("--bcrl-levels", nargs="+", default=["medium"],
                        help="randomisation levels to run the imitation-plus-RL variant at")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="stages to skip: rl bc dagger bcrl aggregate")
    args = parser.parse_args()

    os.chdir(REPO)
    if not os.path.exists(DEMOS):
        print("recording demonstrations first")
        subprocess.check_call([
            sys.executable, "src/record_demos.py", "--episodes", "200",
            "--randomisation", "low", "--seed", "7", "--output", DEMOS,
        ])

    # Stage 1: imitation is cheap and the imitation-plus-RL runs depend on it.
    if "bc" not in args.skip:
        print("\n=== behaviour cloning ===")
        run_batch([job_bc(s, args.hidden) for s in args.seeds], args.jobs, args.dry_run)
    if "dagger" not in args.skip:
        print("\n=== dagger ===")
        run_batch([job_dagger(s, args.hidden) for s in args.seeds], args.jobs, args.dry_run)

    # Stage 2: the expensive reinforcement-learning grid.
    rl_jobs: List[Dict] = []
    if "rl" not in args.skip:
        rl_jobs += [job_rl(lvl, s, args.steps, args.hidden)
                    for lvl in args.levels for s in args.seeds]
    if "bcrl" not in args.skip:
        rl_jobs += [job_bcrl(lvl, s, args.steps, args.hidden)
                    for lvl in args.bcrl_levels for s in args.seeds]
    if rl_jobs:
        print("\n=== reinforcement learning ({} runs) ===".format(len(rl_jobs)))
        run_batch(rl_jobs, args.jobs, args.dry_run)

    if args.dry_run or "aggregate" in args.skip:
        return

    print("\n=== aggregating ===")
    stages = [
        ["experiments/expert_baseline.py"],
        ["experiments/ablation.py", "--prefix", "sac",
         "--output", os.path.join("experiments", "results", "ablation.json")],
        ["experiments/ablation.py", "--prefix", "bcrl",
         "--output", os.path.join("experiments", "results", "ablation_bcrl.json")],
        ["experiments/summarise.py"],
        ["experiments/bc_data_efficiency.py"],
    ]
    for stage in stages:
        print("--- " + " ".join(stage))
        subprocess.call([sys.executable] + stage)
    subprocess.call([sys.executable, "analysis/plots.py", "--all"])
    subprocess.call([sys.executable, "analysis/readme_tables.py"])

    summary_path = os.path.join("experiments", "results", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            print(json.dumps(json.load(fh).get("headline", {}), indent=2))


if __name__ == "__main__":
    main()
