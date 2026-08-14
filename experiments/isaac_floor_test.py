"""Does the entropy floor rescue Isaac too, or was it a MuJoCo artefact?

    C:\\isaac\\venv311\\Scripts\\python.exe experiments/isaac_floor_test.py

The collapse reproduced in both engines: three of five MuJoCo seeds and *all*
five Isaac seeds settle into grasping the box and holding it on the table, which
the reward pays 0.73 per step for against 9.75 at the hold point. That symmetry
is the argument in ``docs/results.md`` that the trap belongs to the task rather
than to MuJoCo.

The fix has only ever been measured in MuJoCo, so the symmetry is untested where
it matters most. If clamping the entropy coefficient from below is a property of
the task, it should work here as well. If it does not, the shared collapse was a
coincidence of two different mechanisms and the MuJoCo explanation is weaker
than it currently reads.

Two arms, three seeds, 15 000 steps x 32 environments each:

``scratch``  the control, unchanged SAC
``floor``    the same, with ``--alpha-floor 0.15``

The budget is set by the control, not by convenience: ``isaac_sac_none`` already
ran from scratch for exactly 15 000 steps (480 000 transitions) and scored
0.000 with a grasp rate of 1.00, so this is the budget at which the failure is
already known to be stable rather than merely early. The control is rerun here
anyway, at settings identical to the floor arm, so the comparison does not rest
on a run configured for a different purpose.

Three seeds rather than five: each run is about forty minutes of GPU, runs
cannot be parallelised (a second environment in the same process hangs, and two
simulator instances do not fit on an 8 GB card), and six runs is already four
hours. Three is enough to separate 0.000 from anything, which is the question.

The lock file is not decoration. This script was launched twice by accident and
both copies ran for two hours, each spawning its own simulator against the same
output directory, so `progress.csv` ended up with every checkpoint written
twice. The metrics in the duplicate pairs are identical -- the runs are
deterministic given the seed, which is the evidence that they did not corrupt
each other -- but only the wall-clock column differed, and a results file that
has to be argued about is worse than one that cannot happen. ``summarise``
de-duplicates by step as it reads, and the lock stops a second driver starting.
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
    "scratch": [],
    "floor": ["--alpha-floor", "0.15"],
}


def job(arm: str, seed: int, steps: int, num_envs: int) -> Dict:
    name = "isaacfloor_{}_s{}".format(arm, seed)
    out = os.path.join(RUNS, name)
    return {
        "name": name,
        "output": out,
        "cmd": [
            sys.executable, "scripts/isaac_train.py",
            "--num-envs", str(num_envs), "--steps", str(steps),
            "--eval-every", str(max(1, steps // 10)), "--eval-episodes", "1",
            "--randomisation", "none", "--seed", str(seed),
            *ARMS[arm], "--output", out,
        ],
    }


def dedupe_progress(run: str) -> None:
    """Keep the first row per step. See the note about the duplicated launch."""
    path = os.path.join(run, "progress.csv")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines:
        return
    kept, seen = [lines[0]], set()
    for line in lines[1:]:
        step = line.split(",")[0]
        if step in seen:
            continue
        seen.add(step)
        kept.append(line)
    if len(kept) != len(lines):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(kept) + "\n")
        print("de-duplicated {} rows in {}".format(len(lines) - len(kept), path), flush=True)


def summarise(arms: List[str], seeds: List[int], output: str) -> None:
    from src.utils.stats import t_interval

    rows = []
    for arm in arms:
        finals, bests, alphas = [], [], []
        for seed in seeds:
            run = os.path.join(RUNS, "isaacfloor_{}_s{}".format(arm, seed))
            dedupe_progress(run)
            path = os.path.join(run, "result.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            finals.append(result["final_success_rate"])
            bests.append(result["best_success_rate"])
            progress = os.path.join(run, "progress.csv")
            if os.path.exists(progress):
                with open(progress, "r", encoding="utf-8") as fh:
                    last = fh.read().strip().splitlines()[-1].split(",")
                alphas.append(float(last[6]))
        if not finals:
            continue
        interval = t_interval(finals)
        rows.append({
            "arm": arm,
            "final_per_seed": finals,
            "best_per_seed": bests,
            "final_alpha_per_seed": alphas,
            "across_seeds": interval.as_dict(),
        })
        print("{:<8s} final {}  mean {:.3f}  95% t [{:.3f}, {:.3f}]  alpha {}".format(
            arm, [round(f, 3) for f in finals], interval.point,
            interval.low, interval.high, [round(a, 4) for a in alphas]), flush=True)

    blob = {
        "question": "does the entropy floor rescue the same local optimum in a "
                    "second simulator?",
        "simulator": "Isaac Sim 5.1.0 / Isaac Lab 2.3.2",
        "alpha_floor": 0.15,
        "seeds": seeds,
        "arms": rows,
        "note": "15 000 steps x 32 environments per run, nominal world. The "
                "control is the same budget at which experiments/runs/"
                "isaac_sac_none already scored 0.000 with grasp rate 1.00.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["scratch", "floor"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "isaac_floor.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)

    if not args.summarise_only:
        os.makedirs(LOGS, exist_ok=True)
        lock = os.path.join(LOGS, ".isaac_floor.lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print("another driver holds {} -- refusing to start a second "
                  "simulator against the same run directories. Delete the lock "
                  "if no driver is running.".format(lock))
            return
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        # Interleaved by seed, so a run that is cut short still leaves a
        # matched pair rather than one complete arm and one empty one.
        jobs = [job(arm, seed, args.steps, args.num_envs)
                for seed in args.seeds for arm in args.arms]
        t0 = time.time()
        for spec in jobs:
            if os.path.exists(os.path.join(spec["output"], "result.json")):
                print("skip {} (finished)".format(spec["name"]), flush=True)
                continue
            print("[{:>5.0f}s] start {}".format(time.time() - t0, spec["name"]), flush=True)
            with open(os.path.join(LOGS, spec["name"] + ".log"), "w", encoding="utf-8") as log:
                subprocess.call(spec["cmd"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            print("[{:>5.0f}s] done  {}".format(time.time() - t0, spec["name"]), flush=True)
        os.remove(lock)

    summarise(args.arms, args.seeds, args.output)


if __name__ == "__main__":
    main()
