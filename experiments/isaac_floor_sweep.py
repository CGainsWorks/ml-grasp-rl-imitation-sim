"""Is 0.15 the wrong floor for Isaac, or is the floor wrong for Isaac?

    C:\\isaac\\venv311\\Scripts\\python.exe experiments/isaac_floor_sweep.py

`experiments/results/isaac_floor.json` says a floor of 0.15 raises the mean here
from 0.194 to 0.463 across five seeds and is nowhere near separated (t = 1.01).
Two explanations were left open, and one of them is testable in a night.

In MuJoCo the useful floor is different for every randomisation level, and a
value tuned on one distribution takes another to zero: `none` needs at least
0.10, `low` needs at most 0.05, and 0.15 -- the value carried into Isaac -- is
the value that kills `low`. Isaac is a different contact model, a different arm
and a different action scaling, which is a larger move than any randomisation
level. So the obvious reading of the flat Isaac result is that 0.15 is simply
the wrong number here.

This sweeps the two values either side of it, three seeds each, against the two
already on disk:

    floor 0.00   isaacfloor_scratch_s*   (the control, five seeds)
    floor 0.05   trained here
    floor 0.15   isaacfloor_floor_s*     (five seeds)
    floor 0.30   trained here

Three seeds for the new values rather than five: each run is about forty
minutes of GPU and cannot be parallelised, so six runs is already four hours.
If a value works, three seeds will show it against a control that scores 0.194.

The alternative explanation -- that the collapse in Isaac is a different
mechanism that happens to look the same -- is not tested here and does not
become true by elimination if this sweep finds nothing.
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

EXISTING = {
    0.0: "isaacfloor_scratch_s{}",
    0.15: "isaacfloor_floor_s{}",
}


def run_dir(floor: float, seed: int) -> str:
    if floor in EXISTING:
        return os.path.join(RUNS, EXISTING[floor].format(seed))
    return os.path.join(RUNS, "isaacsweep_{:03d}_s{}".format(int(round(floor * 100)), seed))


def job(floor: float, seed: int, steps: int, num_envs: int) -> Dict:
    out = run_dir(floor, seed)
    return {
        "name": os.path.basename(out),
        "output": out,
        "cmd": [
            sys.executable, "scripts/isaac_train.py",
            "--num-envs", str(num_envs), "--steps", str(steps),
            "--eval-every", str(max(1, steps // 10)), "--eval-episodes", "1",
            "--randomisation", "none", "--seed", str(seed),
            "--alpha-floor", str(floor), "--output", out,
        ],
    }


def read(run: str) -> Optional[float]:
    path = os.path.join(run, "result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["final_success_rate"]


def summarise(floors: List[float], seeds: List[int], output: str) -> None:
    sys.path.insert(0, REPO)
    from src.utils.stats import t_interval

    rows = []
    for floor in floors:
        values = [v for v in (read(run_dir(floor, s)) for s in seeds) if v is not None]
        if not values:
            continue
        interval = t_interval(values)
        rows.append({
            "alpha_floor": floor,
            "per_seed": values,
            **interval.as_dict(),
        })
        print("floor {:<5.2f} {:.3f} [{:.3f}, {:.3f}]  n={}  {}".format(
            floor, interval.point, interval.low, interval.high, len(values),
            [round(v, 3) for v in values]), flush=True)

    blob = {
        "question": "is 0.15 the wrong entropy floor for Isaac?",
        "simulator": "Isaac Sim 5.1.0 / Isaac Lab 2.3.2",
        "steps": 15_000,
        "num_envs": 32,
        "randomisation": "none",
        "seeds": seeds,
        "rows": rows,
        "note": "floor 0.00 is experiments/runs/isaacfloor_scratch_s* and floor "
                "0.15 is isaacfloor_floor_s*, both five seeds; the new values are "
                "three seeds each. Compare down the column, and against the "
                "MuJoCo matrix in experiments/results/floor_by_level.json.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.05, 0.15, 0.30])
    parser.add_argument("--train", type=float, nargs="+", default=[0.05, 0.30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "isaac_floor_sweep.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)

    if not args.summarise_only:
        os.makedirs(LOGS, exist_ok=True)
        lock = os.path.join(LOGS, ".isaac_sweep.lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print("another driver holds {} -- refusing to start a second simulator "
                  "against the same run directories.".format(lock))
            return
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        # Interleaved by seed, so an interrupted sweep leaves both values at the
        # same number of seeds rather than one complete and one empty.
        jobs = [job(floor, seed, args.steps, args.num_envs)
                for seed in args.train_seeds for floor in args.train]
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

    summarise(args.floors, args.seeds, args.output)


if __name__ == "__main__":
    main()
