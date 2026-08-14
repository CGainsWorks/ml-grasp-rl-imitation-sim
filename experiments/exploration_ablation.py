"""Why do three seeds in five stall, and does anything fix it?

    python experiments/exploration_ablation.py --jobs 6

The finding this investigates is in ``docs/results.md``: SAC from scratch solves
the nominal task on two of five seeds and stalls on the other three, and the
stalled runs sit at an entropy coefficient around 0.025 while the successful
ones sit near 0.17. The stalled policies grasp the box reliably and hold it on
the table, which the reward pays 0.73 per step for against 9.75 at the hold
point. Isaac reproduces the same basin, so it is a property of the task rather
than of MuJoCo.

Two hypotheses, each with a literature-backed remedy, tested one at a time on
exactly the seeds that stalled:

``alpha-floor``
    *The policy stopped exploring.* Automatic entropy tuning drives the
    coefficient towards zero once a policy is confident, which is fatal when it
    is confident about a local optimum. Clamping it from below is the standard
    shape of the fix. A weaker version of this -- raising the target entropy --
    was tried first and failed, which is what motivates the harder floor.

``pink``
    *The missing behaviour is temporally extended.* Lifting is roughly twenty
    consecutive upward commands. SAC explores with white noise, whose
    independent samples average out to almost no net displacement, so the
    behaviour is essentially never sampled. Eberhard et al., "Pink Noise Is All
    You Need" (ICLR 2023), find pink noise beats white and OU across
    continuous control and recommend it as the default.
    https://openreview.net/forum?id=hQ9V5QN27eS

``both``
    Run only if the single-variable arms disagree or both help partially.

The baseline is not rerun: ``experiments/runs/sac_none_s{2,3,4}`` already have
it at twice this budget, and all three scored 0.000.
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
    "alpha-floor": ["--alpha-floor", "0.15"],
    "pink": ["--exploration", "pink"],
    "both": ["--alpha-floor", "0.15", "--exploration", "pink"],
    # Sensitivity sweep over the floor value. "alpha-floor" above is 0.15, so
    # that column of the sweep is already run and is not repeated here.
    "floor005": ["--alpha-floor", "0.05"],
    "floor010": ["--alpha-floor", "0.10"],
    "floor030": ["--alpha-floor", "0.30"],
    "floor050": ["--alpha-floor", "0.50"],
}


def job(arm: str, seed: int, steps: int) -> Dict:
    name = "explore_{}_s{}".format(arm.replace("-", ""), seed)
    out = os.path.join(RUNS, name)
    return {
        "name": name,
        "output": out,
        "cmd": [
            sys.executable, "src/train_rl.py", "--steps", str(steps),
            "--seed", str(seed), "--randomisation", "none", "--hidden", "128",
            "--eval-every", "10000", "--eval-episodes", "30", "--quiet",
            *ARMS[arm], "--output", out,
        ],
    }


def run_batch(jobs: List[Dict], parallel: int) -> None:
    os.makedirs(LOGS, exist_ok=True)
    queue = [j for j in jobs if not os.path.exists(os.path.join(j["output"], "result.json"))]
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
    rows = []
    for arm in ["baseline"] + arms:
        per_seed = []
        for seed in seeds:
            if arm == "baseline":
                path = os.path.join(RUNS, "sac_none_s{}".format(seed), "result.json")
            else:
                path = os.path.join(
                    RUNS, "explore_{}_s{}".format(arm.replace("-", ""), seed), "result.json"
                )
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                per_seed.append(json.load(fh))
        if not per_seed:
            continue
        finals = [r["final_success_rate"] for r in per_seed]
        bests = [r["best_success_rate"] for r in per_seed]
        rows.append({
            "arm": arm,
            "seeds": seeds[: len(per_seed)],
            "final_per_seed": finals,
            "best_per_seed": bests,
            "mean_final": sum(finals) / len(finals),
            "solved": sum(1 for f in finals if f > 0.5),
        })
        print("{:<12s} final {}  mean {:.3f}  solved {}/{}".format(
            arm, [round(f, 3) for f in finals], rows[-1]["mean_final"],
            rows[-1]["solved"], len(finals)), flush=True)

    blob = {
        "question": "does anything rescue the seeds that stall in the "
                    "grasp-and-hold-on-the-table optimum?",
        "seeds": seeds,
        "arms": rows,
        "baseline_note": "baseline is experiments/runs/sac_none_s* at 200k steps, "
                         "twice the budget of the other arms",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["alpha-floor", "pink", "both"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--output",
                        default=os.path.join("experiments", "results", "exploration.json"))
    args = parser.parse_args()

    os.chdir(REPO)
    if not args.summarise_only:
        jobs = [job(arm, seed, args.steps) for arm in args.arms for seed in args.seeds]
        print("=== {} runs ===".format(len(jobs)))
        run_batch(jobs, args.jobs)
    summarise(args.arms, args.seeds, args.output)


if __name__ == "__main__":
    main()
