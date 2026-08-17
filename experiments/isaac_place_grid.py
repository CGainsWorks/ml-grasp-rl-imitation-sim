"""The second task, trained in Isaac, against what MuJoCo already predicts.

    python experiments/isaac_place_grid.py

`docs/limitations.md` said the second task was a MuJoCo-only result. The port
closed the first half of that (same file, shared reward, parity to 7.3e-08 on
the GPU) and the retuned Franka expert closed the second, placing 23 of 24 at
`none`. What was left was compute, and this spends it.

**The prediction is on the record before the runs.** In MuJoCo, pick-and-place is
solved by cloning (0.978) and by demonstration-seeded RL (0.870-0.973), and
from-scratch SAC does not clear 0.05 under seven reward designs and a tripled
budget. If the port is faithful, Isaac should show the same *pattern*: the
demonstration arm works, the from-scratch arm does not. Matching absolute numbers
is not expected and would be suspicious -- different robot, different gripper,
different contact solver.

Two arms, three seeds each, identical budget:

``isaac_place_bcrl``  demonstrations pinned in the buffer, BC anchor held rather
                      than decayed (the decaying schedule collapses at the decay
                      point, which envs/isaac/README.md measures)
``isaac_place_sac``   from scratch, with the entropy floor Isaac's own sweep
                      chose, so the baseline is the fair one rather than the
                      collapsed one

Runs land in `experiments/runs/isaac_place_{bcrl,sac}_s<seed>`. Isaac holds the
GPU, so these run one at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join("experiments", "runs")
LOGS = os.path.join("experiments", "logs")
DEMOS = os.path.join("demonstrations", "isaac_place_none.npz")
# The value Isaac's own floor sweep chose. MuJoCo's 0.15 scores far worse here;
# the floor is distribution-specific and that is one of this repository's
# findings rather than an oversight.
ISAAC_FLOOR = "0.30"


def job(arm: str, seed: int, steps: int, num_envs: int, level: str) -> Dict:
    name = "isaac_place_{}_s{}".format(arm, seed)
    out = os.path.join(RUNS, name)
    cmd = [sys.executable, "scripts/isaac_train.py", "--task", "place",
           "--num-envs", str(num_envs), "--steps", str(steps),
           "--eval-every", str(max(1, steps // 4)), "--eval-episodes", "2",
           "--randomisation", level, "--seed", str(seed), "--output", out]
    if arm == "sac":
        cmd += ["--alpha-floor", ISAAC_FLOOR]
    else:
        cmd += ["--demos", DEMOS, "--bc-decay-steps", "1000000"]
    return {"name": name, "output": out, "cmd": cmd}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["bcrl", "sac"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--randomisation", default="none")
    args = parser.parse_args()
    os.chdir(REPO)
    os.makedirs(LOGS, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    jobs: List[Dict] = [job(a, s, args.steps, args.num_envs, args.randomisation)
                        for a in args.arms for s in args.seeds]
    for spec in jobs:
        if os.path.exists(os.path.join(spec["output"], "result.json")):
            print("reuse " + spec["name"], flush=True)
            continue
        print("start " + spec["name"], flush=True)
        with open(os.path.join(LOGS, spec["name"] + ".log"), "w",
                  encoding="utf-8") as log:
            subprocess.run(spec["cmd"], cwd=REPO, stdout=log,
                           stderr=subprocess.STDOUT, env=env, check=False)

    rows = {}
    for arm in args.arms:
        rates = []
        for seed in args.seeds:
            path = os.path.join(RUNS, "isaac_place_{}_s{}".format(arm, seed),
                                "result.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    rates.append(json.load(fh).get("final_success_rate"))
        rows[arm] = rates
        print("{:<6s} {}".format(arm, [round(r, 3) for r in rates if r is not None]),
              flush=True)

    out = os.path.join("experiments", "results", "isaac_place_grid.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"steps": args.steps, "num_envs": args.num_envs,
                   "randomisation": args.randomisation, "floor": ISAAC_FLOOR,
                   "prediction": "MuJoCo says the demonstration arm works and "
                                 "the from-scratch arm does not; matching "
                                 "absolute numbers across two robots would be "
                                 "suspicious rather than reassuring",
                   "results": rows}, fh, indent=2)
    print("wrote " + out)


if __name__ == "__main__":
    main()
