"""Learn the place task backwards, since forwards does not work.

    python experiments/place_curriculum.py --jobs 5

Seven reward designs, a tripled budget and a travel-ladder decomposition all put
from-scratch RL at zero on pick-and-place, and the diagnosis was that shaping of
this kind buys *segments* and does not chain them. The literature's answer to
exactly that is a reverse curriculum -- learn the last segment first from states
near the end, then move the start backwards as it succeeds (Florensa et al.,
[Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300),
2017).

`GraspEnv(start_progress=p)` sets the world into a partly-finished episode: at
1.0 the object is already in the closed gripper at carry height directly over
the target and only the lowering and release remain; at 0.0 it is the ordinary
task. Five stages of 40 000 steps each, 1.0 down to 0.0, **200 000 steps in
total** -- matched to every from-scratch arm it is being compared against, so
the curriculum is not being handed a bigger budget as well.

Each stage starts from the previous stage's checkpoint **including the critic**,
which is the point: what a stage learns about the value of the later segments
lives in the critic, and carrying only the actor across would make every stage
re-derive it.

Result: it degrades gracefully from 0.888 down to 0.140 and then scores exactly
0.000 the moment the fingers start open. The obvious reading is stagewise
forgetting, and it is wrong -- removing staging entirely by sampling the start
per episode (`--place-start-range 0 1`) gives one policy that scores 0.750 handed
the object lifted, 0.367 handed it grasped, and 0.000 when it has to close the
fingers itself. See docs/limitations.md.

What this costs, stated plainly, because it is not free and it is not "no
supervision": it uses the simulator's ability to be *set* into a mid-task state,
which no real robot has, and it uses task knowledge to construct that state. It
uses no expert *action*, so it is a genuinely different resource from the
demonstrations -- and the honest comparison is three-way, against both the
from-scratch arm and the demonstration-seeded one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.place_task import (  # noqa: E402
    ALPHA_FLOOR,
    HIDDEN,
    LEVEL,
    REPO,
    RUNS,
    run_batch,
)

# Eight stages over the whole task, 25 000 steps each, 200 000 in total.
#
# The first version used five stages from 1.0 to 0.0 and every one of them above
# 0.25 began with the object already grasped and airborne. It reached 0.594 at
# progress 0.25 and then scored exactly 0.000 at 0.0, because the last step asked
# the policy to learn reaching, grasping and lifting at once from a critic that
# had never seen the object on the table. Four easy stages and then a cliff is
# not a curriculum. These stages cross the grasp (0.25) and the lift (0.40)
# rather than jumping over them.
STAGES = [1.0, 0.85, 0.70, 0.55, 0.42, 0.30, 0.15, 0.0]


def job(seed: int, stage: int, steps: int) -> Dict:
    out = os.path.join(RUNS, "place_curr{}_s{}".format(stage, seed))
    cmd = [sys.executable, "src/train_rl.py", "--steps", str(steps),
           "--seed", str(seed + 100 * stage), "--randomisation", LEVEL,
           "--task", "place", "--place-start-progress", str(STAGES[stage]),
           "--hidden", str(HIDDEN), "--eval-every", str(max(1, steps // 2)),
           "--eval-episodes", "30", "--quiet",
           "--alpha-floor", str(ALPHA_FLOOR), "--output", out]
    spec = {"name": os.path.basename(out), "output": out, "cmd": cmd}
    if stage > 0:
        prev = os.path.join(RUNS, "place_curr{}_s{}".format(stage - 1, seed),
                            "policy.pt")
        spec["needs"] = prev
        cmd += ["--init-actor", prev, "--init-critic"]
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stage-steps", type=int, default=25_000)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    os.chdir(REPO)

    # Stage by stage, not seed by seed: every stage depends on the one before it.
    for stage in range(len(STAGES)):
        jobs: List[Dict] = [job(s, stage, args.stage_steps) for s in args.seeds]
        print("stage {} of {}, start_progress {}".format(
            stage + 1, len(STAGES), STAGES[stage]), flush=True)
        run_batch(jobs, args.jobs)

    final = "place_curr{}".format(len(STAGES) - 1)
    out = os.path.join("experiments", "results", "place_curriculum_eval.json")
    print("evaluating " + final, flush=True)
    subprocess.run([sys.executable, "src/evaluate.py", "--runs",
                    "experiments/runs/{}_s*".format(final), "--task", "place",
                    "--eval-levels", "none", "shifted", "--episodes",
                    str(args.episodes), "--label", "place_curriculum",
                    "--output", out], cwd=REPO, check=False)

    summary = os.path.join("experiments", "results", "place_curriculum.json")
    blob = {"stages": STAGES, "stage_steps": args.stage_steps,
            "total_steps": args.stage_steps * len(STAGES),
            "note": "matched to the from-scratch arms at 200 000 steps in "
                    "total, and it uses simulator state resets rather than "
                    "expert actions -- a different resource from the "
                    "demonstrations, not an absence of one"}
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8") as fh:
            blob["final"] = json.load(fh)
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("wrote " + summary)


if __name__ == "__main__":
    main()
