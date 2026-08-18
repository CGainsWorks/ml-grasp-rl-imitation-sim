"""Can Isaac be made to hold what MuJoCo holds? The sweep after the diagnosis.

    C:\\isaac\\venv311\\Scripts\\python scripts/isaac_contact_match.py

`scripts/contact_probe.py` ran one protocol on both engines and found the
mechanism behind poor cross-simulator transfer: at the same commanded grip, the
same box, and the same mass, MuJoCo lifts it 117.9 mm and keeps hold, while
Isaac loses 16.5 mm and drops it. Everything cheaper was already excluded --
not control gain, not grip force, not friction of the *table*, not vertical
positioning.

"The contact model" is a diagnosis, not a fix. This asks the next question: is
there a setting of Isaac's contact parameters at which the same grip holds? That
is worth knowing either way.

* If some combination reproduces MuJoCo's fingerprint, the transfer gap is a
  *tuning* difference and the two simulators can be brought into agreement. The
  honest way to report that is as a matched pair of configurations, not as
  "Isaac was wrong".
* If nothing in the sweep holds the box, the difference is structural -- the
  solvers disagree about what a rigid pinch grasp is -- and no amount of
  parameter matching closes it. That is the more useful negative, because it
  says a policy has to be trained where it will run.

Three knobs, chosen because they are the ones that plausibly govern whether a
pinch slips:

``friction``    the object's static and dynamic friction. MuJoCo's box uses 1.0
                and Isaac's default here is also 1.0, so this sweeps around it
                rather than away from it.
``iterations``  PhysX position-solver iterations on the object. A pinch grasp is
                a two-contact equilibrium and under-solved contacts drift.
``rest``        the collision rest offset, which sets how far apart PhysX keeps
                surfaces. A large rest offset makes a pad "touch" without
                pressing, which would look exactly like the observed slip.

The protocol is `contact_probe.py`'s, unchanged and deliberately so: close at
full command, settle, then raise the hand at a fixed rate and record how much
height the object gained and whether it is still gripped. Comparing against a
different protocol would produce a number that could not be put next to
MuJoCo's 117.9 mm.

This is a sweep over one object at one mass with one grip command. It can show
that a setting holds; it cannot show that the two engines now agree in general.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frictions", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0])
parser.add_argument("--iterations", type=int, nargs="+", default=[4, 16, 64])
parser.add_argument("--rest-offsets", type=float, nargs="+",
                    default=[0.0, 0.001])
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--settle-steps", type=int, default=40)
parser.add_argument("--lift-steps", type=int, default=30)
parser.add_argument("--output",
                    default="experiments/results/isaac_contact_match.json")
args = parser.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MuJoCo's row from contact_probe.py, the target this is trying to reproduce.
MUJOCO_LIFT_MM = 117.9
MUJOCO_HELD = True

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import torch  # noqa: E402

sys.path.insert(0, REPO)

from envs.isaac.grasp_task import ACT_DIM, GraspTask, GraspTaskCfg  # noqa: E402
from src.policies.scripted_expert import ScriptedExpert  # noqa: E402


def measure(friction: float, iterations: int, rest: float) -> dict:
    """One cell of the sweep, on contact_probe.py's protocol exactly."""
    cfg = GraspTaskCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.randomisation_level = "none"
    # The object config hangs off the task config, not off the scene:
    # GraspTask builds it with RigidObject(self.cfg.obj). Reaching through
    # cfg.scene silently finds nothing, which is how the first run of this
    # sweep produced eighteen failed cells instead of eighteen measurements.
    cfg.obj.spawn.physics_material.static_friction = friction
    cfg.obj.spawn.physics_material.dynamic_friction = friction
    cfg.obj.spawn.rigid_props.solver_position_iteration_count = iterations
    cfg.obj.spawn.collision_props.rest_offset = rest

    env = GraspTask(cfg)
    obs_dict, _ = env.reset()

    # Drive down with the scripted expert, as the probe does: Isaac's hand is on
    # an arm and cannot be teleported the way the welded one can.
    import numpy as np

    experts = [ScriptedExpert() for _ in range(env.num_envs)]
    for e in experts:
        e.reset()
    for _ in range(45):
        obs_np = obs_dict["policy"].cpu().numpy()
        act = np.stack([e.act(obs_np[i]) for i, e in enumerate(experts)])
        obs_dict, _, _, _, _ = env.step(torch.as_tensor(act, device=env.device))

    close = torch.zeros((env.num_envs, ACT_DIM), device=env.device)
    close[:, 3] = 1.0
    for _ in range(args.settle_steps):
        obs_dict, _, _, _, _ = env.step(close)

    start_z = env._object_pos()[:, 2].clone()
    up = close.clone()
    up[:, 2] = 1.0
    for _ in range(args.lift_steps):
        obs_dict, _, _, _, _ = env.step(up)

    lift_mm = float((env._object_pos()[:, 2] - start_z).mean()) * 1e3
    held = float((env._grasped() > 0.5).float().mean())
    env.close()

    row = {"friction": friction, "solver_iterations": iterations,
           "rest_offset": rest, "lift_gained_mm": lift_mm,
           "fraction_still_held": held,
           "matches_mujoco": bool(held > 0.5 and lift_mm > 0.5 * MUJOCO_LIFT_MM)}
    print("friction {:.1f}  iters {:3d}  rest {:.4f}  ->  lift {:+7.1f} mm  "
          "held {:.2f}  {}".format(friction, iterations, rest, lift_mm, held,
                                   "MATCH" if row["matches_mujoco"] else ""),
          flush=True)
    return row


rows = []
for friction, iterations, rest in itertools.product(
        args.frictions, args.iterations, args.rest_offsets):
    try:
        rows.append(measure(friction, iterations, rest))
    except Exception as exc:  # a cell that will not build is data, not a crash
        print("friction {:.1f} iters {} rest {}: FAILED {}".format(
            friction, iterations, rest, exc), flush=True)
        rows.append({"friction": friction, "solver_iterations": iterations,
                     "rest_offset": rest, "error": str(exc)})

matches = [r for r in rows if r.get("matches_mujoco")]
best = max((r for r in rows if "lift_gained_mm" in r),
           key=lambda r: r["lift_gained_mm"], default=None)

os.makedirs(os.path.join(REPO, os.path.dirname(args.output)), exist_ok=True)
with open(os.path.join(REPO, args.output), "w", encoding="utf-8") as fh:
    json.dump({
        "protocol": "identical to scripts/contact_probe.py: close at full "
                    "command, settle, raise at a fixed rate, record height "
                    "gained and whether still gripped",
        "mujoco_reference": {"lift_gained_mm": MUJOCO_LIFT_MM,
                             "still_held": MUJOCO_HELD},
        "note": "one object, one mass, one grip command. A cell that matches "
                "shows the gap is tunable at this operating point; it does not "
                "show the two engines agree in general",
        "rows": rows,
        "n_matching": len(matches),
        "best_lift": best,
    }, fh, indent=2)

print("\n{} of {} cells hold the box; MuJoCo reference is {:+.1f} mm, held".format(
    len(matches), len(rows), MUJOCO_LIFT_MM))
if best is not None:
    print("best Isaac cell: {:+.1f} mm at friction {}, iters {}, rest {}".format(
        best["lift_gained_mm"], best["friction"], best["solver_iterations"],
        best["rest_offset"]))
_app.close()
