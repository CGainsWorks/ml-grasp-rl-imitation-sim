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
parser.add_argument("--contact-offsets", type=float, nargs="+", default=[None],
                    help="PhysX contact offset on the object, metres. The "
                         "distance at which contacts start being generated")
parser.add_argument("--velocity-iterations", type=int, nargs="+",
                    default=[None],
                    help="PhysX velocity-solver iterations on the object. "
                         "Position iterations fix penetration; these fix "
                         "sliding, which is what a slipping pinch looks like")
parser.add_argument("--grip-stiffness", type=float, nargs="+", default=[None],
                    help="scale on the finger drive stiffness. A pinch that "
                         "slips because the fingers give way is a drive "
                         "problem rather than a contact one")
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


def measure(friction: float, iterations: int, rest: float,
            contact_offset=None, velocity_iterations=None,
            grip_stiffness=None) -> dict:
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
    if contact_offset is not None:
        cfg.obj.spawn.collision_props.contact_offset = contact_offset
    if velocity_iterations is not None:
        cfg.obj.spawn.rigid_props.solver_velocity_iteration_count = (
            velocity_iterations)
    if grip_stiffness is not None:
        # The *finger* drive only. The Franka config has three actuator groups
        # -- panda_shoulder, panda_forearm, panda_hand -- and the first version
        # of this scaled all of them, which detuned the arm at 5x and 20x so
        # thoroughly that the box was never touched. Both cells reported
        # "+0.0 mm, held 0.00", which reads in a table exactly like a grip that
        # slipped and is nothing of the kind.
        hand = cfg.robot.actuators["panda_hand"]
        hand.stiffness = hand.stiffness * grip_stiffness

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
    # Did the fingers ever actually touch the box? Without this a cell that
    # never made contact and a cell whose grip slipped both report
    # "lift ~0, held 0.00", which are entirely different failures. Two cells in
    # this sweep were misread that way before it was recorded.
    ever_touched = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for _ in range(args.settle_steps):
        obs_dict, _, _, _, _ = env.step(close)
        ever_touched |= env._grasped() > 0.5

    start_z = env._object_pos()[:, 2].clone()
    up = close.clone()
    up[:, 2] = 1.0
    for _ in range(args.lift_steps):
        obs_dict, _, _, _, _ = env.step(up)
        ever_touched |= env._grasped() > 0.5

    lift_mm = float((env._object_pos()[:, 2] - start_z).mean()) * 1e3
    held = float((env._grasped() > 0.5).float().mean())
    env.close()

    touched = float(ever_touched.float().mean())
    row = {"friction": friction, "solver_iterations": iterations,
           "rest_offset": rest, "lift_gained_mm": lift_mm,
           "fraction_still_held": held, "fraction_ever_touched": touched,
           "interpretable": bool(touched > 0.5),
           "matches_mujoco": bool(held > 0.5 and lift_mm > 0.5 * MUJOCO_LIFT_MM)}
    print("friction {:.1f}  iters {:3d}  rest {:.4f}  ->  lift {:+7.1f} mm  "
          "held {:.2f}  touched {:.2f}  {}{}".format(
              friction, iterations, rest, lift_mm, held, touched,
              "MATCH" if row["matches_mujoco"] else "",
              "" if row["interpretable"]
              else "  NO CONTACT -- not a grip failure"),
          flush=True)
    return row


rows = []
for friction, iterations, rest, coff, viters, gstiff in itertools.product(
        args.frictions, args.iterations, args.rest_offsets,
        args.contact_offsets, args.velocity_iterations, args.grip_stiffness):
    try:
        row = measure(friction, iterations, rest, coff, viters, gstiff)
        row.update({"contact_offset": coff, "velocity_iterations": viters,
                    "grip_stiffness": gstiff})
        rows.append(row)
    except Exception as exc:  # a cell that will not build is data, not a crash
        print("friction {:.1f} iters {} rest {}: FAILED {}".format(
            friction, iterations, rest, exc), flush=True)
        rows.append({"friction": friction, "solver_iterations": iterations,
                     "rest_offset": rest, "contact_offset": coff,
                     "velocity_iterations": viters, "grip_stiffness": gstiff,
                     "error": str(exc)})

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
