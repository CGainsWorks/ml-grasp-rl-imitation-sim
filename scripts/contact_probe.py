"""How hard can each simulator's gripper hold? The contact-level comparison.

    python scripts/contact_probe.py                      # MuJoCo side
    C:\\isaac\\venv311\\Scripts\\python scripts/contact_probe.py --isaac

Cross-simulator transfer is poor and everything cheaper than a contact
measurement has been ruled out: not control gain (seven action scalings, both
directions), not grip force, not friction, and not vertical positioning -- adding
the missing 10 cm back makes it *worse*, while pressing down improves grip and
lift without improving success. `docs/limitations.md` says what is left is the
contact model. This measures it.

The protocol is deliberately crude and identical on both sides, because a
like-for-like number is worth more here than a sophisticated one:

1. place the object on the table and the hand around it, fingers open;
2. close the gripper at full command and let it settle;
3. record the pad penetration depth and the total normal force -- how deep the
   solver lets the pads sink, and how hard it pushes back;
4. raise the hand at a fixed rate and record the object's height after a fixed
   number of steps, plus whether it is still held.

Step 3 is the contact model's fingerprint. Step 4 is what a policy actually
depends on. If Isaac's pads sink deeper for less force, or the object slips
during the lift at a grip MuJoCo holds, the transfer failure has a mechanism and
it is not in the action space.

Both sides write to the same JSON so the two rows sit next to each other. This is
one object at one mass with one grip command: a fingerprint, not a sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--isaac", action="store_true", help="measure the Isaac side")
parser.add_argument("--settle-steps", type=int, default=40)
parser.add_argument("--lift-steps", type=int, default=30)
parser.add_argument("--output", default="experiments/results/contact_probe.json")
args = parser.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_row(row: dict) -> None:
    os.chdir(REPO)
    rows = []
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as fh:
            rows = json.load(fh).get("rows", [])
    rows = [r for r in rows if r["simulator"] != row["simulator"]]
    rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"protocol": "close at full command, settle, measure pad "
                               "penetration and normal force, then raise the "
                               "hand at a fixed rate and record height and grip",
                   "note": "one object, one mass, one grip command: a "
                           "fingerprint of the contact model, not a sweep",
                   "rows": sorted(rows, key=lambda r: r["simulator"])}, fh,
                  indent=2)
    print(json.dumps(row, indent=2))


if not args.isaac:
    sys.path.insert(0, REPO)
    import mujoco  # noqa: E402
    import numpy as np  # noqa: E402

    from envs.mujoco.grasp_env import make_env  # noqa: E402

    env = make_env("none", seed=0)
    obs, _ = env.reset(seed=1)
    obj = env._object_pos().copy()
    hadr = env._hand_qadr
    mujoco.mj_forward(env.model, env.data)
    offset = env._grip_pos() - env.data.qpos[hadr : hadr + 3]
    env.data.qpos[hadr : hadr + 3] = obj - offset
    env.data.mocap_pos[env._mocap_id] = obj - offset
    mujoco.mj_forward(env.model, env.data)

    lo, hi = env._grip_range
    for _ in range(args.settle_steps):
        env.data.ctrl[:] = hi
        for _ in range(env.n_substeps):
            mujoco.mj_step(env.model, env.data)

    depths, forces = [], []
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        if not ({c.geom1, c.geom2} & set(env._pad_gids)):
            continue
        if env._object_gid not in (c.geom1, c.geom2):
            continue
        depths.append(-float(c.dist))
        f = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, i, f)
        forces.append(abs(float(f[0])))

    start_z = float(env._object_pos()[2])
    for _ in range(args.lift_steps):
        env.data.mocap_pos[env._mocap_id][2] += 0.004
        for _ in range(env.n_substeps):
            mujoco.mj_step(env.model, env.data)
    row = {
        "simulator": "mujoco-3.11",
        "pad_contacts": len(depths),
        "mean_penetration_mm": (float(np.mean(depths)) * 1e3) if depths else 0.0,
        "total_normal_force_N": float(np.sum(forces)) if forces else 0.0,
        "object_mass_kg": float(env.model.body_mass[env._object_bid]),
        "lift_gained_mm": (float(env._object_pos()[2]) - start_z) * 1e3,
        "commanded_lift_mm": args.lift_steps * 4.0,
        "still_held": bool(env._grasped_raw()),
    }
    env.close()
    write_row(row)
else:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher  # noqa: E402

    _app = AppLauncher(headless=True).app

    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    sys.path.insert(0, REPO)
    from envs.isaac.grasp_task import ACT_DIM, GraspTask, GraspTaskCfg  # noqa: E402

    cfg = GraspTaskCfg()
    cfg.scene.num_envs = 8
    cfg.randomisation_level = "none"
    env = GraspTask(cfg)
    env.reset()

    # Drive the hand down onto the box with the scripted expert's own approach,
    # then close: Isaac's hand is on an arm, so it cannot simply be teleported
    # the way the welded one can.
    from src.policies.scripted_expert import ScriptedExpert  # noqa: E402

    obs_dict, _ = env.reset()
    experts = [ScriptedExpert() for _ in range(env.num_envs)]
    for _ in range(45):
        obs_np = obs_dict["policy"].cpu().numpy()
        act = np.stack([e.act(obs_np[i]) for i, e in enumerate(experts)])
        obs_dict, _, _, _, _ = env.step(torch.as_tensor(act, device=env.device))
    close = torch.zeros((env.num_envs, ACT_DIM), device=env.device)
    close[:, 3] = 1.0
    for _ in range(args.settle_steps):
        obs_dict, _, _, _, _ = env.step(close)

    forces = env._robot.root_physx_view.get_link_incoming_joint_force()
    grip_force = float(torch.linalg.norm(forces[:, -2:, :3], dim=-1).sum(-1).mean())
    start_z = env._object_pos()[:, 2].clone()
    up = close.clone()
    up[:, 2] = 1.0
    for _ in range(args.lift_steps):
        obs_dict, _, _, _, _ = env.step(up)
    row = {
        "simulator": "isaac-sim-5.1.0",
        "pad_contacts": None,
        "mean_penetration_mm": None,
        "total_normal_force_N": grip_force,
        "object_mass_kg": None,
        "lift_gained_mm": float((env._object_pos()[:, 2] - start_z).mean()) * 1e3,
        "commanded_lift_mm": args.lift_steps * 20.0,
        "still_held": bool(env._grasped().float().mean() > 0.5),
        "caveat": "Isaac reports joint forces rather than per-contact normals "
                  "here, so the force column is a finger-joint magnitude and is "
                  "NOT comparable in units to MuJoCo's contact normal. The lift "
                  "and grip columns are comparable; the force column is not, and "
                  "saying so is the point of writing it down.",
    }
    env.close()
    write_row(row)
    _app.close()
