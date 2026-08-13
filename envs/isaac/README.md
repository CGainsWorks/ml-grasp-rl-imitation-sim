# Isaac Lab port

**Status: runs. Four of five bring-up checks pass. The fifth does not, and the
reason is understood and measured.**

Brought up against **Isaac Sim 5.1.0** and **Isaac Lab 2.3.2**, Python 3.11,
CUDA 12.8, on an RTX 4060. Reproduce with:

```bash
python scripts/isaac_bringup.py --num-envs 8 --episodes 2 --randomisation none
```

| # | Check | Result |
| --- | --- | --- |
| 1 | Environment constructs and resets, `num_envs` worlds | **pass** |
| 2 | Box stays on the table under zero actions | **pass** (8/8 envs at 0.422 m) |
| 3 | Observation layout matches the MuJoCo table, index by index | **pass** |
| 4 | Reward inside Isaac matches the shared numpy implementation | **pass** (max difference 5.1e-08) |
| 5 | Scripted expert grasps and lifts reliably | **fail** (see below) |

Check 4 is the one that matters most. The claim this port exists to support is
"it is the same task, not a similar one", and that claim is now measured inside
the running simulator rather than asserted: the reward computed on the GPU from
torch tensors agrees with the numpy implementation the MuJoCo environment uses
to eight decimal places, on the same states.

## What was wrong when this file was written blind

It was written against the documentation before Isaac Sim was installed, and
none of it survived contact. Recorded because the list is the honest content of
"ported but never run":

| Written | Actually required |
| --- | --- |
| Assets declared inside a custom `InteractiveSceneCfg` subclass | Assets are fields on the *env* config; the scene config only carries `num_envs`/`env_spacing` |
| No `_setup_scene` | Mandatory: instantiate the assets, register them on `self.scene`, clone the environments, add a light |
| `omniverse://localhost/NVIDIA/Assets/...` USD path | `FRANKA_PANDA_HIGH_PD_CFG` from `isaaclab_assets`; the high-PD variant, because the default gains track an IK target too softly to grasp |
| Grip point = `panda_hand` origin | The fingertips are 0.107 m along the hand's local z; without the offset the controller servos the wrist to the box and the fingers close 10 cm short |
| IK in relative *position* mode | Relative position leaves orientation to drift, and the Franka's home pose holds the gripper at 45°, which cannot do a top-down grasp. Absolute *pose* mode with a fixed downward quaternion reproduces the MuJoCo hand, whose orientation is pinned by its weld |
| Target recomputed from the measured pose each step | A persistent setpoint, like the MuJoCo mocap body. Recomputing from the measurement makes a zero action mean "stay wherever gravity has dragged me", and the arm sags out of the workspace in about two seconds |
| Setpoint floor at the table top | The floor has to sit *below* the table, because the achieved pose lags the setpoint (see below); clamping at the table top makes the fingertips unable to reach a box resting on it |
| Franka's shipped home pose | Puts the fingertips at z = 0.383, below the 0.40 m table top: the arm starts inside the table and flicks the box off it. `scripts/isaac_pregrasp.py` solves a pose that matches the reset setpoint |

## The open problem: standing IK error

The arm's implicit PD holds against gravity with a finite stiffness, so the
achieved grip point lags the commanded setpoint by a standing offset that grows
as the arm extends:

| Commanded setpoint z | Achieved grip z | Error |
| ---: | ---: | ---: |
| 0.60 | 0.673 | 0.073 |
| 0.52 | 0.635 | 0.115 |

At the lower setpoint the wrist also tilts off vertical (finger axis
`(-0.29, 0.01, -0.96)` instead of `(0, 0, -1)`).

The MuJoCo hand tracks its mocap setpoint to within a millimetre, so the
scripted expert — a state machine with 12 mm phase tolerances — assumes an
accurate servo. Against a Franka that lags by 70-115 mm, its DESCEND phase
times out and the gripper closes above the box.

This is not a tuning problem and loosening the expert's tolerances made it worse
(0/8 against 1/8). It needs a control path that actually reaches its setpoint:
gravity compensation, stiffer joint gains, or an operational-space controller
instead of position-mode IK. That is the next piece of work, and it is not done.

**The mechanism itself is fine.** With a favourable spawn the whole sequence
executes correctly — here is a trace of the gripper closing on the box and
lifting it to the hold point:

```
step ph  grip                  obj                   width
 56  1  [0.480 0.085 0.451]  [0.484 0.084 0.422]   0.080
 64  2  [0.487 0.084 0.422]  [0.484 0.084 0.422]   0.043   <- closed on the box
 72  3  [0.487 0.083 0.439]  [0.484 0.084 0.439]   0.043   <- lifting
 80  3  [0.487 0.084 0.506]  [0.484 0.084 0.507]   0.043
 88  3  [0.487 0.084 0.550]  [0.484 0.084 0.551]   0.043   <- at the hold point
```

## What is shared with the MuJoCo task, and what is not

Shared by import, not by copy — `tests/test_isaac_port.py` asserts this:

| Shared | File |
| --- | --- |
| Reward terms and weights | `src/rewards/grasp_reward.py` |
| Success and drop conditions | `src/rewards/grasp_reward.py` |
| Randomisation ranges per level | `src/randomisation/configs/*.json` |

Deliberately different:

| | MuJoCo | Isaac Lab |
| --- | --- | --- |
| Embodiment | free-floating parallel-jaw hand | Franka Panda |
| Cartesian control | mocap body plus a weld | differential IK, absolute pose mode |
| Vectorisation | one world | `num_envs` worlds, partial resets |
| Grasp test | contact list, both pads | geometric proximity plus finger closure |
| Control period | 0.04 s (20 substeps of 2 ms) | 0.04 s (decimation 8 of 5 ms) |

## Still not done

1. **Fix the standing IK error** — the blocker for check 5.
2. **Contact-sensor grasp test.** The geometric test reports a grasp for a pinch
   that is merely close. Add a `ContactSensor` per finger and read it, as the
   MuJoCo environment reads its contact list.
3. **Wire up the randomisation.** The ranges are loaded from the shared configs
   but are not yet applied through Isaac Lab's event manager, so every episode
   currently runs at nominal.
4. **Train something.** No policy has been trained in Isaac. Every learned
   number in this repository comes from MuJoCo.
5. **Cross-simulator evaluation.** Not attempted; the control paths differ, so a
   MuJoCo policy is not expected to transfer unchanged.

## Installing

```bash
python -m venv C:\isaac\venv311                      # Python 3.11 specifically
C:\isaac\venv311\Scripts\python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
C:\isaac\venv311\Scripts\python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
C:\isaac\venv311\Scripts\python -m pip install "setuptools<81"    # see below
C:\isaac\venv311\Scripts\python -m pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation
```

Two things that will bite:

* **`setuptools<81` and `--no-build-isolation`.** Isaac Lab pulls `flatdict`,
  whose `setup.py` imports `pkg_resources`, which setuptools 81+ no longer
  ships. Without the pin the install dies in metadata generation.
* **Disk.** About 15 GB, and it wants a fast disk: on a drive writing at
  0.26 MB/s the download alone projects to roughly 17 hours.

Isaac Sim needs about 8 GB of VRAM. Everything above ran on an RTX 4060 with
`num_envs=8`; the config default of 64 has not been tried on that card.
