# Isaac Lab port

**Status: working. All seven bring-up checks pass, at both `none` and `medium`
randomisation.**

Brought up against **Isaac Sim 5.1.0** and **Isaac Lab 2.3.2**, Python 3.11,
CUDA 12.8, on an RTX 4060. Reproduce with:

```bash
python scripts/isaac_bringup.py --num-envs 8 --episodes 2 --randomisation none
python scripts/isaac_bringup.py --num-envs 8 --episodes 2 --randomisation medium
```

| # | Check | Result |
| --- | --- | --- |
| 1 | Environment constructs and resets, `num_envs` worlds | pass |
| 2 | Box stays on the table under zero actions | pass, 8/8 envs |
| 3 | Observation layout matches the MuJoCo table, index by index | pass |
| 4 | Reward inside Isaac matches the shared numpy implementation | pass, max difference **~5e-08** |
| 5 | Scripted expert grasps and lifts | pass, **16/16** |
| 6 | Scripted expert holds it at the hold point | pass, **16/16** |
| 7 | Randomisation is live at `medium`, inert at `none` | pass |

Check 4 is the one that matters most. The claim this port exists to support is
"it is the same task, not a similar one", and it is now measured inside the
running simulator rather than asserted: the reward computed on the GPU from
torch tensors agrees with the numpy implementation the MuJoCo environment uses
to eight decimal places, on the same states.

## Cross-simulator transfer

A policy trained in MuJoCo, exported to TorchScript and dropped into Isaac with
no adaptation:

| | success on Isaac | 95% Wilson |
| --- | ---: | --- |
| Scripted expert (reference) | 1.000 | [0.862, 1.000] |
| `bcrl_high_s0`, trained in MuJoCo | **0.500** | [0.314, 0.686] |

24 episodes each, `scripts/isaac_cross_sim.py`, results in
`experiments/results/cross_sim.json`. Different physics engine, different
contact solver, different embodiment — a Franka driven by differential IK
against a free-floating hand dragged by a mocap weld — and half the episodes
still succeed with no fine-tuning. This is a stronger transfer result than the
`shifted` proxy, and unlike that proxy it is not a distribution this repository
designed.

## What was wrong when this file was written blind

It was written against the documentation before Isaac Sim was installed, and
none of it survived contact. The list is the honest content of "ported but
never run":

| Written | Actually required |
| --- | --- |
| Assets declared inside a custom `InteractiveSceneCfg` subclass | Assets are fields on the *env* config; the scene config only carries `num_envs`/`env_spacing` |
| No `_setup_scene` | Mandatory: instantiate assets, register them on `self.scene`, clone the environments, add a light |
| `omniverse://localhost/NVIDIA/Assets/...` USD path | `FRANKA_PANDA_HIGH_PD_CFG` from `isaaclab_assets`; the high-PD variant, because the default gains track an IK target too softly to grasp |
| Grip point = `panda_hand` origin | The fingertips are 0.107 m along the hand's local z; without the offset the controller servos the wrist to the box and the fingers close 10 cm short |
| IK in relative *position* mode | Relative position leaves orientation to drift, and the home pose holds the gripper at 45°, which cannot do a top-down grasp. Absolute *pose* mode with a fixed downward quaternion reproduces the MuJoCo hand, whose orientation is pinned by its weld |
| Target recomputed from the measured pose each step | A persistent setpoint, like the MuJoCo mocap body. Recomputing from the measurement makes a zero action mean "stay wherever gravity has dragged me", and the arm sags out of the workspace in about two seconds |
| Setpoint floor at the table top | The floor has to sit *below* the table: the achieved pose lags the setpoint, so clamping at the table top leaves the fingertips unable to reach a box resting on it |
| Franka's shipped home pose | Puts the fingertips at z = 0.383, below the 0.40 m table top: the arm starts inside the table and flicks the box off it |
| Geometric grasp test | Filtered `ContactSensor` on each finger, as the MuJoCo environment reads its contact list |

## Two bugs worth naming

**A near-singular start pose, not gravity.** The first fix attempt assumed the
arm was drooping under gravity — wrong: `FRANKA_PANDA_HIGH_PD_CFG` already sets
`disable_gravity = True`. Measuring instead showed `panda_joint4` pinned against
its −0.07 limit, with the arm oscillating between two configurations: the start
pose left the elbow nearly straight, close enough to a singularity that the
damped-least-squares solver never converged. `scripts/isaac_pregrasp.py` searches
configurations with the simulator's own forward kinematics, batched across
environments, and finds one with joint 4 at −1.42, clear of both limits. The
table also moved to x = 0.48, inside the Franka's comfortable reach.

**The measurement, not the port.** `DirectRLEnv` auto-resets the instant the
time-out fires, so reading state after the last step of an episode reports the
*next* episode's freshly placed box. Every successful grasp looked like a drop.
The bring-up now stops two steps short of the horizon.

## Randomisation

Wired through Isaac Lab's event manager, driven by the same JSON the MuJoCo
environment uses, with the same interval arithmetic (`_scaled` reproduces
`ParamSpec.sample`):

| Parameter | How it is applied in Isaac |
| --- | --- |
| `object_mass` | `randomize_rigid_body_mass`, scale operation |
| `object_friction` | `randomize_rigid_body_material` on the box |
| `table_friction` | `randomize_rigid_body_material` on the table |
| `gripper_gain` | `randomize_actuator_gains` on the finger joints |
| `action_noise` | the environment's action noise model |
| `obs_noise_pos`, `obs_noise_vel` | applied in `_get_observations`, *not* through Isaac's observation noise model — see below |

Sensing noise is applied by hand because Isaac's noise model perturbs every
element of the observation independently, which breaks an invariant the MuJoCo
environment maintains: a real system has one pose estimate, and the
relative-position entries are derived from it. Noising them separately would
hand the policy two inconsistent readings of the same quantity.

Not mapped, and honestly so: `object_half_size` (needs a pre-startup scale
term), `hand_compliance` (a property of the MuJoCo weld, no Isaac analogue),
`action_latency` (would need a command queue in the task), and `gravity`
(per-scene rather than per-environment in Isaac).

Check 7 exists because a randomisation config that silently does nothing is the
worst kind of bug: training still runs, curves still look fine, and the ablation
quietly compares identical conditions. It immediately earned its keep —
`GraspTaskCfg.__post_init__` fires when the config is *constructed*, before a
caller can set `randomisation_level`, so every level was running the default's
ranges, including `none`. `GraspTask.__init__` now rebuilds the events from the
level actually on the config.

## Still not done

1. **Nothing trained to success here.** `scripts/isaac_train.py` runs this
   repository's own SAC against the vectorised environment. A 15 000-step run
   with 32 environments (480 000 transitions, 29 minutes on the 4060) learned to
   grasp almost perfectly and never learned to lift:

   | env steps | grasp rate | mean peak lift | success |
   | ---: | ---: | ---: | ---: |
   | 2 500 | 0.63 | 0.008 m | 0.00 |
   | 5 000 | 0.91 | 0.005 m | 0.00 |
   | 10 000 | 1.00 | 0.006 m | 0.00 |
   | 15 000 | 0.94 | 0.053 m | 0.00 |

   That is the *same* local optimum the stalled MuJoCo seeds fall into — grasp
   the box, hold it on the table, collect 0.73 per step instead of the 9.75
   available at the hold point. Reproducing it in a different physics engine,
   with a different embodiment, says the trap belongs to the task and the reward
   rather than to MuJoCo. `experiments/runs/isaac_sac_none/progress.csv` has the
   curve. Every headline number in the top-level README still comes from MuJoCo.
2. **The unmapped randomisation parameters** listed above.
3. **A proper sim-to-sim ablation** — cross-simulator transfer is measured for
   one policy, not across the randomisation levels.

## Installing

```bash
python -m venv C:\isaac\venv311                      # Python 3.11 specifically
C:\isaac\venv311\Scripts\python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
C:\isaac\venv311\Scripts\python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
C:\isaac\venv311\Scripts\python -m pip install "setuptools<81"    # see below
C:\isaac\venv311\Scripts\python -m pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation
```

Three things that will bite:

* **`setuptools<81` and `--no-build-isolation`.** Isaac Lab pulls `flatdict`,
  whose `setup.py` imports `pkg_resources`, which setuptools 81+ no longer
  ships. Without the pin the install dies in metadata generation.
* **Disk.** About 15 GB, and it wants a fast disk: on a drive writing at
  0.26 MB/s the download alone projects to roughly 17 hours.
* **One environment per process.** Constructing a second `GraspTask` after
  closing the first hangs. Every script here builds one environment and exits.

Isaac Sim needs about 8 GB of VRAM. Everything above ran on an RTX 4060 with
`num_envs` of 8 to 32.
