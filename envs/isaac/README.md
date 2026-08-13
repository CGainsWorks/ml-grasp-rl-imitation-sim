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

Policies trained in MuJoCo, exported to TorchScript and run here with no
adaptation. Five seeds per randomisation level, 32 episodes each
(`scripts/isaac_cross_sim_ablation.py`, results in
`experiments/results/cross_sim_ablation.json`):

| trained with | per-seed success | across seeds | 95% t |
| --- | --- | ---: | --- |
| `none` | 0.06, 0.13, 0.00, 0.00, 0.16 | 0.069 | [0.000, 0.157] |
| `low` | 0.13, 0.16, 0.00, 0.00, 0.00 | 0.056 | [0.000, 0.153] |
| `medium` | 0.00, 0.00, 0.25, 0.00, 0.00 | 0.050 | [0.000, 0.189] |
| `high` | 0.41, 0.00, 0.00, 0.00, 0.00 | 0.081 | [0.000, 0.307] |
| scripted expert (reference) | — | **1.000** | — |

They mostly do not transfer, and no randomisation level reliably helps. The
expert scores 1.000 in the same environment, so this is not a broken task — the
learned behaviour is what fails.

Worth recording how this nearly went wrong: the first version used one seed per
level and showed `high` at 0.41 against ≤0.08 for the rest, which reads as a
clean "wide randomisation transfers across simulators" result. It was seed 0 of
`high` and the other four scored zero.

## Training here

`scripts/isaac_train.py` runs this repository's own SAC against the vectorised
environment — not a framework wrapper, so a difference between the simulators
cannot be blamed on the algorithm.

| run | steps x envs | result |
| --- | --- | ---: |
| SAC from scratch | 15 000 x 32 = 480 000 transitions | **0.000** (grasp rate 1.00, never lifts) |
| BC-seeded, coefficient decaying | 8 000 x 32 | peaks **1.000**, collapses to 0.000 at the decay point |
| BC-seeded, coefficient held | 8 000 x 32 | **0.938** final, 1.000 best |

Demonstrations are recorded here rather than reused from MuJoCo
(`scripts/isaac_record_demos.py`); the expert succeeded on 128 of 128 attempts.
The middle row is the cleanest evidence in the repository that the
behaviour-cloning anchor cannot simply be annealed away on a fixed schedule:
identical seed and configuration, and the only difference is whether the
coefficient reaches zero at step 4 000.

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

All eleven parameters are mapped:

| Parameter | How it is applied in Isaac |
| --- | --- |
| `object_mass` | `randomize_rigid_body_mass`, scale operation |
| `object_half_size` | `randomize_rigid_body_scale`, pre-startup |
| `object_friction` | `randomize_rigid_body_material` on the box |
| `table_friction` | `randomize_rigid_body_material` on the table |
| `gripper_gain` | `randomize_actuator_gains` on the finger joints |
| `hand_compliance` | `randomize_actuator_gains` on the *arm* joints, inverted |
| `action_latency` | a per-environment command queue in the task |
| `gravity` | `randomize_physics_scene_gravity`, per scene |
| `action_noise` | the environment's action noise model |
| `obs_noise_pos`, `obs_noise_vel` | applied in `_get_observations`, *not* through Isaac's observation noise model — see below |

Two of those are analogues rather than translations, and are labelled as such:

* **`hand_compliance`** is the `solref` of the MuJoCo weld that drags the hand —
  how softly the hand follows its setpoint. There is no weld here. The closest
  honest analogue is the arm's joint stiffness, which governs the same thing:
  how hard the arm insists on reaching the pose it was told. The mapping is
  inverted, because a more compliant weld is a less stiff arm.
* **`gravity`** is per *scene* in Isaac, not per environment, so all
  environments share one draw. MuJoCo draws one per episode per world. Same
  interval, coarser granularity.

Randomising object scale is a USD-level edit, and Isaac refuses to combine it
with scene replication: replicated instances share properties, so a
per-environment size would silently apply to all of them. `replicate_physics`
is therefore switched off exactly when the level randomises size. It costs
scene-setup time; the alternative is a randomisation that lies.

Sensing noise is applied by hand because Isaac's noise model perturbs every
element of the observation independently, which breaks an invariant the MuJoCo
environment maintains: a real system has one pose estimate, and the
relative-position entries are derived from it. Noising them separately would
hand the policy two inconsistent readings of the same quantity.

Check 7 exists because a randomisation config that silently does nothing is the
worst kind of bug: training still runs, curves still look fine, and the ablation
quietly compares identical conditions. It immediately earned its keep —
`GraspTaskCfg.__post_init__` fires when the config is *constructed*, before a
caller can set `randomisation_level`, so every level was running the default's
ranges, including `none`. `GraspTask.__init__` now rebuilds the events from the
level actually on the config.

## Still not done

1. **No multi-seed grid here.** The training runs above are one seed each. Every
   headline number in the top-level README still comes from MuJoCo, where the
   grid is five seeds per condition.
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
