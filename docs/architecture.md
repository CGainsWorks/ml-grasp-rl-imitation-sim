# Architecture

```
                    src/randomisation/configs/*.json
                                 |
                         (ranges per level)
                                 |
                                 v
  envs/mujoco/assets/    +--------------------+        src/rewards/grasp_reward.py
   grasp_scene.xml  ---> |    GraspEnv        | <----  (reward, success, drop)
     (MJCF scene)        |  gymnasium API     |            ^
                         +--------------------+            |
                            |            ^                 | same functions,
              observation   |            | action          | numpy or torch
                 (32,)      v            |                 |
        +----------------------------------------+         |
        |  scripted expert  |  SAC actor  |  BC  |          |
        +----------------------------------------+         |
                 |                  |                       |
                 | demonstrations   | policy.pt             |
                 v                  v                       |
       demonstrations/*.npz    experiments/runs/*     envs/isaac/grasp_task.py
                 |                  |                  (ported, never run)
                 +--------+---------+
                          v
                 src/evaluate.py  ->  experiments/results/*.json
                          |
                          v
                 analysis/plots.py  ->  docs/plots/*.png
```

## The environment

[`envs/mujoco/grasp_env.py`](../envs/mujoco/grasp_env.py) wraps a hand-written
MJCF scene. The full observation and action tables are in that file's docstring
and are not repeated here; the design decisions worth explaining are:

**A free-floating hand, driven by a mocap weld.** The policy commands a
Cartesian displacement of a mocap body; a weld constraint drags the hand towards
it and the solver resolves contact along the way. Nothing is teleported, so a
misaligned finger pair pushes the box away rather than passing through it. What
is missing is the arm: no joint limits, no self-collision, no arm inertia. That
is the largest single gap between this environment and a real cell, and it is
the reason the Isaac port uses a Franka instead.

**A 6-D rotation in the observation, not a quaternion.** Quaternions are
double-covered and discontinuous as a regression target; the first two columns
of the rotation matrix are neither. It matters here because behaviour cloning
regresses straight through this block.

**The grasp flag is read from the contact list, then debounced.** Inferring a
grasp from finger position would call a squeeze on thin air a grasp. Reading
contacts instead is correct but chatters: a held box vibrates against the pads
and one of the two contacts drops out on individual steps. Since success is
evaluated at the final step, that chatter alone was failing roughly half of the
genuinely successful holds, so the flag is latched for three environment steps
(0.12 s) after the last two-pad contact.

**Randomisation is applied by editing `MjModel` in place at reset.** No
recompilation, no reallocation, so per-episode randomisation costs nothing
measurable.

## The learning code

| File | What it is |
| --- | --- |
| `src/policies/networks.py` | Tanh-squashed Gaussian actor, twin critics, and a running observation normaliser that is saved *inside* the checkpoint |
| `src/policies/sac.py` | SAC, with a replay buffer whose demonstration prefix is pinned, an optional critic warm-up, and a scale-normalised behaviour-cloning term |
| `src/policies/scripted_expert.py` | Four-phase state machine used for demonstrations and DAgger labels |
| `src/train_rl.py` | SAC training, and — with `--demos`/`--init-actor` — the imitation-plus-RL run |
| `src/train_il.py` | Behaviour cloning and DAgger |
| `src/evaluate.py` | The one definition of "success rate" |

SAC rather than PPO because the budget per run is a few hundred thousand steps
on a CPU, and an off-policy learner with a replay buffer gets far more out of
that than an on-policy one. Written out rather than imported from Stable
Baselines3 because the imitation-plus-RL variant reaches inside the algorithm:
pinned demonstrations in the buffer, a BC term on the demonstration slice of
each batch, and an actor initialised from a cloned policy. Doing that through a
framework's callback surface is more code, and more obscure code, than the 250
lines in `sac.py`.

All numbers in this repository come from networks of two hidden layers of 128
units (`--hidden 128`; the module default is 256). The observation is 32 floats of
structured state, not pixels; width was never the bottleneck, exploration was.

## Normalisation lives in the checkpoint

`RunningNorm` is an `nn.Module` with buffers, so `state_dict()` carries the
observation statistics. A policy loaded with the wrong statistics behaves
exactly like a badly trained policy, and that is an expensive bug to chase from
the symptom.

## Seed blocks

Three disjoint blocks of episode seeds, so no number is ever measured on the
episodes it was tuned against:

| Block | Base | Used by |
| --- | ---: | --- |
| Training resets | 0 | `src/train_rl.py` environment resets |
| Training-time evaluation | 500 000 | the success column in `progress.csv` |
| Final evaluation | 900 000 | `src/evaluate.py`, every table in the README |

Demonstration recording uses its own offset (100 000) again.
