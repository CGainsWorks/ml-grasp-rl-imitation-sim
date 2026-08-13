# Domain randomisation

## What is randomised

Eleven parameters, grouped by how they fail. The grouping is not cosmetic: the
three groups need different justifications and they break policies in different
ways.

**Dynamics** — change what the optimal policy *is*.

| Parameter | Nominal | Applied to |
| --- | ---: | --- |
| `object_half_size` | 0.022 m | box geom size, with inertia recomputed |
| `object_mass` | 0.08 kg | box body mass |
| `object_friction` | 1.0 | box sliding friction |
| `table_friction` | 0.8 | table sliding friction |
| `gravity` | 9.81 m/s² | world gravity |

**Actuation** — change how a command becomes motion. These are the ones that
bite hardest on real hardware.

| Parameter | Nominal | Applied to |
| --- | ---: | --- |
| `gripper_gain` | 300 N/m | position-actuator gain on both fingers |
| `hand_compliance` | 0.02 | `solref` time constant of the hand weld |
| `action_latency` | 0 steps | queue depth in front of the actuator |

**Sensing** — cheap to randomise and cheap to overdo.

| Parameter | Nominal | Applied to |
| --- | ---: | --- |
| `obs_noise_pos` | 0 m | Gaussian noise on grip and object positions |
| `obs_noise_vel` | 0 m/s | Gaussian noise on velocities |
| `action_noise` | 0 | Gaussian noise added to the commanded action |

A note on friction, because it is the trap in this scene: MuJoCo combines the
friction of two geoms by taking the **elementwise maximum**. The finger pads
therefore have a friction of 0.05, far below anything the randomiser can draw
for the box, so the pad-box contact friction *is* the box's value. With grippy
pads — the obvious first choice — the friction randomisation would have been
silently inert and the ablation would have compared four identical conditions.

## Levels

A level is one scale factor applied to every range, so the ablation varies a
single knob:

| Level | Scale | Parameters randomised |
| --- | ---: | ---: |
| `none` | 0.0 | 0 |
| `low` | 0.4 | 11 |
| `medium` | 1.0 | 11 |
| `high` | 1.8 | 11 |
| `shifted` | 1.0 | 10, all outside the training ranges |

For a multiplicative parameter with base range `[lo, hi]`, a level of scale `s`
draws from `[1 + (lo-1)s, 1 + (hi-1)s]`. For an additive one it draws from the
midpoint plus or minus half the width times `s`. So `none` is exactly the
nominal world, and every wider level contains every narrower one.

![randomisation ranges](plots/randomisation_ranges.png)

The exact numbers live in [`src/randomisation/configs/`](../src/randomisation/configs/)
and are copied into the `config.json` of every training run, so a result can
always be traced to the distribution it came from.

## `shifted` is not a training level

It is the held-out evaluation distribution and the stand-in for a real robot:
heavier box (3.2–4.0× nominal), more slippery (0.50–0.62×), a weaker gripper
(0.36–0.46× gain), a much more compliant wrist, three to four steps of command
latency, and roughly ten times the sensing noise of the `medium` level.

Those ranges sit **outside** the ranges `none`, `low` and `medium` ever draw
from, and partially outside `high`. That partial overlap is deliberate: if the
widest training level could never reach the shifted worlds, the ablation would
only be able to show that everything fails, which is not an interesting result.
`tests/test_randomisation.py::test_shifted_is_outside_the_training_ranges`
asserts the separation, so it cannot rot.

What `shifted` deliberately does **not** change is the object size. The hand has
no wrist rotation, so the pads always close along world *x*; a square box at 45°
of yaw presents √2 times its side, and above roughly 27 mm of half-size it is
ungraspable at the worst yaw no matter what the policy does. Testing
generalisation to sizes that are geometrically impossible would measure the
gripper, not the policy. Sizes are capped at 0.024 m everywhere, and the missing
wrist degree of freedom is recorded in [limitations](limitations.md).

## What the randomisation costs

Nothing measurable in wall-clock: parameters are written into `MjModel` fields
at reset, with no recompilation. The cost is in sample efficiency, and that is
what the [ablation](../README.md#randomisation-ablation) measures.
