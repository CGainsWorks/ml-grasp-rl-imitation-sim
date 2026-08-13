# Reading the results

The tables in the README are generated from `experiments/results/`. This is what
they say, including the parts that are unflattering.

## 1. The task is solvable, and the scripted expert is the yardstick

The expert scores 1.00 on the nominal world, 0.94 at medium randomisation and
0.47 on the held-out `shifted` worlds. Everything else is read against those
numbers. When a learned policy scores 0.03 on `shifted`, the relevant
comparison is 0.47, not 1.00.

## 2. Imitation is strong where its data is, and only there

Behaviour cloning on 200 demonstrations reaches 1.00 on the nominal world and
0.90 at medium randomisation — it matches the expert on the distribution the
demonstrations came from. On `shifted` it drops to 0.24, *below* the expert's
0.47 on the same worlds. That is the interesting part: the clone is not merely
copying a policy that degrades, it degrades faster than the thing it copied.
The expert filters its pose estimate and caps its speed per phase; a memoryless
clone of its input-output behaviour inherits neither, so under sensing noise and
command latency it is strictly worse.

The data-efficiency sweep puts a number on the compounding-error story: 5
demonstrations give 0.37, 25 give 0.89, 50 give 0.97, and the validation action
error barely moves across that range (0.020 to 0.008). Small action error and
poor success at the same time is the signature.

DAgger, run against the shifted worlds, recovers most of the gap: 0.41 against
BC's 0.24, close to the expert's 0.47. It gets there by querying the expert in
the worlds where the learner actually fails — which is exactly the assumption
that does not hold on real hardware, where the expert is a person with a
joystick.

## 3. SAC from scratch is unreliable at this budget, and the interval says so

On the nominal world, five seeds of SAC scored 1.00, 1.00, 0.00, 0.00, 0.00.
Mean 0.40, 95% interval across seeds [0.00, 1.00]. That interval is the result.
Any one of those seeds, reported alone, would be a lie in one direction or the
other — and it is why the brief for this repository asked for multiple seeds.

The failure has a mechanism. The stalled seeds settle at an entropy coefficient
near 0.025 while the successful ones sit near 0.17, and the stalled policies do
something specific: they grasp the box reliably (grasp rate 0.9) and hold it on
the table, which the reward pays 0.73 per step for against 9.75 at the hold
point. A nearly deterministic policy in that basin has no route out.
`docs/plots/entropy_collapse.png` is one point per run.

Raising the target entropy — the obvious fix — was tried on the three stalled
seeds and did not work: the coefficient roughly doubled and none of the three
escaped within 100 000 steps. Mechanism identified, fix not found. Details in
[limitations](limitations.md).

Under randomisation, from-scratch SAC barely gets off the ground at all: 0.15,
0.22 and 0.12 at low, medium and wide randomisation on the nominal world, and
essentially zero on `shifted`. The right conclusion is not "domain
randomisation does not work" but "200 000 steps is not a budget from-scratch
SAC can absorb randomisation on".

## 4. Demonstrations are what make the budget viable

The same algorithm, seeded with a cloned actor and 20 000 demonstration
transitions pinned in the replay buffer, reaches 0.97 on the nominal world and
0.73 at medium randomisation — against 0.22 and 0.13 from scratch. It gets
there fast: the medium-randomisation runs are above 0.9 within 20 000–30 000
steps, which is before from-scratch SAC has finished its random-action phase in
any useful sense.

That is the practical finding of this repository, and it is not a subtle one:
on a CPU budget, on a contact-rich task with a shaped reward, the difference
between "works" and "does not work" was the demonstrations, not the algorithm.

### The dip at 100 000 steps is the schedule, not noise

Every seeded curve in the middle panel of the training figure falls off a cliff
at exactly 100 000 steps — from about 0.95 to about 0.5 at medium randomisation
— and then climbs back to roughly 0.75 by the end. That is where
`--bc-decay-steps 100000` takes the behaviour-cloning coefficient to zero. Up to
that point the policy is anchored to the demonstrations and scores near the
clone; after it, the policy is on pure SAC and the critic's errors are no longer
held in check by anything.

Two things follow, and both are worth stating rather than smoothing over.

The final numbers in the tables are therefore taken *after* the drop, not at the
peak — they are what the method produces at 200 000 steps, not the best point on
its curve. Reporting the peak would have made every seeded row look better by
roughly 0.2, and it would have been selection on the evaluation the table
reports.

And the schedule is visibly wrong. A coefficient that decays over the whole run,
or one that decays only once the critic's loss has settled, would very likely
avoid the cliff. That was not rerun: the honest reason is that the grid is about
two hours of CPU and the finding — that removing the anchor costs a fifth of the
success rate — is more useful than a tuned curve would have been.

## 5. Randomisation buys transfer, and here it costs nothing measurable

From the imitation-seeded ablation, five seeds per row:

| trained with | on its own distribution | on `shifted` | gap |
| --- | ---: | ---: | ---: |
| `none` | 1.00 | 0.002 | +0.998 |
| `low` | 0.80 | 0.028 | +0.776 |
| `medium` | 0.73 | 0.032 | +0.694 |
| `high` | 0.66 | 0.072 | +0.590 |

The gap column is the sim-to-real problem in four rows: a policy trained without
randomisation is *perfect* on its own worlds and completely useless off them,
and every widening of the randomisation narrows the gap.

**The falling first column is not the cost of randomisation.** It is tempting to
read that column as a trade-off — robustness bought with performance — and it is
not, because "its own distribution" is a different, harder distribution in every
row. The comparison that controls for that is the headline table, where every
method is scored on the *same* worlds:

| trained with | on `none` | on `medium` | on `shifted` |
| --- | ---: | ---: | ---: |
| `none` | 1.000 | 0.516 | 0.002 |
| `low` | 0.968 | 0.640 | 0.028 |
| `medium` | 0.968 | 0.726 | 0.032 |
| `high` | 0.976 | 0.756 | 0.072 |

On a fixed evaluation distribution, wider randomisation is never worse here and
is monotonically better on the two harder ones. Whatever the usual cost of
randomisation is, this task and this budget do not show it — which is worth
saying plainly, because the trade-off is often asserted rather than measured,
and the measurement here does not support it.

Two honest caveats. First, the absolute transfer numbers stay low: 0.072 at
best, against 0.47 for the scripted expert on the same worlds. Wide
randomisation moves the needle in the right direction and does not come close to
solving the shift at this budget. Second, on `shifted` only `none` versus `high`
is separated by more than its confidence intervals; the `low` and `medium` rows
overlap each other.

## 6. The same task in a second simulator

The Isaac Lab port is not decoration: it is a second opinion, and it agrees with
two of the findings above.

**Cross-simulator transfer.** A policy trained in MuJoCo (`bcrl_high_s0`),
exported to TorchScript and dropped into Isaac with no adaptation, scores
**0.500** [0.314, 0.686] against **1.000** for the scripted expert in the same
environment. Different physics engine, different contact solver, and a Franka
driven by differential IK instead of a free-floating hand on a mocap weld. Half
the episodes still succeed. That is a far stronger statement than the `shifted`
proxy can make, because Isaac is not a distribution this repository designed.

**The local optimum is not a MuJoCo artefact.** Running the same SAC
implementation in Isaac, from scratch, for 480 000 transitions produced a policy
that grasps the box on essentially every episode and lifts it on none:

| env steps (x32 envs) | grasp rate | mean peak lift | success |
| ---: | ---: | ---: | ---: |
| 2 500 | 0.63 | 0.008 m | 0.00 |
| 10 000 | 1.00 | 0.006 m | 0.00 |
| 15 000 | 0.94 | 0.053 m | 0.00 |

That is exactly the basin three of the five MuJoCo seeds settled into. Two
different engines, two different embodiments, the same trap: the reward pays
0.73 per step for holding the box on the table, and a policy that has stopped
exploring has no route to the 9.75 available at the hold point. It is a property
of the task and the shaping, not of the simulator.

## 7. What would change these numbers

In rough order of expected effect:

* **More steps.** Every randomised condition was still improving when training
  stopped.
* **A wrist degree of freedom**, so the task involves alignment rather than
  only reach-and-close.
* **Measured randomisation ranges** instead of plausible ones — see
  [sim-to-real](sim-to-real.md).
* **A fix for the entropy collapse**, which would make the from-scratch
  baseline a fair comparison rather than a demonstration of variance.
