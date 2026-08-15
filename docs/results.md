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

### The clone depends on an input its teacher never reads

This is the sharpest version of the point above, and it was found by accident
while sourcing the randomisation ranges from the literature
([randomisation-sources.md](randomisation-sources.md)). The observation carries
the object's orientation — two columns of its rotation matrix, indices 14:20 —
and no level of randomisation ever perturbed them, because real pose estimators
were not consulted when the ranges were chosen. They report 8°–18° of rotation
error.

Adding that error, five seeds, 100 episodes, everything else held at the same
sourced ranges (`experiments/measured_level.py`):

| policy | with orientation error | without | cost |
| --- | ---: | ---: | ---: |
| **scripted expert** | 0.840 | 0.810 | **+0.030** |
| behaviour cloning | 0.236 | 0.680 | **−0.444** |
| BC + SAC, wide randomisation | 0.158 | 0.272 | −0.114 |
| BC + SAC, medium randomisation | 0.168 | 0.230 | −0.062 |
| SAC + entropy floor, wide randomisation | 0.086 | 0.086 | 0.000 |

The expert does not care, and the reason is in its source: `OBJ_ROT_X` is
*defined* in `src/policies/scripted_expert.py` and never read. The expert is a
function of grip position, object position and the commanded goal. Its
orientation input could be replaced with noise, or deleted, and its behaviour
would be identical.

Its clone loses nearly half its success rate to that same input being wrong.
Behaviour cloning fits observations to actions over 200 demonstrations; nothing
in that procedure distinguishes an input the teacher used from one that merely
correlated with what the teacher did. Object yaw correlates with graspability
here — a square box at 45° presents √2 times its side — so the clone has every
statistical reason to lean on it, and no way to learn that the *demonstrated
behaviour* did not.

Two things follow. The compounding-error story in the paragraph above is not
the only way a clone degrades faster than its teacher: it can also inherit a
dependence the teacher never had. And a randomisation axis that is missing from
training is not merely untested — it is an invitation for the policy to become
sensitive to it, which is exactly backwards from what randomisation is for.

The last row is the control that makes the reading safe. `SAC + entropy floor`
is unaffected because at 0.086 it is not grasping reliably enough for pose
accuracy to matter; the effect tracks how much a policy has to lose.

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

Raising the target entropy — the obvious fix — was tried first and did not work:
the coefficient roughly doubled, to about 0.05, and none of the three seeds
escaped. A hard **floor** under the coefficient does work, and completely:

| | per-seed | mean | 95% t | solved |
| --- | --- | ---: | --- | ---: |
| baseline, 200k steps | 1.0, 1.0, 0.0, 0.0, 0.0 | 0.400 | [0.000, 1.000] | 2/5 |
| entropy floor, 100k steps | 0.97, 1.0, 1.0, 1.0, 1.0 | **0.993** | [0.975, 1.000] | **5/5** |

Every stalled seed is rescued at half the budget that failed without it, and the
interval goes from carrying no information to being tight. Pink exploration
noise — the other hypothesis, that the missing behaviour is temporally extended
and white noise never samples it — rescues one seed of three: real, but not the
explanation. The full investigation is in [exploration.md](exploration.md).

**The floor value does not transfer between distributions.** Applying the same
0.15 under randomisation, both arms at 300 000 steps and five seeds, gives a
much weaker result and one outright regression: `medium` 0.680 against 0.460,
`high` 0.407 against 0.160, and `low` **0.000 against 0.113** — at that level a
floor of 0.15 stops the policy learning to grasp at all (grasp rate 0.33 against
0.91, p ≈ 0.002), which is a different failure from the collapse basin.

Tuning the floor per level instead fixes that, and the full matrix says
something more useful than either version
(`experiments/results/floor_by_level.json`, five seeds per cell):

| level | floor 0.00 | floor 0.05 | floor 0.15 | floor 0.30 |
| --- | --- | --- | --- | --- |
| `none` | 0.400 | 0.289 (n=3) | **0.993** | 0.989 (n=3) |
| `low` | 0.113 | **0.587** | 0.000 | 0.000 |
| `medium` | 0.460 | 0.587 | **0.680** | — |
| `high` | 0.160 | **0.467** | 0.407 | — |

A floor beats no floor at **every** level — best value against that level's own
control at a matched budget: `none` t = 2.42, `low` t = 3.13, `medium` t = 1.60,
`high` t = 2.70. And no single value does it: `none` needs at least 0.10 and
fails at 0.05, `low` needs at most 0.05 and dies at 0.15, `medium` and `high` are
indifferent between them. The value-sensitive levels are the two mildest, in
opposite directions. `docs/plots/entropy_floor.png` plots it;
[exploration.md](exploration.md) has the controls.

Two earlier versions of this section were wrong and are worth recording. The
first claimed the floor works under randomisation and merely needs three times
the budget — that came from comparing floored runs at 300 000 steps against a
baseline at 200 000. The second, after the matched control, claimed the floor is
harmful at `low` — that came from testing one floor value at one level.

**The grid has been rerun with the tuned floor, at the same budget.** The
original rows are kept beside it — they record what a standard SAC configuration
does here — and the floored rows record what the task allows. Both at 200 000
steps, five seeds, evaluated on the same fixed distributions:

| trained with | on `none` | on `medium` | on `shifted` |
| --- | ---: | ---: | ---: |
| `none` → `none` + floor | 0.402 → **0.986** | 0.272 → **0.640** | 0.002 → 0.000 |
| `low` → `low` + floor | 0.146 → **0.364** | 0.080 → **0.206** | 0.008 → 0.002 |
| `medium` → `medium` + floor | 0.220 → **0.582** | 0.128 → **0.364** | 0.004 → 0.004 |
| `high` → `high` + floor | 0.122 → **0.390** | 0.058 → **0.240** | 0.000 → 0.006 |

Roughly a tripling everywhere except the `shifted` column, which does not move.
That last part matters: **fixing the collapse does not fix transfer.** The
obvious objection to section 5's sim-to-real result — that the baseline was
simply undertrained — is ruled out by policies that are three times better on
their own distribution and no better at all off it.

Under randomisation, from-scratch SAC still barely gets off the ground: even
with the floor, `low` reaches 0.364 and `high` 0.390 on the nominal world. The
right conclusion is not "domain randomisation does not work" but "200 000 steps
is not a budget from-scratch SAC can absorb randomisation on" — the same runs
given 300 000 steps reach 0.587 and 0.467 on their own distributions.

## 4. Demonstrations are what make the budget viable

The same algorithm, seeded with a cloned actor and 20 000 demonstration
transitions pinned in the replay buffer, reaches 0.97 on the nominal world and
0.73 at medium randomisation — against 0.22 and 0.13 from scratch. It gets
there fast: the medium-randomisation runs are above 0.9 within 20 000–30 000
steps, which is before from-scratch SAC has finished its random-action phase in
any useful sense.

That was written as the practical finding of this repository — on a CPU budget,
on a contact-rich task with a shaped reward, the difference between "works" and
"does not work" was the demonstrations, not the algorithm. The entropy
investigation has since taken it apart. At an **identical 200 000-step budget**,
from-scratch SAC with each level's own entropy floor:

| evaluated on | from scratch | + tuned floor | demonstration-seeded |
| --- | ---: | ---: | ---: |
| `none` | 0.402 | **0.986** | 1.000 |
| `medium` | 0.272 | **0.640** | 0.516 |
| `shifted` | 0.002 | 0.000 | 0.002 |

(`none`-trained policies in every column, so the comparison is like for like.)
From scratch with a floor is level with the seeded version on the nominal world
and *ahead of it* on the harder evaluation.

What demonstrations still buy is real but narrower than claimed: **speed**,
reaching that level inside 30 000 steps rather than 200 000, and **no
hyperparameter** — nobody using them has to know the collapse exists or what
floor value this distribution wants. On a CPU budget an order of magnitude in
sample efficiency is the difference between an experiment that fits in a lunch
break and one that does not, so this is still the practical finding. But
"demonstrations or it does not work" is not what the data says, and the version
of this paragraph that claimed it was written before the comparison existed.

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

The Isaac Lab port is not decoration: it is a second opinion. It confirms one
finding, and it contradicts an optimistic reading of another.

### It confirms the local optimum

Running the same SAC implementation in Isaac, from scratch, for 480 000
transitions produced a policy that grasps the box on essentially every episode
and lifts it on none:

| env steps (x32 envs) | grasp rate | mean peak lift | success |
| ---: | ---: | ---: | ---: |
| 2 500 | 0.63 | 0.008 m | 0.00 |
| 10 000 | 1.00 | 0.006 m | 0.00 |
| 15 000 | 0.94 | 0.053 m | 0.00 |

That is exactly the basin three of the five MuJoCo seeds settled into. Two
engines, two embodiments, the same trap: the reward pays 0.73 per step for
holding the box on the table against 9.75 at the hold point, and a policy that
has stopped exploring has no route between them. It is a property of the task
and the shaping, not of the simulator.

Seeding with demonstrations rescues it there too, exactly as in MuJoCo. Five
seeds per arm in Isaac:

| arm | per-seed | mean | 95% t |
| --- | --- | ---: | --- |
| from scratch | 0.00, 0.00, 0.00, 0.00, 0.00 | **0.000** | [0.000, 0.000] |
| demonstration-seeded | 0.88, 0.97, 1.00, 1.00, 1.00 | **0.969** | [0.902, 1.000] |

The from-scratch arm fails on every seed — this is a reliable property of the
task, not seed variance — and demonstrations take the same algorithm to 0.969.
That is the MuJoCo headline finding reproduced in a second engine, with the same
five-seed standard.

### It confirms the cure too, at a value nobody would have guessed

The floor was first tried here at **0.15**, the value that works on the MuJoCo
nominal world, and the result looked like a failure: 0.463 against a control of
0.194 across five seeds, t = 1.01, both arms spanning 0.000 to 1.000. That was
written up as "the collapse reproduces and the cure does not".

It was the wrong reading, and the MuJoCo matrix said which experiment to run
next: there the useful floor is different for every distribution, and Isaac is a
larger move than any randomisation level. Sweeping the value instead of the
intervention (`experiments/isaac_floor_sweep.py`, 15 000 steps x 32
environments, nominal world):

| floor | per-seed | mean | 95% t | against no floor |
| ---: | --- | ---: | --- | --- |
| 0.00 (control) | 0.09, 0.88, 0.00, 0.00, 0.00 | 0.194 | [0.000, 0.669] | — |
| 0.05 | 0.00, 0.00, 0.00 | 0.000 | [0.000, 0.000] | t = −1.13 |
| 0.15 | 0.81, 0.00, 0.50, 1.00, 0.00 | 0.463 | [0.000, 1.000] | t = 1.01 |
| **0.30** | **1.00, 1.00, 1.00** | **1.000** | **[1.000, 1.000]** | **t = 4.71** |

Three seeds of three at 1.000, a zero-width interval, and the largest effect in
this repository. The curves show the escape happening: at floor 0.30 the peak
lift sits at 0.002 m through 10 500 steps and then jumps to 0.203 m at 12 000 —
the policy is in the grasp-and-hold basin and climbs out of it. At floor 0.05 it
never leaves: 0.0056 m at 15 000 steps, flat throughout.

So the second engine reproduces the failure **and** the fix, and the reason the
first attempt missed it is the finding rather than an excuse: **the useful floor
is a property of the distribution, not of the algorithm**. In MuJoCo the value
that works ranges from 0.05 at `low` to 0.15 at `none`, and here — a different
contact solver, a different arm, a different action scale — it is 0.30, with
0.05 scoring zero. A single number carried across simulators would have been
wrong, and was.

### Randomisation in Isaac costs much more than in MuJoCo, and more steps make it worse

The first randomised training grid in the port, demonstration-seeded, five
seeds (`experiments/results/isaac_seed_grid_medium.json`):

| world | per-seed | mean | 95% t |
| --- | --- | ---: | --- |
| nominal | 0.88, 0.97, 1.00, 1.00, 1.00 | 0.969 | [0.902, 1.000] |
| `medium` | 0.22, 0.31, 0.38, 0.19, 0.28 | **0.275** | [0.182, 0.368] |

In MuJoCo the same arm goes from 0.97 nominal to 0.726 at `medium`. Here it
goes from 0.969 to 0.275, for the same nominal ranges applied through the same
JSON and the same interval arithmetic.

This document previously explained that away as the budget — 4 000 steps solves
the nominal world and looked too short for the randomised one. **That was a
guess, and it is wrong.** The same grid at 15 000 steps, five seeds:

| budget | per-seed | mean | 95% t |
| --- | --- | ---: | --- |
| 4 000 steps | 0.22, 0.31, 0.38, 0.19, 0.28 | 0.275 | [0.182, 0.368] |
| **15 000 steps** | 0.03, 0.09, 0.28, 0.22, 0.03 | **0.131** | [0.000, 0.272] |

Nearly four times the training makes it *worse*, and by more than noise
(Welch t = −2.37, dof 6.9). The per-seed curves say what happens: every seed
peaks at its first or second evaluation and declines from there — seed 2 reads
0.53, 0.16, 0.25, 0.34; seed 3 reads 0.44, 0.19, 0.13, 0.16. Mean of the
per-seed best is 0.34 against a final of 0.131.

That is the same shape as the MuJoCo cliff in section 4, but it cannot have the
same cause: the behaviour-cloning coefficient is *held* in these runs, not
decayed. A policy that peaks early under randomisation and then degrades while
its anchor is still in place is a critic problem, not a schedule problem, and
this repository has not diagnosed it.

Two things follow. Randomisation costs far more in Isaac than in MuJoCo and the
budget does not explain it. And the 0.275 figure is a peak caught by a short
run rather than a converged result — reporting finals, as this repository does
everywhere, the honest number at a longer budget is 0.131.

### The behaviour-cloning anchor, isolated

The Isaac runs also settle the question the MuJoCo curves only hinted at. Two
runs, identical seed and configuration, differing only in whether the
behaviour-cloning coefficient decays to zero half-way:

| env steps | coefficient decaying | coefficient held |
| ---: | ---: | ---: |
| 1 000 | 1.000 | 1.000 |
| 3 000 | 0.969 | 0.969 |
| **4 000** | **0.000** | **0.906** |
| 5 000 | 0.000 | 0.938 |
| 8 000 | 0.000 | 0.938 |

Step 4 000 is exactly where the coefficient reaches zero. The MuJoCo curves show
the same cliff, softened: 0.95 down to 0.5, recovering to about 0.75. Here it is
total and does not recover. The anchor is not a formality that can be annealed
away on a schedule chosen in advance; removing it hands the policy back to a
critic that is not ready for it.

### It contradicts cross-simulator transfer

Policies trained in MuJoCo, exported and run in Isaac with no adaptation, five
seeds at each randomisation level, 32 episodes each:

| trained with | per-seed success | across seeds | 95% t |
| --- | --- | ---: | --- |
| `none` | 0.06, 0.13, 0.00, 0.00, 0.16 | 0.069 | [0.000, 0.157] |
| `low` | 0.13, 0.16, 0.00, 0.00, 0.00 | 0.056 | [0.000, 0.153] |
| `medium` | 0.00, 0.00, 0.25, 0.00, 0.00 | 0.050 | [0.000, 0.189] |
| `high` | 0.41, 0.00, 0.00, 0.00, 0.00 | 0.081 | [0.000, 0.307] |
| scripted expert | — | **1.000** | — |

MuJoCo policies mostly do not transfer to Isaac, and no randomisation level
changes that: every interval includes zero and they overlap each other
completely. The expert scores 1.000 in the same environment, so the Isaac task
is not broken — what fails is the learned behaviour.

This measurement was first run with **one seed per level**, and it looked like a
clean result: wide randomisation transferring at 0.41 against ≤0.08 for the
others. That reading was wrong. It was seed 0 of `high`, and its four siblings
scored zero. The single-seed version of this table would have been the most
quotable number in the repository and it would have been an artefact — which is
the same lesson as section 3, arriving from the opposite direction.

The honest conclusion: the `shifted` proxy overstates how transferable these
policies are. Randomisation buys robustness *within* a simulator's contact and
actuator model; it does not, at these ranges, buy robustness to a different
model of contact altogether.

## 7. What would change these numbers

In rough order of expected effect:

* **More steps.** Every randomised condition was still improving when training
  stopped.
* **A wrist degree of freedom**, so the task involves alignment rather than
  only reach-and-close.
* **Measured randomisation ranges** instead of plausible ones — see
  [sim-to-real](sim-to-real.md).
* **Rebuilding the headline grid on top of the entropy floor.** The floor now
  exists and, tuned per level, makes from-scratch SAC a fair baseline rather
  than a demonstration of variance — 0.993 and 0.680 instead of 0.400 and 0.120.
  Every from-scratch row in the tables above predates it. That rerun is about
  three hours of CPU and is the single change that would move the most numbers
  in this document.
* **Sweeping the floor inside Isaac.** The one place the fix has been tried
  without tuning is the one place it did not work (§6). Six hours of GPU per
  value would say whether that is the value or the simulator.
