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

### Correlated sensing error costs the filter, not the clone

Independent per-step noise is the easy model and it flatters anything that
averages, which is why `obs_noise_corr` exists: a first-order filter, magnitude
held constant (lag-1 autocorrelation 0.906 at rho 0.9, standard deviation
unchanged — there is a test). Same ranges, same magnitude, only the temporal
structure differs:

| policy | independent | correlated | change |
| --- | ---: | ---: | ---: |
| **scripted expert** | 0.840 | 0.740 | **−0.100** |
| BC + SAC, wide | 0.158 | 0.110 | −0.048 |
| BC + SAC, medium | 0.168 | 0.144 | −0.024 |
| SAC + entropy floor, wide | 0.086 | 0.074 | −0.012 |
| behaviour cloning | 0.236 | 0.252 | +0.016 |

The ordering is the point, and it is the reverse of the orientation-error table
above. The **expert** loses most, because the expert is the only thing here that
low-passes its pose estimate — and a low-pass filter is exactly what a
correlated error defeats. Its clone loses nothing, because a memoryless clone
was never averaging anything to begin with; it has no filter to defeat.

So the two sensing results say opposite things about the same pair. Give the
expert an input it ignores and its clone collapses; give the expert an error its
filter cannot remove and the clone is unmoved. A clone of a policy is not a
noisier version of that policy, and neither result would have shown up against
the independent-noise model this repository shipped with.

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
decayed.

#### The mechanism is the critic, the critic is fixable, and fixing it does not help

Two candidates fit the symptom, both with literature behind them. The runs end
with the entropy coefficient at **0.0011** — twenty times below the value that
marks the collapse basin — and with critic loss twenty times *higher* than it
started, which is the signature of
[Q-value divergence in offline-to-online fine-tuning](https://arxiv.org/pdf/2310.04411);
a demonstration-seeded run is exactly that setting. The alternative is
[plasticity loss](https://arxiv.org/abs/2411.04832), where a network's ability
to keep learning degrades with training regardless of data quality.

One flag separates them. Adding the floor that solves Isaac's nominal world
(0.30) to the same demonstration-seeded runs, three seeds, everything else
identical:

| | critic loss across training | final success |
| --- | --- | ---: |
| no floor, s0 | 29 → 53 → **639** → 208 | 0.031 |
| no floor, s1 | 24 → 74 → 236 → 324 | 0.094 |
| no floor, s2 | 24 → 80 → 181 → 165 | 0.281 |
| floor 0.30, s0 | 21 → 21 → **17** → 17 | 0.000 |
| floor 0.30, s1 | 21 → 53 → 41 → 18 | 0.031 |
| floor 0.30, s2 | 27 → 44 → 33 → 31 | 0.000 |

**The floor prevents the critic blow-up on every seed** — peaks of 639, 324 and
236 become 21, 53 and 44. So the two candidates are not competing explanations
at all: the entropy coefficient collapsing is what drives the critic loss up,
and clamping it fixes the value function.

**And success gets worse, not better**: 0.010 [0.000, 0.055] against 0.131 for
the unfloored control, Welch t = −2.33. The floored runs still peak at their
first evaluation and decline, exactly as before.

So both hypotheses are dead as *causes* of the poor randomised performance.
They are real, they are linked, they are fixable, and fixing them costs
performance rather than buying it. Whatever makes randomisation expensive in
Isaac is not in the optimiser — it is upstream, in the reward or the
randomisation ranges as this port applies them, and that is where the next
investigation should start rather than in another SAC hyperparameter.

This is worth stating plainly because the tempting write-up was available and
wrong: "critic loss grows twentyfold, we clamped the entropy coefficient, the
critic is stable now" is true, publishable-sounding, and describes an
intervention that makes the task harder.

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

### It is not a calibration constant, which was the hopeful answer

The cheapest explanation for the transfer failure is that the two simulators
agree on what a command *means* and disagree on what it *does* — a policy tuned
to MuJoCo's compliant mocap weld overshooting against Isaac's stiffer IK
controller. If that were it, a scalar would fix it.

`scripts/isaac_transfer_probe.py` scales the action and nothing else, on a
policy that genuinely fails there (64 episodes per arm):

| arm | success | mean peak lift |
| --- | ---: | ---: |
| baseline | 0.062 | 0.082 m |
| every action x0.5 | 0.047 | 0.024 m |
| every action x0.25 | 0.000 | 0.000 m |
| every action x1.5 | 0.062 | 0.070 m |
| every action x2.0 | 0.016 | 0.027 m |
| lateral only x0.5 | 0.078 | **0.130 m** |
| gripper held once grasped | 0.109 | 0.088 m |

**No arm clears its own interval, in either direction.** Scaling down starves
the motion inside the horizon; scaling up destabilises it. So the mismatch is
not a gain anybody forgot to calibrate, and the hopeful answer is gone.

One row points somewhere, weakly. Halving only the *lateral* commands takes
peak lift from 0.082 m to 0.130 m — roughly what a successful MuJoCo policy
reaches — while success stays at 0.078. The policy gets the box up and then
loses it, which is a contact-and-holding failure rather than a reaching one.
That is a hypothesis with one supporting number, not a result.

Also worth recording as a near miss: the first version of this probe used
`bcrl_high_s0`, which produced a baseline of 0.484 and looked like a
contradiction of the 0.05–0.08 in the table above. It is not — that policy is
the single outlier seed that scored 0.406 in the five-seed ablation, and 0.484
agrees with it. Probing the one policy that transfers would have measured
nothing about why the others do not.

### It fails on vertical positioning, and every earlier guess was about grasping

The probes above all assumed the policy was *losing* the box: that is why they
tested grip force, forcing the gripper closed, and contact friction. Reading
the state at the end of the episode rather than the peak shows the assumption
was wrong. The hold point is 0.15 m above the table; success needs the object
within 0.05 m of it, with both pads in contact, on the final step.

| Isaac condition | peak lift | final gap to goal | final grasp | success |
| --- | ---: | ---: | ---: | ---: |
| nominal | 0.091 m | 0.096 m | **0.55** | 0.052 |
| object friction x3 | **0.268 m** | 0.120 m | 0.30 | 0.042 |
| *the same policy in MuJoCo* | *0.13 m* | *—* | *—* | *0.976* |

**The policy is still holding the box in 55% of nominal episodes when the
episode ends.** It is not dropping it; it is stopping about ten centimetres
short. Raise the friction and the opposite happens: the box goes to 0.268 m,
nearly double the target, and ends further from the goal than before.

Under-lift at nominal friction, overshoot at high friction, and in neither case
does it settle. The policy's vertical control does not transfer, and friction
is not a cause but a knob that slides the failure from one side of the target to
the other — which is what the non-monotone friction curve was showing:

| object friction | peak lift | success |
| --- | ---: | ---: |
| x1 | 0.091 m | 0.070 |
| x2 | 0.079 m | 0.031 |
| **x3** | **0.204 m** | 0.070 |
| x5 | 0.053 m | 0.023 |

(128 rollouts per point, one policy. The x3 row replicates across three
independent runs and three different policies; x2 and x5 do not lift at all,
and why the effect is confined to x3 is not explained here.)

A plausible mechanism, stated as a hypothesis because it has not been tested:
MuJoCo's hand is dragged by a compliant weld that sags under load, so a policy
trained there learns to command more vertical motion than it needs. Isaac's
IK controller tracks its setpoint accurately, so the same command overshoots —
unless the box is slipping in the fingers, which at nominal friction it is.
That would make the two simulators disagree about what a vertical command
*achieves* while agreeing about what it means, and it predicts that matching
the weld compliance would transfer better than matching anything else.

## 7. A second task, and the first place the method failed

Everything above comes from one task. `src/rewards/place_reward.py` adds a
second -- pick the box up, carry it somewhere else on the table, put it down --
chosen to invert the first one's assumptions rather than to be easy: the goal
moves laterally, success requires *releasing*, and success requires the object
to have been picked up rather than slid across.

Five seeds, 100 episodes each, same budget and settings as the lift grid:

| | none | shifted |
| --- | ---: | ---: |
| scripted expert | 1.000 | 0.330 |
| behaviour cloning | 0.978 [0.968, 0.988] | 0.010 |
| BC + RL | 0.916 [0.845, 0.987] | 0.112 |
| SAC from scratch | **0.002** [0.000, 0.008] | 0.000 |

**Imitation transferred to a new task unchanged. From-scratch RL did not.** That
is the most useful sentence in this document, because it is the one the first
task could not have produced: on lift-and-hold, demonstrations bought sample
efficiency and a floored from-scratch run got there too, so "what do
demonstrations buy" had a comfortable answer. On the second task they buy
feasibility.

### The from-scratch failure is shaping, not budget, and it repeated a documented mistake

Read from behaviour rather than from the return curve, which was climbing
happily throughout:

| what the shaping was keyed on | success | peak lift |
| --- | ---: | ---: |
| `carry` on `grasped` only | 0.007 | 0.012 m |
| `carry` on the binary lift latch | 0.000 | 0.010 m |
| `carry` on a clearance ramp, `clear` 0.18/step | 0.000 | 0.011 m |
| `carry` on a clearance ramp, `clear` 0.48/step | 0.007 | 0.011 m |
| + `approach` peaking while hovering over the target | 0.000 | **0.130 m** |
| + `approach` peaking at the release point | 0.000 | 0.036 m |
| + `approach` rising monotonically through both | 0.000 | 0.063 m |
| the first design at 600 000 steps | 0.044 | 0.021 m |

The peak-lift column carries the information, not the success column. The first
four designs never pick the box up: they close the pads and push, or grasp and
sit. The fifth produced *carrying* — the first thing in the investigation to do
so — because its maximum sits where the object is held above the target, which
is also where it pays nothing for finishing. Moving the maximum to the release
point stopped the lifting. Making it rise through both recovers half the lifting
and still scores zero.

Two controls close off the boring explanations:

* **Budget** — the first design at 600 000 steps, more gradient updates than the
  entire lift grid was trained with, reaches 0.044 and is still sliding.
* **Task length** — `experiments/place_ladder.py` shrinks the object-to-target
  distance to nothing. All three rungs, fifteen seeds, score 0.000, including
  the rung with no transport to do at all.

And the measurement that explains it, from decomposing the scripted expert's
per-step return:

| | positive reward | share from terms that only pay once the task is done |
| --- | ---: | ---: |
| lift-and-hold | 5.948/step | 51.8% |
| pick-and-place | 3.202/step | **80.7%** |

**Shaping buys segments.** Put a maximum where a segment ends and the policy
learns that segment and stops there — reaching, grasping, lifting and carrying
were each bought exactly this way. Chaining segments is what these designs do
not do, and pick-and-place is four in series where lift-and-hold is two.
Demonstrations supply the chain for free, which is why cloning solved this task
on the first attempt under every one of the seven rewards.

A reverse curriculum then made the same point from the other side. Trained on
start states sampled across the whole task, one policy reaches 0.750 when handed
the object already lifted, 0.367 when handed it grasped on the table, and
**0.000** when it has to close the fingers itself — while the shaped from-scratch
runs grasp on 63-90% of steps and never lift. Each half is learnable; nothing
tried here learns both in sequence. Details and the full stage table are in
[limitations.md](limitations.md).

Meanwhile the imitation column never moved: 0.978 cloned, and 0.870-0.967
demonstration-seeded across the reward variants. **Seven reward designs and a
tripled budget against a clone that worked immediately** is the shape of this
result, and it is the strongest evidence in the repository that hand-designed
dense shaping here is a per-task craft rather than a method.

### The arm, and what the mocap weld was hiding

Everything else in this document trains through a mocap weld: the action is a
Cartesian displacement of a target the hand is dragged towards. `arm=True`
replaces that with a six-jointed UR5-proportioned chain and damped-least-squares
IK, keeping the same action space. Five seeds, same budget, same pipeline:

| | none | shifted |
| --- | ---: | ---: |
| scripted expert | 0.680 | 0.000 |
| behaviour cloning | 0.202 [0.175, 0.229] | 0.000 |
| BC + RL | 0.176 [0.083, 0.269] | 0.000 |
| SAC from scratch | **0.000**, grasp rate 0.000 | 0.000 |

Against the weld's 0.593 from scratch and ~1.000 cloned, this is a large drop,
and the two most informative numbers in it are the ones that are not the success
rate:

* **grasp rate 0.000 from scratch.** Not a working grasp with poor follow
  through — the policy never closes on the box in 200 000 steps. Exploration is
  where the abstraction was paying.
* **BC + RL does not beat BC.** Through the weld it does. Four one-flag variants
  say the damage is not fine-tuning as such:

  | | success on `none` | vs the clone |
  | --- | ---: | ---: |
  | clone, no RL | 0.448 | — |
  | BC term never decays | **0.536** | +0.088, t = +1.04 |
  | fine-tune at `none`, where the demonstrations came from | 0.422 | −0.026, t = −0.33 |
  | standard fine-tune, at `medium` | 0.298 | −0.150, t = −1.88 |
  | critic warmup 3 000 → 20 000 | 0.228 | −0.220, **t = −2.38** |

  Leash the actor to the demonstrations and it is the best arm on the board;
  match the fine-tuning distribution to the demonstrations and the loss goes
  away; give the critic seven times longer to warm up — the change that sounds
  most like good practice — and it is the only one that separates from the
  clone, downwards.

The teacher is also worse, and the demonstration set worse still: the arm expert
succeeds on 19% of `low` episodes, so 200 kept demonstrations came from 1054
attempts and are a biased sample of the easy worlds. Recording on the nominal
world instead, where the same expert manages 0.671, takes the clone from 0.202
to 0.448 (t = −3.35) — the largest single effect measured on the arm, and it is
a data-collection decision rather than an algorithmic one.

### Shape variety costs more than it buys, at this budget

Ten seeds per arm, matched budget, entropy floor and update-to-data ratio:

| trained on | tested on `none` | tested on `shapes` |
| --- | ---: | ---: |
| mixed shapes | 0.142 [0.000, 0.322] | 0.124 [0.000, 0.271] |
| boxes only | **0.593** [0.315, 0.871] | **0.450** [0.247, 0.653] |

Welch t = −3.08 and −2.94. **Box-only training beats shape-trained policies even
when both are tested on shapes**, which is the result the three-seed version
could not reach. Seven of ten shape seeds finish at exactly 0.000; the outcome is
bimodal, not noisy, and that is precisely the distribution where five seeds
mislead.

## 8. What would change these numbers

In rough order of expected effect:

* **Understanding why the second task needs demonstrations.** Four reward
  designs and a tripled budget have not got from-scratch RL off zero on
  pick-and-place while the same code solves lift-and-hold. The likely reason is
  structural rather than a weight: in the lift task the progress term and the
  height term point the same way, so climbing one climbs the other, and in the
  place task they are orthogonal. Testing that costs a curriculum, not a search.
* ~~**More steps.**~~ **Measured.** Every randomised condition was still
  improving when training stopped, and tripling the budget to 600 000 steps
  confirms it: paired by seed, `medium` goes 0.467 → 0.811 and `high` goes
  0.144 → 0.600, with the two seeds that scored 0.000 and 0.133 finishing at
  0.600 and 0.833. Transfer to `shifted` does not move (0.030), so more compute
  buys the own-distribution score and nothing else. The headline grid stays at a
  matched 200 000 steps regardless — see
  [limitations.md](limitations.md).
* **Measured randomisation ranges** instead of plausible ones — the guessed ones
  have now been audited against published measurements
  ([randomisation-sources.md](randomisation-sources.md)) and are optimistic on
  latency by 2-5x, but a survey is not hardware.
* ~~**An Isaac port of the place task.**~~ **Done, and it reproduces the
  finding.** Three seeds each, 4 000 steps, the same shared reward:

  | | Isaac | MuJoCo |
  | --- | ---: | ---: |
  | demonstration-seeded RL | **0.419** (0.414, 0.461, 0.383) | 0.870-0.973 |
  | from scratch | **0.000** (0.000, 0.000, 0.000) | ≤0.007 over seven designs |

  The prediction was recorded in `experiments/isaac_place_grid.py` before the
  runs: if the port is faithful, the *pattern* should survive — demonstrations
  work, from-scratch does not — while the absolute numbers should not match,
  because it is a different robot, gripper and contact solver on a budget fifty
  times smaller. Both halves held. Every from-scratch seed reaches a grasp rate
  of 0.23-0.56 and a success rate of exactly zero, which is the same behaviour
  MuJoCo shows: the policy learns to hold the box and not what to do with it.

  A second simulator agreeing is worth more than an eighth reward design on the
  first one, because it could have disagreed.
