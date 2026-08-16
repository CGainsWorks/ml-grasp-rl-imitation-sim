# Limitations

Written to be read before the results, not after them.

## The environment is a simplification, in specific ways

**There is no arm in the MuJoCo half.** The hand is a free body dragged by a
mocap weld: no joint limits, no self-collision, no arm inertia, no
singularities, no reachability constraint. Every one of those is a real failure
mode in a real cell. A policy trained here needs a reachability check and a
joint-limit clamp between it and any servo loop. (The Isaac port *does* have a
real Franka, which is how its near-singular start pose was found — so this
limitation is specific to MuJoCo, not to the repository.)

**A six-jointed arm variant now works.** `make_env(..., arm=True)` replaces the
mocap weld with a 6R arm carrying the hand on its flange, driven by
damped-least-squares IK so the policy keeps its Cartesian action space. Joint
limits, self-collision, finite reach and arm inertia now sit between the command
and the fingers, which is what "there is no arm" was complaining about.

Proportions follow the published Denavit-Hartenberg parameters for the UR5
(Universal Robots, [DH parameters](https://www.universal-robots.com/developer/hardware-and-motion/robot-motion-dh-parameters/)):
d1 = 0.089, a2 = 0.425, a3 = 0.392, d4 = 0.109, d5 = 0.095, d6 = 0.082. Published
kinematic constants, plain capsule geoms, no manufacturer meshes.

The scripted expert, 100 episodes each:

| | success | grasp | peak lift |
| --- | ---: | ---: | ---: |
| weld hand (default) | 1.000 [0.963, 1.000] | 1.000 | 0.127 m |
| **six-jointed arm** | **0.680** [0.583, 0.763] | 0.700 | 0.089 m |

A third of episodes now fail for reasons the weld version cannot produce:
configurations the arm cannot reach cleanly, tracking lag under load, and the
inertia of half a metre of steel between the setpoint and the pads. That gap is
the point of the variant, not a defect in it.

Getting there took four wrong diagnoses, and each is worth recording because
each looked like the obvious answer:

1. **Tuning the solver.** Restricting the elbow range, a nullspace posture bias
   and IK restarts with collision checks all failed. The nullspace attempt was
   mathematically vacuous — six joints against a six-dimensional pose target
   leave no redundancy, so `I − J⁺J` is zero except at singularities.
2. **The chain's structure.** The first arm was a stack of collinear capsules.
   Real arms offset the shoulder, elbow and wrist out of the arm plane, and
   those offsets are what let the elbow go somewhere other than through the
   table. Adopting UR5 proportions took penetrating contacts from 105 per twenty
   resets to zero.
3. **Sampling start poses instead of start configurations.** Solving IK to a
   sampled Cartesian pose is collision-blind and produced clean starts on barely
   a third of resets. Sampling *configurations* and rejecting the ones that
   collide makes validity constructive; the start pose is not part of the task,
   so choosing it to suit the arm is free.
4. **The hand frame.** The last and most expensive: "pads facing the table" was
   encoded as the grip frame pointing down, which turns the hand over — the palm
   sits 22 mm along the hand's local +z and the pads hang 46 mm the other way.
   IK then solved the grip site to exactly the right height with the palm 4 cm
   inside the table. A base-placement search run against that inverted hand
   reported the grasp height as unreachable from all 25 candidate placements;
   with the frame corrected the same search reports 0.887–1.000 coverage almost
   everywhere. **The search was measuring the bug, not the geometry.**

Base placement was then chosen by that search rather than by hand
(`experiments/arm_base_search.py`, method after the reachability-map literature,
e.g. [B*: Efficient and Optimal Base Placement for Fixed-Base
Manipulators](https://arxiv.org/pdf/2504.12719)): y = −0.72, z = 0.85, which
reaches every cell of the grasp region with the pads down and nothing inside the
table.

**Policies have now been trained on it, and they are poor.** Five seeds, 100
episodes each, the same 200 000 steps, network, entropy floor and demonstration
pipeline the weld runs used:

| | none | shifted |
| --- | ---: | ---: |
| scripted expert | 0.680 | 0.000 |
| behaviour cloning | 0.202 [0.175, 0.229] | 0.000 |
| BC + RL | 0.176 [0.083, 0.269] | 0.000 |
| **SAC from scratch** | **0.000**, grasp rate 0.000 | 0.000 |

Three things in that table are worth separating.

**From-scratch RL through six joints never closes on the box at all** — grasp
rate 0.000 across every seed, not a low success rate on top of a working grasp.
The weld version at the same settings reaches 0.593. Whatever the mocap weld is
abstracting away, exploration is where the cost lands.

**RL fine-tuning stops helping.** 0.176 against the clone's 0.202, Welch
t = 0.74 — indistinguishable. Through the weld, demonstration-seeded RL improves
on its clone; here it does not, and the honest reading is that a critic trained
on a distribution the policy cannot reliably reach has nothing to add.

**The teacher is much worse, and the demonstration set is worse still.** The arm
expert succeeds on 19% of `low` episodes against the weld expert's ~100%, so 200
kept demonstrations came from 1054 attempts and are a biased sample of the easy
worlds. A second set recorded on the nominal world, where the same expert
manages 0.671, is trained separately rather than assumed to be better.

So the arm variant is structurally sound and the *method* does not carry across
to it at this budget. That is the more useful version of the claim than "there
is an arm".

Randomisation parameters are applied to the arm unchanged apart from
`hand_compliance`, which maps to arm joint stiffness the way the Isaac port maps
it.

**The hand cannot rotate — by default.** The pads close along world *x*, so a
square box at 45° of yaw presents √2 times its side and above roughly 27 mm of
half-size it is ungraspable at the worst yaw regardless of the policy. Object
sizes are therefore capped at 24 mm half-size, and at that cap yaw cannot bind.

`make_env(..., wrist=True)` adds the yaw degree of freedom (34-D observation,
5-D action), and the honest summary is that it helps the *scripted* policy and
defeats the *learned* one. On boxes spanning 15–35 mm, the expert with a wrist
roughly doubles success in the bands where alignment matters (0.167 → 0.333 at
27–31 mm). Trained from scratch on the same distribution, 200 000 steps, three
seeds:

| | per-seed | mean |
| --- | --- | ---: |
| no wrist (4-D) | 0.333, 0.033, 0.000 | 0.122 |
| with wrist (5-D) | 0.000, 0.000, 0.000 | **0.000** |

Adding the degree of freedom made the task unlearnable at this budget, and the
obvious fix does not help. A yaw-alignment reward term was added
(`w_yaw` in `src/rewards/grasp_reward.py`, `src/rewards/configs/wrist.json`) in
two versions — paid everywhere, and gated on proximity to the object. Nine runs,
three seeds each:

| | per-seed | mean |
| --- | --- | ---: |
| no wrist (4-D control) | 0.333, 0.033, 0.000 | 0.122 |
| wrist, no yaw term | 0.000, 0.000, 0.000 | 0.000 |
| wrist + yaw term | 0.000, 0.000, 0.000 | 0.000 |
| wrist + yaw term, gated on proximity | 0.000, 0.000, 0.000 | 0.000 |

The term does what it was designed to do and it is not enough. Measured on the
trained policies, final yaw error falls from **25.9°** (the value at reset, i.e.
no alignment) to **18.1°** without the term and **4.4°** with it: the policy
learns to square the closing axis to the box within a few degrees. Success stays
at zero.

The first version also showed what a badly-placed shaping term costs. Paid
everywhere, it was satisfiable *without doing the task* — hover in mid-air,
perfectly aligned, collect the absence of a penalty — and the grasp rate fell
from 0.50 (no term) to 0.10 while the yaw error improved. Gating it on proximity
removed that exploit and did not recover the success rate. This is the third
shaping failure recorded in this repository, and the cleanest example of a term
that is easier to satisfy than the objective it was meant to support.

What is left untested is the weight. `w_yaw = 1.5` was chosen to be comparable
with the other terms and never swept, so "the alignment reward does not rescue
the wrist" is established at one weight, in two placements, and no further.

**Objects are boxes by default, and cylinders and spheres optionally.**
`src/randomisation/configs/shapes.json` draws the geom type per episode with the
width the pads must close on held equal across the three, so the comparison is
about shape rather than size.

The scripted expert handles all three, and the ordering is not the intuitive
one: **sphere 1.000, cylinder 1.000, box 0.950**. A sphere is the *easiest*
shape for this hand, because a hand that cannot rotate has no way to profit from
knowing a box's yaw — the box's difficulty *is* its orientation, and a sphere
has none.

Learned, the three-seed version of this comparison could not settle anything --
0.167 [0.000, 0.884] against 0.407 -- and the reason was visible in the seeds
rather than in the interval: the outcome is *bimodal*, a run either finds the
behaviour or collapses to exactly zero. Five samples from a bimodal distribution
is the case where a mean and a t interval mislead most, so both arms were taken
to **ten seeds** at matched budget, matched entropy floor and matched
update-to-data ratio (`experiments/shapes_seeds.py`).

| trained on | tested on `none` | tested on `shapes` |
| --- | ---: | ---: |
| mixed shapes | 0.142 [0.000, 0.322] | 0.124 [0.000, 0.271] |
| boxes only | **0.593** [0.315, 0.871] | **0.450** [0.247, 0.653] |

Welch t = −3.08 and −2.94. Shape variety at a fixed budget costs about 45
points, and the part that is not obvious: **box-only training beats
shape-trained policies even when both are tested on shapes.** Widening the
training distribution did not buy performance on the wider distribution; it
bought fewer seeds that learn anything at all. Seven of the ten shape seeds
finish at exactly 0.000 against two of ten for boxes.

This is a budget statement, not a claim that shape variety is harmful. Every
widening of the distribution in this repository costs at 200 000 steps.

No bottles, no bags, no clutter, no bin, and still nothing here demonstrates
grasp *selection* — choosing where to grasp an unfamiliar shape — which is what
most of the grasping literature is actually about.

**The policy is handed the object pose by default, and there is now a camera
path that does not.** `src/perception/pose_cnn.py` estimates the object position
from 64x64 renders of the fixed front camera; `scripts/collect_pose_data.py`
builds the dataset from mixed expert and random rollouts (24 000 frames, 44% of
them with the hand between camera and box); `experiments/perception_eval.py`
substitutes the estimate for the object entries of the observation and runs the
policies on it.

The estimator is deliberately small — four strided convolutions, about 200k
parameters, a few minutes on a CPU. A better one would estimate pose better and
answer the question worse, because the question is what the error *looks like*,
not how small it can be made.

**Two of the three claims this paragraph used to make are now measured and one
is wrong.** Validation split by episode, not by frame, because neighbouring
frames are near-duplicates and a frame split reports several times better than
the truth:

| | modelled | measured |
| --- | --- | --- |
| position error | 0.004–0.010 m, from published YCB-Video results | **0.0065 m** ✓ |
| lag-1 autocorrelation | 0.9, chosen for plausibility | **0.947, 0.933** ✓ |
| worse under occlusion | asserted, never modelled | **0.0068 visible vs 0.0060 occluded** ✗ |

The magnitude taken from the literature holds, and the correlation coefficient
guessed for `obs_noise_corr` turns out to be slightly conservative rather than
invented. The occlusion claim does not survive: error under occlusion is if
anything *lower*. The likely reason is specific and worth stating — when the
hand is over the box, the hand is large, unoccluded and directly above it, so
the estimator can read the box's position off the gripper. A real system with a
wrist camera, clutter, or a moving viewpoint would not get that for free.

Closing the loop, five demonstration-seeded policies, 40 episodes each:

| | success |
| --- | ---: |
| ground-truth pose | 0.985 [0.957, 1.000] |
| **estimated pose** | **0.805** [0.678, 0.932] |

So a policy trained on perfect pose loses about 18 points when it has to use an
estimate whose error is 6.5 mm and strongly correlated in time — a larger cost
than the independent-noise model predicts, and the reason the model was worth
checking.

What this is **not**: one camera, one lighting condition, one box texture, no
domain gap, no detector failures, no clutter. It is not evidence that any of
these policies would work from real images.

**The reward is dense and hand-designed, and removing it is not recoverable
here.** The sparse version was built and run: reward 1.0 on the step the success
condition holds and 0 everywhere else, with hindsight relabelling
([Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495), future strategy,
k = 4) as the standard remedy. `src/train_her.py`, six runs of 200 000 steps:

| | per-seed | mean |
| --- | --- | ---: |
| sparse + hindsight relabelling | 0.000, 0.000, 0.000 | **0.000** |
| sparse alone (control) | 0.000, 0.000, 0.000 | **0.000** |

Zero success *and* zero grasp rate throughout: nothing learns to touch the box.
The nine shaped terms are doing essentially all the work in this task, which is
what [reward-design.md](reward-design.md) implies when it records two shapings
that failed before the current one worked.

The control is what makes that readable. Both arms at zero means the sparse task
is not being solved at this budget; had the control scored anything, the right
conclusion would have been a bug in the relabelling instead. What this does
*not* establish is that hindsight replay fails on this task in general — it is
one configuration at one budget, and the original paper uses far more compute
than a CPU evening. The relabelling itself is tested: goal entries, the derived
goal-minus-object entries and the recomputed reward, against the simulator's
true object position rather than the noisy observed one.

## The results are honest but small

**Five seeds.** Enough to report a t interval across seeds instead of an
anecdote; not enough for a tight one. Where two conditions overlap, this
repository says they overlap rather than picking the favourable framing.

**200 000 steps per run.** Chosen so the whole grid finishes in a few hours on
eight cores. It is not enough for SAC from scratch under randomisation, and the
results say so rather than quietly extending the budget for the conditions that
needed it.

About 90% of that time is gradient updates and 8% is physics — 0.54 ms per
environment step against roughly 10 ms per update — so the budget is set by the
optimiser, not the simulator. `experiments/compute_ablation.py` tests the
obvious reductions on the one condition that reliably reaches 1.000, three seeds
each:

| | success | wall | speedup |
| --- | --- | ---: | ---: |
| baseline: 128x128, batch 256, 1 update/step | 0.989 | 1064 s | 1.00x |
| 64x64 | 0.933 | 889 s | 1.20x |
| batch 128 | **0.633** | 869 s | 1.22x |
| **0.5 updates/step** | **1.000** | 548 s | **1.94x** |
| 64x64 + 0.5 updates/step | 0.989 | 446 s | 2.39x |

Halving the gradient updates is faster *and* better — 1.000 on three seeds of
three against 0.989 — which is the signature of over-updating a critic on a
small replay buffer. Halving the batch is a regression with a seed at 0.000 and
is not adopted.

**The nominal world hides a capacity loss, and nearly cost this a silent
disaster.** It saturates at 1.000, so a network too small to solve anything
harder still looks perfect there. Rechecked on `medium`, where nothing
saturates, three seeds:

| | per-seed | mean | vs baseline |
| --- | --- | ---: | --- |
| baseline, 128x128, 1 update/step | 0.13, 0.67, 0.60 | 0.467 | — |
| **0.5 updates/step** | 0.77, 0.00, 0.80 | **0.522** | t = 0.18 |
| 64x64 + 0.5 updates/step | 0.00, 0.00, 0.00 | **0.000** | t = −2.78 |

The 2.39x setting scores **1.000 on the nominal world and 0.000 on every
randomised seed**. Adopting it on the nominal result — which was the obvious
thing to do, and what the first table alone recommends — would have made every
randomised experiment afterwards produce zeros for a reason nobody would have
thought to look for.

So the adopted setting is `--updates-per-step 0.5` at the unchanged 128x128:
1.94x faster, indistinguishable from the baseline where the task is hard, and
better where it is easy. A benchmark that saturates cannot be used to approve a
reduction in capacity, which is the general form of the lesson.

**The from-scratch runs have an entropy-collapse failure mode, and it is now
fixed.** The seeds that stall settle
at an entropy coefficient around 0.025 while the seeds that solve the task sit
near 0.17. A floor under that coefficient rescues all five seeds at half the
budget (0.993 mean, 95% t [0.975, 1.000], against 0.400 and [0.000, 1.000]
without it). The investigation, including the hypothesis that turned out to be
wrong, is in [exploration.md](exploration.md).

**The fix has a hyperparameter, and the hyperparameter does not transfer.** A
floor beats no floor at every randomisation level, but the value that does it is
different for each: `none` needs at least 0.10 and fails at 0.05, `low` needs at
most 0.05 and scores 0.000 at 0.15, `medium` and `high` are indifferent between
them. The 0.15 that rescues the nominal world takes `low` to zero — it stops the
policy learning to grasp at all. So "clamp the entropy coefficient" is a real
fix for a real failure mode and *not* a setting that can be copied between
distributions, which is the most transferable thing this repository learned and
the least convenient.

Getting there took three wrong turns, all recorded in
[exploration.md](exploration.md): the floor looked inert at a matched 100 000
steps, then decisive against an unmatched baseline at 300 000 against 200 000,
then harmful at `low` when only one floor value had been tried there.

The original grid is kept and the floored grid sits beside it, at the same
200 000-step budget: `none` goes from 0.402 to 0.986 on a fixed evaluation,
`medium` from 0.220 to 0.582, `high` from 0.122 to 0.390. Read the un-floored
rows as what a standard SAC configuration does here, not as the best this
algorithm can do on this task.

Fixing the collapse does **not** fix transfer. The floored policies score 0.000
to 0.006 on `shifted`, the same as the un-floored ones, so the poor transfer in
this repository is not an artefact of an undertrained baseline — which is the
first thing one would want to check before believing it.

**Two tasks now, and the second one is where the reward-design method stopped
working.** `task="place"` is pick-and-place: carry the box to a target patch
elsewhere on the table and let go of it there. It was chosen to break the lift
task's assumptions rather than to be easy -- the goal is somewhere else on the
table so the object has to travel laterally, success requires the hand to have
*let go* (the exact opposite of lift's "still holding at the final step"), and
success requires the object to have been picked up rather than slid. The
observation and action spaces are unchanged, which is a property of the original
observation design and the only part of this that came free.

Five seeds, 100 episodes each, at the same budget and settings the lift grid
used:

| | none | shifted |
| --- | ---: | ---: |
| scripted expert | 1.000 | 0.330 |
| behaviour cloning | 0.978 [0.968, 0.988] | 0.010 |
| BC + RL | 0.916 [0.845, 0.987] | 0.112 |
| **SAC from scratch** | **0.002** [0.000, 0.008] | 0.000 |

Imitation transfers to the second task without changes. **From-scratch RL does
not**, and four reward designs and a tripled budget have not moved it off zero.
What each attempt actually did, read from behaviour rather than from the return
curve:

| `carry` gated on | `clear` pays | peak lift | success |
| --- | ---: | ---: | ---: |
| `grasped` only | 0.18/step | 0.010 m | 0.002 |
| the binary lift latch | 0.18/step | 0.006-0.016 m | 0.000 |
| clearance / 4 cm | 0.18/step | 0.009 m | 0.000 |
| clearance / 4 cm | **0.48/step** | 0.010 m | 0.002 |

The first design paid the largest term in the reward to a policy that closed the
pads on the box and *pushed*; the second closed that exploit and replaced it
with a cliff the policy never crossed; the third turned the cliff into a hill
and the fourth made the hill as steep as the lift task's own height term
(4.0 x 0.12, taken from there rather than searched for, because getting the box
off the table is the identical sub-problem). The first two failures are exactly
what [reward-design.md](reward-design.md) already records for the lift task, met
again on a task written after they were written down.

The budget control excludes the boring explanation. The first design at 600 000
steps, three seeds -- more gradient updates than the entire lift grid was
trained with -- finishes at 0.000, 0.067, 0.067, still grasping on 73-97% of
steps and still with a peak lift of 0.021 m. It is sliding the box, faster.

Demonstrations solve the same task on the first attempt, on both reward
versions: 0.978 cloned, 0.916 and 0.870 demonstration-seeded.

The most likely structural reason, and it is a hypothesis rather than a result:
in the lift task the shaped progress term and the height term point the same
way, so climbing one climbs the other. In the place task they are orthogonal --
the object must go *up*, which earns little, before going *sideways*, which
earns most of the reward. Demonstrations supply exactly that missing sequence,
which is consistent with cloning working on the first attempt.

**One box shape by default**, and see the shape paragraph above for what happens
with three.

**`shifted` is a proxy, not a robot.** See [sim-to-real.md](sim-to-real.md). No
hardware was involved at any point.

## The Isaac Lab port works, but produces no headline number

It was brought up against Isaac Sim 5.1.0 and Isaac Lab 2.3.2 on an RTX 4060,
and all seven of its bring-up checks now pass at both `none` and `medium`
randomisation. That includes the one the port exists for: the reward computed
inside Isaac on the GPU agrees with the shared numpy implementation to about
5e-08 on the same states, so "it is the same task" is measured rather than
asserted. Randomisation is driven through Isaac's event manager from the same
JSON ranges, with the same interval arithmetic.

What it does *not* do is produce any number quoted in the README's headline
tables; every success rate there was produced in MuJoCo. It does now carry its
own five-seed grids: from scratch against demonstration-seeded on the nominal
world (0.000 against 0.969), the entropy floor against its control (0.194
against 0.463, t = 1.01 — not separated), and a randomised grid at `medium`
(0.275 [0.182, 0.368] at 4 000 steps, 0.131 [0.000, 0.272] at 15 000), and two
floor sweeps. The randomised runs get *worse* with more training, and the
mechanism is now known without the cause being known: the entropy coefficient
collapses to 0.0011 and critic loss grows twentyfold, and clamping the
coefficient prevents the critic blow-up on every seed while making success worse
(0.010 against 0.131, Welch t = −2.33). A real, linked, fixable pair of symptoms
that is not what makes randomisation expensive here.

The floor result turned around once the value was swept rather than assumed.
Carrying MuJoCo's 0.15 across gave 0.463 against a 0.194 control and read as a
failed replication; at 0.30 it is 1.000 on three seeds of three, t = 4.71 — the
largest effect in this repository. Both engines therefore show the failure and
the fix, and the value belongs to the distribution: 0.05 at MuJoCo's `low`, 0.15
at MuJoCo nominal, 0.30 here, with 0.05 scoring zero here.

All eleven MuJoCo-side randomisation parameters that predate this work are
mapped in Isaac; the two sensing parameters added since (obs_noise_rot,
obs_noise_corr) are MuJoCo-only. Two of the mapped eleven are
*analogues* rather than translations and are labelled as such: hand compliance
is the `solref` of a MuJoCo weld that has no counterpart here, so it maps to the
arm's joint stiffness with the sign inverted; and gravity is per-scene in Isaac
rather than per-environment, so all environments share one draw where MuJoCo
takes one per episode. Same intervals, coarser granularity. The mapping table is
in `envs/isaac/README.md` rather than quietly omitted.

**Cross-simulator transfer has been measured properly, and it is poor.** Twenty
policies — five seeds at each of four randomisation levels — run in Isaac with
no adaptation score 0.05 to 0.08, with every interval including zero, against
1.000 for the scripted expert in the same environment. No randomisation level
helps reliably.

The first version of this measurement used one seed per level and appeared to
show wide randomisation transferring at 0.41–0.50. It does not: that was seed 0
of `high`, and the other four seeds scored zero. The single-seed reading was
wrong in exactly the way this repository warns about everywhere else, which is
why the five-seed version replaced it.

## Things that would be next, in order of value

This list has been rewritten twice as items came off it. Three of the original
six are now done and are recorded above rather than here: the yaw-alignment
reward term (built, in two placements, and it does not rescue the wrist), a
perception stack (built, and it refuted one of the three claims this file made
about sensing error), and a sparse-reward variant with hindsight replay (built,
and it scores zero with its control also at zero).

1. **Why the second task needs demonstrations and the first does not.** The
   place task is solved by cloning at 0.978 and by demonstration-seeded RL, and
   from-scratch RL has now been through four reward designs and a 3x budget
   without clearing zero. The lift task's from-scratch runs work. Something
   about the longer chain -- most likely that lifting and transporting are
   orthogonal in the place task where they are collinear in the lift task -- is
   the actual obstacle, and it is more interesting than anything else on this
   list.

2. Measured randomisation ranges from real hardware. The guessed ones have now
   been checked against published measurements rather than defended
   ([randomisation-sources.md](randomisation-sources.md)): they are optimistic
   on latency by 2-5x, optimistic on sensing, and omitted orientation error
   altogether. Real hardware would still be better than a literature survey.

3. **Why cross-simulator transfer fails.** The evaluation exists and the answer
   is not a control-gain constant -- seven scalings of the action, in both
   directions, none of which clears its interval. Halving lateral commands alone
   raises peak lift to what a successful MuJoCo policy reaches while success
   does not move, which points at contact rather than reaching.

4. A wrist camera rather than a fixed one, with clutter. The fixed camera's
   error turned out to be *lower* under occlusion because the gripper is itself
   a cue; a moving viewpoint would remove that and is the honest version of the
   test.

5. Grasp-point selection -- choosing *where* to grasp an unfamiliar shape --
   which is what most of the grasping literature is about and which nothing here
   demonstrates.

6. An Isaac port of the place task. The reward is written backend-agnostically
   for it; the port has not been done, and until it is, "the second task" is a
   MuJoCo-only result.
