# Limitations

Written to be read before the results, not after them.

## The environment is a simplification, in specific ways

**The MuJoCo default is a hand with no arm behind it.** It is a free body
dragged by a mocap weld: no joint limits, no self-collision, no arm inertia, no
singularities, no reachability constraint. Every one of those is a real failure
mode in a real cell, and every headline number in this repository was produced
that way. A policy trained on the default needs a reachability check and a
joint-limit clamp between it and any servo loop. (The Isaac port *does* have a
real Franka, which is how its near-singular start pose was found.)

There is now a six-jointed arm variant as well, and **it is measured rather than
offered** — the next section is what happens when policies are trained through
it, and the short version is that from-scratch RL never closes the fingers at
all. That is the honest cost of the abstraction the defaults use.

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

0. **A sentinel used as an index, found months later by a video.** Not one of the
   four below, and the worst of them. When the handled shape was added the arm
   scene had no handle geom, so its id was set to `-1` as a sentinel — and then
   used to write `geom_contype[-1] = 0`, which numpy reads as the *last* geom in
   the model. In the arm scene that was the table. The object fell straight
   through it, four steps to a "dropped" episode, with nothing in the arm's own
   code touched. It surfaced because a README clip came out six frames long
   instead of a hundred and two. Every arm number in this section predates the
   bug and was re-verified after the fix: the expert scores 0.690 [0.594, 0.772]
   against the 0.680 [0.583, 0.763] recorded here.


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

**Policies have now been trained on it.** Five seeds, 100 episodes each, the
same 200 000 steps, network and entropy floor the weld runs used:

| | none | shifted |
| --- | ---: | ---: |
| scripted expert | 0.680 | 0.000 |
| behaviour cloning, `low` demonstrations | 0.202 [0.175, 0.229] | 0.000 |
| behaviour cloning, nominal demonstrations | 0.448 [0.246, 0.650] | 0.000 |
| **SAC from scratch** | **0.000**, grasp rate 0.000 | 0.000 |

**From-scratch RL through six joints never closes on the box at all** — grasp
rate 0.000 across every seed, not a low success rate on top of a working grasp.
The weld version at the same settings reaches 0.593. Whatever the mocap weld is
abstracting away, exploration is where the cost lands, and that is the cleanest
single statement about the abstraction the rest of this repository trains under.

**Where the demonstrations are recorded matters more than anything else tried
here.** The arm's expert succeeds on 19% of `low` episodes against the weld
expert's ~100%, so a `low` set is both a weaker teacher and a biased sample of
the easy worlds — 200 kept episodes out of 1054 attempted. Recording on the
nominal world instead, where the same expert manages 0.671, more than doubles
the clone: 0.202 to 0.448, Welch t = −3.35. For the weld this choice is free,
because its expert succeeds everywhere.

### Fine-tuning the arm's clone: four diagnoses, and two of my own errors

Demonstration-seeded RL made the clone *worse*, which is the opposite of what it
does through the weld. Four one-flag variants, all from the same clones, all
evaluated on the same distribution:

| | success on `none` | vs the clone |
| --- | ---: | ---: |
| clone, no RL at all | 0.448 | — |
| **BC term never decays** | **0.536** [0.415, 0.657] | +0.088, t = +1.04 |
| fine-tune at `none`, where the demonstrations came from | 0.422 [0.339, 0.505] | −0.026, t = −0.33 |
| standard fine-tune, at `medium` | 0.298 [0.207, 0.389] | −0.150, t = −1.88 |
| critic warmup 3 000 → 20 000 | 0.228 [0.070, 0.386] | −0.220, **t = −2.38** |

Nothing here beats the clone by a margin that clears its interval. What the
table does establish is where the damage comes from, and it is *not* fine-tuning
as such: leash the actor to the demonstrations for the whole run and the result
is the best of the five; match the fine-tuning distribution to the demonstrations
and the loss disappears; give the critic seven times longer to warm up — the
change that sounded most like good practice — and it is the only arm that
separates from the clone, downwards.

**Two corrections belong in this section**, because both were mine and both
would have been reported as findings.

The first: I compared the clone's score on `none` against fine-tuned scores on
`medium` and called it destruction of a working policy. The clone scores **0.030
on `medium`**. There was never a working policy on the fine-tuning distribution
to destroy.

The second: I read the same variants' *training-level* evaluations, saw
0.033-0.067, and concluded the BC leash had not helped. On the common evaluation
it is the best arm on the board. Both mistakes have the same shape — comparing
numbers measured on different distributions — and it is the mistake this
repository warns about everywhere else.

Randomisation parameters are applied to the arm unchanged apart from
`hand_compliance`, which maps to arm joint stiffness the way the Isaac port maps
it.

**The hand cannot rotate — by default.** The pads close along world *x*, so a
square box at 45° of yaw presents √2 times its side and above roughly 27 mm of
half-size it is ungraspable at the worst yaw regardless of the policy. Object
sizes are therefore capped at 24 mm half-size, and at that cap yaw cannot bind.

`make_env(..., wrist=True)` adds the yaw degree of freedom (34-D observation,
5-D action), and the honest summary is that it helps the *scripted* policy and
defeats *from-scratch* RL -- a narrower claim than the one this section used to
make, and the difference is the whole of the subsection below. On boxes spanning
15–35 mm, the expert with a wrist
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

#### Demonstrations do rescue the wrist, and the anchor has to be held

Everything above tests *from-scratch* RL. Nobody had put demonstrations through
the wrist, and the zero was being read as a statement about the joint when it was
a statement about exploration. 200 demonstrations were recorded with the wrist
(`demonstrations/expert_wrist.npz`, expert 0.548 while recording); five seeds,
200 000 steps, 100 evaluation episodes each at `wrist_bench`:

| | per-seed | mean | 95% CI |
| --- | --- | ---: | :---: |
| from-scratch SAC, no wrist | 0.333, 0.033, 0.000 | 0.122 | -- |
| from-scratch SAC, with wrist | 0.000, 0.000, 0.000 | 0.000 | -- |
| behaviour cloning | 0.42, 0.40, 0.40, 0.37, 0.40 | 0.398 | [0.376, 0.420] |
| BC + RL, anchor decayed | 0.35, 0.28, 0.36, 0.39, 0.33 | 0.342 | [0.291, 0.393] |
| **BC + RL, anchor held** | 0.46, 0.50, 0.46, 0.51, 0.46 | **0.478** | [0.447, 0.509] |

The wrist is learnable. What it is not is *discoverable* -- SAC never finds the
alignment on its own, and no shaping term tried here made it find one.

**And learnable is not the same as useful -- but measuring that took three
attempts, and the first two were wrong in opposite directions.**

The table above has no control in it: every row has the wrist. The first
correction added a no-wrist row and reported a rout, 0.838 against 0.478. That
control was not one. `GraspEnv` sets the object size cap *from the wrist flag* --
0.034 m with a wrist, 0.024 m without -- so it compared a hand that rotates on
boxes up to 34 mm against a hand that cannot on boxes capped at 24 mm. The
comment above that line says it is overridable "so the wrist can be ablated
properly ... or the comparison is between two different tasks", and until this
session nothing had ever passed the override. Every wrist number in this
repository's history was measured that way.

Matched on the cap, same demonstration budget, same level, 100 episodes over
five seeds:

| | wrist | no wrist | delta |
| --- | ---: | ---: | ---: |
| **cloning**, cap 0.034 (yaw binds) | 0.398 [0.376, 0.420] | 0.432 [0.361, 0.503] | -0.034 |
| **cloning**, cap 0.024 (yaw never binds) | 0.792 [0.745, 0.839] | 0.838 [0.811, 0.865] | -0.046 |
| **held-anchor RL**, cap 0.034 | 0.478 [0.447, 0.509] | 0.464 [0.424, 0.504] | +0.014 |
| **held-anchor RL**, cap 0.024 | 0.892 [0.838, 0.946] | 0.818 [0.808, 0.828] | **+0.074** |

The scripted expert agrees: matched at 0.034 it scores 0.548 with the wrist
against 0.506 without, and at 0.024 it is 0.980 against 0.855. Nothing like the
gap the unmatched comparison produced, in either direction.

So the answer is small, and its sign depends on the method. **Cloning is
slightly worse with the wrist** at both caps -- a fifth action dimension is one
more thing to imitate from the same 200 episodes -- and both intervals overlap.
**Held-anchor RL is slightly better with it**, and at the small-box cap the
intervals are disjoint (0.892 against 0.818), the only cell here where the joint
clearly earns anything. The plausible reading is that yaw improves grip quality
even when the box would fit unaligned, and that RL can exploit that where
imitation of a five-dimensional action from a fixed budget cannot.

What does not survive is the headline. "The wrist does not help" is defensible
as a summary of four small deltas straddling zero. It is not defensible as the
two-to-one rout published here for a few hours.

The older from-scratch claim carried the same confound, and matching it changes
what the number was about. "0.000 with the wrist against 0.122 without" compared
a wrist on boxes up to 34 mm against no wrist on boxes capped at 24 mm. Matched,
five seeds, 200 000 steps with the entropy floor:

| from scratch | wrist | no wrist |
| --- | ---: | ---: |
| cap 0.034 (yaw binds) | 0.000, grasp 0.45 | 0.000, grasp 0.49 |
| cap 0.024 (yaw never binds) | 0.158 [0.000, 0.424] | 0.424 [0.105, 0.743] |

**Both hands score zero on the big boxes.** The zero was never about the wrist:
it is what from-scratch RL does when the object is large enough that yaw
matters, and it does it whether or not the hand can yaw. Both variants still
*grasp* -- 0.45 and 0.49 -- so the failure is in lifting and holding a box near
the limit of the pads, not in reaching it.

At the small cap both learn, the wrist is lower, and the intervals overlap so
heavily (0.158 [0.000, 0.424] against 0.424 [0.105, 0.743]) that the honest
statement is five seeds is not enough to separate them. What can be said is that
the fifth dimension does not help exploration, which is the weaker version of
what this section used to claim.

This is recorded at length because the error was made twice, in opposite
directions, and both times the wrong number was the more dramatic one. First a
conclusion was drawn from three arms that all had the wrist, which established
only that the wrist beats its own baseline. Then the control meant to fix that
silently changed the task. A comparison is only a comparison if the thing not
being tested is held still, and a flag that moves with the thing being tested is
the hardest way to get that wrong.

The middle row is the interesting one. Decaying the cloning anchor is **worse
than not doing the RL at all**, and the failure has a visible timestamp: those
runs peak at 0.533-0.633 and fall to 0.367-0.500 at exactly the step where
`--bc-decay-steps` retires the anchor. Holding it (`--bc-decay-steps 0`) removes
the drop and is the only arm that beats cloning.

This is the same effect the Isaac grid and the arm fine-tuning both hit, and it
is now recorded once as a cross-cutting result rather than three times as a
local quirk:

| held vs decayed anchor, 5 seeds | decayed | held |
| --- | ---: | ---: |
| six-jointed arm, level `none` | 0.176 | **0.536** |
| wrist, `wrist_bench` | 0.342 | **0.478** |

Neither pair's intervals overlap. The mechanism is not mysterious -- the anchor
is the only thing keeping the policy in the part of the space the demonstrations
cover, and the entropy term walks it out as soon as the anchor stops paying --
but it does mean the decaying schedule, which is the more common default, is the
wrong one on every variant measured here.

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

### The arm: a grid, and the bug that was most of the old one

Every headline number in this repository was produced on the welded hand. The
arm now carries its own four-level grid -- and the first version of that grid
was mostly measuring a broken actuator.

**The bug.** A MuJoCo position servo produces
``gainprm[0] * ctrl + biasprm[1] * qpos``, and it is only a position servo while
those are equal and opposite. This file's own code says so, in a comment above
the *gripper*, where both terms are set. The arm set only the gain. So when
`hand_compliance` scaled the arm's stiffness, the bias kept its original value
and every joint settled at ``(new/old)`` times its commanded angle -- up to 43%
off at `medium`. The hand stopped 143 mm short of the box and the scripted
expert simply never reached it.

Finding it took a one-parameter-at-a-time sweep, because the symptom pointed
elsewhere. Every other randomisation parameter left the expert's closest
approach at about 50 mm; `hand_compliance` alone pushed it to 152 mm. Two
plausible explanations were tested first and both were wrong, which is worth
recording because both were the obvious guess:

* **not a time limit** -- best reach is identical at 100, 200 and 400 steps, so
  the arm converges and then sits there. That is a steady-state error;
* **not gravity droop** -- hand-rolled gravity compensation made it *worse*
  (0.650 to 0.050 at `none`, because the servo was tuned with the load present),
  and MuJoCo's own `body_gravcomp` changed nothing at all.

Scripted expert through the arm, 40 episodes:

| | before | after |
| --- | ---: | ---: |
| `none` | 0.633 | 0.650 |
| `low` | 0.200 | **0.575** |
| `medium` | 0.133 | **0.525** |
| `high` | 0.133 | **0.350** |

**The grid, retrained on the corrected servo.** Five seeds, 100 episodes:

| trained and evaluated at | cloning | + held anchor | `shifted` |
| --- | ---: | ---: | ---: |
| `none` | 0.402 [0.264, 0.540] | **0.530** [0.484, 0.576] | 0.010 |
| `low` | 0.502 [0.469, 0.535] | 0.452 [0.344, 0.560] | 0.054 |
| `medium` | 0.398 [0.378, 0.418] | 0.388 [0.382, 0.394] | 0.104 |
| `high` | 0.354 [0.327, 0.381] | 0.354 [0.312, 0.396] | 0.116 |

The previous version of this section reported 0.522, 0.158 and 0.052 and
concluded that the arm "does not survive randomisation". **That conclusion was
wrong, and it was wrong because of the bug.** Corrected, the arm declines mildly
across the whole range -- 0.530 to 0.354 -- rather than collapsing tenfold.

The second thing the bug hid is arguably more interesting. Every arm row used to
score **0.000** on the held-out `shifted` distribution, at every training level.
It now *rises with training randomisation*: 0.010, 0.054, 0.104, 0.116. Training
wider transfers better, which is the entire premise of domain randomisation, and
the broken servo had erased the effect completely. The weld shows the same
ordering; the arm now agrees with it.

What is still true: the arm is worse than the weld. 0.530 against 0.973 at
`none` is roughly half, and the remaining gap is now *located* rather than
attributed to the abstraction in general.

The scripted expert through the arm scores 0.650 at `none`, and its
ever-grasped rate is **also** 0.650 -- identical, on 40 episodes. Whenever the
arm grasps, it succeeds; the whole deficit is episodes where it never closes on
the box. Splitting those by where the hand ends up:

| | closest lateral approach | closest vertical |
| --- | ---: | ---: |
| succeeded | 0.0007 m | 0.0208 m |
| failed | 0.0032 m | **0.1157 m** |

The failures are laterally aligned to 3 mm and stall **116 mm above** the box.
The arm reaches the right place over the table and does not descend.

Three explanations were tested and rejected. It is not IK precision: sweeping
the damped-least-squares solver from (0.08, 20 iterations) down to (0.01, 100)
makes it monotonically *worse*, 0.633 to 0.500, so the shipped setting is
already the best of those tried. It is not workspace: successes average 0.726 m
from the arm base and failures 0.725 m, with fully overlapping ranges. And it is
not lateral alignment, per the table above.

Two further explanations were tested and also rejected. It is **not collision**:
counting arm-to-table contacts over 50 episodes gives 0.0 per episode for both
successes and failures, so the collision-blind IK is not folding the arm through
the table on the way down. And it is **not the solver leaving the feasible set**:
failing episodes do spend 87.4 frames per episode pinned at a joint limit
against 22.8 for successes, which looked like the cause, but clamping the IK
iteration inside the joint ranges -- so it searches only reachable
configurations instead of relying on the actuators to clip afterwards --
produced *identical* success at every level (0.650 / 0.575 / 0.525 / 0.350).
The limit-pinning is a symptom of a stalled descent, not its cause, and the
change was reverted rather than kept as a no-op.

It is also **not the expert's logic** -- the scripted controller enters all four
of its phases in failed episodes exactly as in successful ones, so it commands
the descent and the arm does not follow -- and **not time**: closest vertical
approach is identical at 100, 150, 250 and 400 steps (0.0535 m), so the arm
converges to a pose above the box and stays there.

So: a located failure with six eliminated causes and no established one. The
arm reaches over the box, aligns laterally to 3 mm, is commanded down, and
settles ~50 mm high in about a third of configurations. Not IK precision, not
workspace, not lateral alignment, not collision, not infeasible solver iterates,
not episode length. What remains is a steady-state property of this arm, its
damped-least-squares controller and its servo gains together, and separating
those three needs a different experiment than any run here. Every number produced on the weld should be read as an upper bound
on what the same recipe does through six joints, IK, joint limits and
self-collision. But "an upper bound about twice as high" is a different claim
from "the arm falls apart", and only the first one is supported.

### Perception in the loop: the estimator is a pipeline now

The section above measures what perception *costs* a policy trained on ground
truth: 18 points, by substituting the CNN's estimate at evaluation. A robot has
never had ground truth, so the question it actually asks is different -- what
does a policy trained *through* the camera reach?

`envs/mujoco/perception_env.py` wraps the environment so the object's position
comes from a 64x64 render at training and evaluation alike, sharing substitution
indices with `experiments/perception_eval.py` so the two stay comparable.
`--perception` runs through `record_demos`, `train_il`, and `evaluate`, and
`evaluate` reads it from the run's own config so a policy trained through the
camera cannot accidentally be scored on true state.

200 demonstrations, five seeds, 100 episodes, camera in the loop throughout:

| | success |
| --- | ---: |
| scripted expert, through the camera | 0.971 |
| **cloning, trained and evaluated through the camera** | **0.934** [0.900, 0.968] |
| cloning on ground truth, for reference | ~0.973 |

So the pipeline works, and it costs about four points rather than eighteen.
Training through the estimator is not the same problem as inheriting it.

**The hard camera, and a noise model that turned out to be harsher than the
estimator it was built from.**

The paragraph above used to end by conceding that the camera here is fixed and
unoccluded, and that the wrist camera with clutter -- 0.0513 m error -- is the
realistic case that nothing survives. That has now been run, and it changes the
sensing story in this repository.

The wrist estimator in the loop makes the error the section above quotes:
0.0499 m measured over 2 000 frames, against the 0.0513 m on record. It is not
better near the object (0.0491 m inside 5 cm) and it is not a simple offset
(bias is 0.36 of the scatter). It is, however, a **function**: a CNN returns the
same wrong pose for the same scene every time.

That distinction turns out to be worth more than the magnitude. Five seeds,
100 episodes, privileged demonstrations throughout:

| pose error of about 0.05 m, from | dynamics | success | grasp |
| --- | --- | ---: | ---: |
| the CNN, in the loop | nominal | **0.960** [0.937, 0.983] | 0.99 |
| the CNN, in the loop | `measured_camera`'s | **0.728** [0.677, 0.779] | 0.93 |
| injected random noise | `measured_camera`'s | 0.406 [0.345, 0.467] | 0.70 |

The middle and bottom rows are the controlled pair, and they are the point. Same
friction 0.46-1.9, same mass 0.5-2.0, same 1-6 step latency, same compliance and
gripper gain. The only difference is where the pose error comes from, and the
intervals are disjoint: **0.728 against 0.406**.

`measured_camera` draws its error from a band calibrated on this estimator's
*magnitude*, redrawn every episode. The estimator's error is instead a
deterministic distortion of the scene, and a policy trained through it learns to
invert one. Matching the magnitude and discarding the structure made the level
substantially harsher than the thing it was modelling.

That control needed building: `measured_camera_realsensor` is `measured_camera`
with the injected sensing noise removed, so the CNN supplies the only pose
error. Without it the comparison was the CNN at nominal dynamics against
injected noise at full randomisation, which is sensing *and* five dynamics
parameters at once -- and it flattered the conclusion by about twenty points.

What this does **not** say is that real cameras are easy. 0.728 is still short
of 0.973 on ground truth, so perception costs about a quarter of the task even
when its error is learnable. And a physical camera has error sources this one
does not -- calibration drift, changing light, motion blur, a moving base --
several of which are closer to a fresh random draw than to a repeatable
function. The honest reading is narrower than "sensing is fine": **a noise model
calibrated on error magnitude alone overstates the damage**, because the
structure it throws away is the part a policy can learn around.

**Unfreezing the estimator buys nothing, and the reason is worth more than the
result.** The obvious next step was DAgger for perception: an estimator is only
accurate where it saw data, the shipped one saw scripted-expert and random
trajectories, and a trained policy visits different states. So 12 000 frames
were collected from a trained policy's own distribution -- rolled out *through*
the camera, since feeding it ground truth would sample states it never actually
visits -- and the estimator was retrained on those plus the original 20 000.

| estimator | error | success |
| --- | ---: | ---: |
| frozen, expert and random states | 0.0499 m | 0.728 [0.677, 0.779] |
| retrained on policy states | 0.0477 m | 0.734 [0.687, 0.781] |

The error improves by 4% and the policy by 0.006, which is nothing: the
intervals overlap almost entirely. The collection report says why. On-policy
frames flag **0.0%** as hard, against a mixture in the original set -- a trained
policy keeps the box in view, so the states it visits are the *easy* ones.
DAgger's usual premise is that the learner wanders somewhere the demonstrator
never went; here it wanders somewhere better, and retraining on that adds data
the estimator had already mastered.

So "the estimator is frozen" was a real caveat and it is now a measured
non-issue at this operating point. What would move the number is a harder
estimator problem rather than a better-matched one.

Two caveats remain, and they matter more than the number. The camera was **fixed and
unoccluded** here; the wrist camera with clutter is measured directly above. And the
error is small enough to servo on: at 0.0066 m the scripted expert works
unchanged, which is why this needed ordinary demonstrations where
`measured_camera` needed privileged ones.

**From-scratch RL through the camera works, and the reason it had not been run
was not the one given.** This section used to end by declining the experiment on
cost: a step costs 75.9 ms against roughly 0.3 ms for the state environment, so
200 000 steps through a renderer is over four hours. That arithmetic is right
and it was not the obstacle. `train_rl.py` had no `--perception` flag at all --
the wrapper was plumbed into recording and cloning only -- so the run was not
expensive, it was impossible. With the flag, 150 000 steps finishes comfortably:

| through the fixed camera, no demonstrations | success |
| --- | ---: |
| **SAC from scratch, camera in the loop** | **0.950** [0.875, 1.000] |
| cloning through the same camera | 0.934 [0.900, 0.968] |
| cloning on ground truth, for reference | ~0.973 |

Higher than cloning and level with ground truth. A policy can learn this task
from CNN pose estimates without ever seeing the simulator's state, and without a
demonstrator.

**And the randomised levels have now been tried through it, which is where the
camera stops.** Three seeds, 120 000 steps, evaluated at `medium`:

| at `medium` | success | grasp rate |
| --- | ---: | ---: |
| ground-truth state | 0.582 | -- |
| camera in the loop | **0.000** [0.000, 0.000] | 0.55-0.65 |

Zero on every seed, and the grasp column says it is not a perception failure:
the policy closes on the box in more than half of episodes, so it is finding the
box through the estimator and then failing to lift and hold it under randomised
mass, friction and latency. Perception and randomisation together cost far more
than either alone -- ground truth loses 0.39 over that range while the camera
path loses everything.

That is a third distinct failure mode in this document, and the three are worth
separating because they look identical in a success column: the arm at `none`
cannot *reach* (stalls 116 mm up), the sparse recipe at `medium` cannot *grasp*
(0.13 at best), and the camera at `medium` grasps at 0.55-0.65 and cannot
*hold*. Each needs a different fix and none of them is the reward.

The speed figure governs what is comfortable rather than what is possible:
150 000 steps is about three hours a seed, so the camera path is not the default
for RL.

### Grasp-point selection

`make_env(..., handled=True)` puts out a shape whose reported pose is not a
graspable point: a 96 mm cube, ungraspable along every axis against a 78 mm pad
gap, with a 20 mm handle offset 118 mm out and 34 mm up. The observation reports
the **body frame**, which sits on the cube. The handle's direction is recoverable
from the reported orientation, so the information is present and has to be used.

The environment is measured rather than asserted, and the separation is total:

| scripted expert | success |
| --- | ---: |
| aiming at the reported pose | **0/30** |
| aiming at the handle | **30/30** |

Same hand, same episodes, same everything else. Five seeds, 100 episodes, the
settings the rest of the repository uses:

| | `none` | `shifted` |
| --- | ---: | ---: |
| behaviour cloning | 0.896 [0.804, 0.988] | 0.146 [0.080, 0.212] |
| **BC + RL** | **0.996** [0.989, 1.000] | 0.010 |
| SAC from scratch | 0.128 [0.094, 0.162] | 0.008 |

**Selection is learnable, and demonstrations are not required for the selection
itself.** The from-scratch arm was predicted, in writing in
`experiments/grasp_point.py` before it ran, to score zero — the reward's reach
term pulls towards the reported pose, which is exactly the wrong place. It does
not score zero. It grasps on 80-100% of steps, which on this shape means it finds
the handle *against* a reward gradient pointing elsewhere. Success stays at 0.128
because holding a 100 g cube by a thin offset handle is a poor grasp, not because
the selection was not learned. That is a better result than the prediction and
the prediction is left on the record.

Cloning reaches 0.896 and demonstration-seeded RL 0.996 — the highest number on
any task in this repository, which is worth reading carefully: the handled shape
has one graspable point, so once found there is nothing left to choose. It is a
test of *finding* the grasp, not of choosing among several.

Transfer is poor as everywhere else, and the clone transfers best (0.146 against
0.010) — the same ordering the lift task shows.

No bottles, no bags and no bin. Clutter now exists — three free distractors of
similar size and colour, `make_env(..., clutter=3)` — and so does an environment
that *requires* grasp selection: `make_env(..., handled=True)` puts out a 96 mm
cube, ungraspable along every axis, with a 20 mm handle offset 118 mm from the
body frame the observation reports. A policy that goes to the reported position
and closes scores 0/30 on it.

What is still missing is anything *solving* that environment. No scripted expert
manages it yet, so there are no demonstrations, and this repository shows no
learned grasp-point selection — which is what most of the grasping literature is
actually about.

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
the estimator can read the box's position off the gripper.

**That explanation has since been tested rather than left as a story.** A camera
mounted on the palm cannot be occluded by the hand it is bolted to, so it removes
that cue by construction; three distractor objects of similar size and colour
were added at the same time. Same network, same training, same episode-level
split:

| | fixed camera | wrist camera, 3 distractors |
| --- | ---: | ---: |
| mean position error | 0.0065 m | **0.0513 m** |
| on easy frames | 0.0068 m | 0.0487 m |
| on hard frames | **0.0060 m** | **0.0778 m** |
| relationship | hard frames *better* | hard frames **60% worse** |
| lag-1 autocorrelation | 0.947 | 0.950 |

The relationship reverses, which is what the explanation predicted. Take away the
gripper-as-a-cue and the intuitive result appears: the hard frames really are
harder. "Hard" is not the same flag for both cameras and cannot be — a wrist
camera is never occluded by its own hand, so there it means the object has left
the frustum. Calling both "occlusion" would compare two quantities and report
them as one.

The second number in that table matters more than the reversal. **A realistic
viewpoint with clutter is eight times worse than the fixed camera and five to
thirteen times outside the range `measured.json` models** (0.004-0.010 m, from
published YCB-Video results). The sensing randomisation in this repository is
optimistic, and now measurably so rather than suspectedly.

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

### Sensing: the randomisation is optimistic by the margin that matters

`measured.json` randomises object-position error over 0.004-0.010 m, taken from
published YCB-Video results. That range is right for the *fixed* camera measured
above (0.0065 m). It is not right for a wrist camera with clutter, which is the
realistic case and which this repository's own estimator delivers at 0.0513 m.
`measured_camera.json` is that level. Ten seeds, 100 episodes:

| | `none` | `measured` | **`measured_camera`** |
| --- | ---: | ---: | ---: |
| BC + RL | 0.973 | 0.197 | **0.001** |
| SAC + entropy floor | 0.587 | 0.115 | **0.003** |
| scripted expert | 30/30 | 27/30 | **7/30** |

**Every policy in that table collapses to zero under the sensing error its own
perception stack produces**, and so does the scripted expert, which has no
learning to blame. Training directly against that level does not rescue it
either: five demonstration-seeded seeds at 200 000 steps finish at 0.000 with
grasp rates of 0.00-0.43.

#### What does work: a demonstrator that can see, teaching a policy that cannot

Two fixes were tried. The first was the obvious one and it failed:

**An observation window does not help.** `make_env(..., history=4)` stacks four
frames so the network can filter. Three seeds, 200 000 steps, evaluated at
`measured_camera`: **0.000, 0.000, 0.000**, grasp rate 0.023. The reason it
cannot work is in the noise model rather than the network. The error is
correlated in time at **0.947** -- measured from this repository's own
estimator (§ the perception table above), implemented in
`GraspEnv._sensor_noise` -- so four
consecutive frames are four copies of nearly the same wrong number, and
averaging them removes almost none of it. A window filters *independent* noise;
this noise is not independent, and that was measurable before the runs rather
than after.

**Privileged distillation does work.** The demonstrator reads the true state,
the transition stores the noisy observation the policy will actually get
(`GraspEnv.clean_observation()`, `record_demos.py --privileged`). The policy
never sees privileged input at training or evaluation time -- only the
demonstrator does, and only while recording. 200 episodes, five seeds, 100
evaluation episodes each, all at `measured_camera`:

| | per-seed | mean | 95% CI |
| --- | --- | ---: | :---: |
| SAC + 4-frame window, from scratch | 0.00, 0.00, 0.00 | 0.000 | [0.000, 0.000] |
| **privileged cloning** | 0.43, 0.38, 0.34, 0.47, 0.41 | **0.406** | [0.345, 0.467] |
| privileged cloning + held-anchor RL | 0.35, 0.31, 0.30, 0.41, 0.26 | 0.326 | [0.255, 0.397] |
| privileged demonstrator (reference) | -- | 0.877 | -- |
| ordinary demonstrator (reference) | -- | 7/30 | -- |

A control matters here too, and this one survives it. "Privileged" has to be
doing the work, rather than merely "demonstrations at this level", so the same
200 episodes were recorded with an *ordinary* demonstrator -- one that reads the
same noisy observation the policy will get -- and cloned identically:

| at `measured_camera`, 5 seeds, 100 episodes | mean | 95% CI | grasp |
| --- | ---: | :---: | ---: |
| ordinary demonstrations | 0.204 | [0.136, 0.272] | 0.570 |
| **privileged demonstrations** | **0.406** | [0.345, 0.467] | 0.698 |

Privileged access roughly doubles it and the intervals are disjoint, so the
mechanism is the demonstrator's sight rather than the presence of
demonstrations. Note the control is also not zero: an ordinary clone reaches
0.204 where this repository previously recorded ~0.00 for everything at this
level, so part of that old zero was the *training method* rather than the
sensing.

Three things are worth separating here. The task is **doable** at this sensing
level -- 0.877 for a demonstrator that can see, against 7/30 for one that cannot,
so what the noise destroys is the *servoing*, not the physics. A policy given
only the noisy view reaches **0.406**, which is the first non-zero number this
repository has at `measured_camera` and moves the claim from "nothing works" to
"open-loop reaching does not work". And RL on top does **not** improve on the
clone: 0.326 against 0.406, intervals overlapping, so the honest reading is no
gain rather than a loss.

What this does not do is close the gap. 0.406 is well short of the 0.973 the
same pipeline reaches at `none`, and the remaining distance is the part of the
task that genuinely needs sensing this camera does not deliver. The claim that
changes is the absolute one: realistic sensing is not an extinction event, it is
roughly a 60% cut.

So the gap is not a caveat, it is the difference between working and not working.
Every transfer number in this repository was produced under sensing five to
thirteen times better than the estimator in the same repository provides.

### Contact: the grip does not survive in Isaac, and that is the transfer failure

`scripts/contact_probe.py` runs one protocol on both engines -- close at full
command, settle, then raise the hand -- because the cheaper explanations for poor
cross-simulator transfer are all excluded (§ the transfer section: not control
gain, not grip force, not friction, not vertical positioning).

| | MuJoCo | Isaac |
| --- | ---: | ---: |
| pad contacts | 4 | — |
| mean penetration | 0.35 mm | — |
| lift gained | **+117.9 mm** | **−16.5 mm** |
| still held afterwards | **yes** | **no** |

Under the same instruction the MuJoCo grip is effectively rigid and the Isaac one
lets go: the object ends *lower* than it started. That is a contact-level
difference, and it is the first mechanism offered for the transfer failure that
is not about the action space.

Two caveats keep this honest. The commanded lift differs between the columns
because the action scales differ, so "lift gained" is not a like-for-like
magnitude — the sign and the grip flag are what carry the result. And the Isaac
side reaches the object with the scripted expert rather than by teleporting the
hand, so a poor initial grasp cannot be fully separated from a poor *sustained*
one. Narrowing that is the next measurement, and it is a measurement rather than
an open question.

#### Is the gap tunable? Four corners say no, and one of them nearly lied

A mechanism is a diagnosis, not a fix, so the obvious next question is whether
some setting of Isaac's contact parameters holds what MuJoCo holds.
`scripts/isaac_contact_match.py` sweeps object friction, PhysX position-solver
iterations and collision rest offset on `contact_probe.py`'s protocol unchanged,
so the numbers sit next to MuJoCo's +117.9 mm rather than beside it.

Two things about how this was run, stated because neither is visible in the
result. **The scope was cut**: the intended grid was 3 frictions x 3 iteration
counts x 2 rest offsets, 18 cells, and building a second `GraspTask` inside a
live Isaac app turned out to be pathologically slow -- one cell took ~10 minutes
including startup, the next had burned 28 minutes of full-core CPU without
finishing. The sweep was reduced to the four corners at rest offset 0.0, each in
its own process. **The measurement is unseeded**: Isaac warns as much, and
identical settings returned -16.2, -32.4 and -32.6 mm, so a single cell is a
noisy sample rather than a value.

| friction | solver iterations | lift gained | still held |
| ---: | ---: | ---: | ---: |
| 1.0 | 4 | -32.6 mm | 0.00 |
| 1.0 | 64 | -16.5 mm | 0.00 |
| 4.0 | 4 | **+121.3 mm** | 0.00 |
| 4.0 | 64 | -29.8 mm | 0.00 |

That third row is why this section exists. **+121.3 mm is within 3 mm of
MuJoCo's +117.9**, and taken alone it says the transfer gap is a friction
setting. Replicated three times at identical settings it gave **+25.8, +14.6 and
+15.7 mm**. It was noise. Nothing but replication separated a headline finding
from a fluke, and the only reason it was replicated is that it disagreed with
its own `still_held` column.

What the corners do support, across every measurement taken:

* **No setting holds the box.** `still_held` is 0.00 in all four corners and all
  three replicates -- seven measurements, no grip retained, against MuJoCo's
  rigid hold.
* **Friction moves the lift and does not fix the grasp.** Quadrupling it takes
  the lift from clearly negative (-16 to -33 mm) to slightly positive (+15 to
  +26 mm). The box is dragged upward a little further; it is still not held.
* **More solver iterations do not help.** At friction 4.0, going from 4 to 64
  iterations moved the lift from positive back to -29.8 mm.

So the honest conclusion is the structural one this script was written to allow.
At the extremes of the parameters that plausibly govern a pinch, PhysX does not
reproduce MuJoCo's rigid grasp, and the difference is not a number waiting to be
tuned. A policy that depends on the grip surviving has to be trained in the
engine it will run in.

What this does **not** establish: that no parameterisation anywhere reproduces
it. Four corners of a three-dimensional space, at one object, one mass and one
grip command, cannot rule out an interior setting -- and joint drive stiffness,
contact offset and solver *velocity* iterations were never varied at all.

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
conclusion would have been a bug in the relabelling instead.

#### It does chain, and everything above was diagnosing the wrong thing

The paragraphs above, and the note in `place_reward.py` calling hindsight
*structurally inapplicable*, are wrong. The task chains from a sparse binary
reward with no demonstrations and no shaping:

| 5 seeds, 100 episodes, evaluated from the true start | success |
| --- | ---: |
| **sparse + hindsight + annealed start curriculum** | **0.944** [0.914, 0.974] |
| sparse + hindsight, fixed start band | 0.000 |
| sparse + fixed start band, no hindsight | 0.003 |
| sparse + hindsight, no curriculum | 0.000 |
| *for reference:* demonstration-seeded BC + RL | 0.916 |
| *for reference:* seven shaped from-scratch designs | 0.000 |

It reaches a task the nine-term shaped reward never solved from scratch, and
slightly exceeds the demonstration-seeded pipeline. Three corrections got there,
and each was a different kind of mistake.

**The structural claim was false.** `train_her.py` already stored ``lifted``
per transition and recomputed the success condition with it, so the latch did
travel with the relabelled transition. "The latch is history" was confused with
"the latch is unavailable"; only the first is true. The latch is now also part
of the observation (`--observe-latch`), which the policy needs anyway -- without
it the place task is partially observed, since the policy cannot tell whether it
has already lifted the box.

**The zero was exploration, and that is measurable separately.**
`scripts/her_relabel_probe.py` runs a *random* policy, so the number is about
the relabeller rather than any agent:

| | relabelled successes | latch set on |
| --- | ---: | ---: |
| no curriculum | 0 / 16 000 | 0.00 of frames |
| curriculum 0.2-0.8 | 7 961 / 16 000 | 0.57 of frames |

The first row reproduces the documented zero and shows its cause in the second
column: the box is never lifted, so every relabelled goal is scored against
``lifted = 0``. Hindsight reinterprets experience; it cannot invent it.

**A fixed start band is not a curriculum.** The first attempt drew start states
from a fixed 0.2-0.8 band and called that reverse curriculum generation. It is
not: Florensa et al. walk the start distribution *back to the true start* as the
policy improves. A fixed mid-task band trains a policy that finishes the job and
never starts it, which is exactly what the numbers showed -- 0.100 at 50 000
steps decaying to 0.000 by 100 000, while evaluation always began at the true
start. Annealing the band's upper bound to zero over the first 80% of training,
with the band always anchored at 0 so earlier stages stay in every batch, is the
difference between 0.000 and 0.944.

One thing this does not claim: the curriculum needs ``start_progress``, which is
scripted knowledge of the task -- cheaper than 200 demonstrations, and not
nothing, so "no demonstrations" is not "no prior knowledge".

**And it does not survive randomisation.** That caveat was written as untested
and is now measured. The same recipe at `medium`, five seeds, 200 000 steps:

| | success | peak grasp rate in training |
| --- | ---: | ---: |
| nominal | **0.944** [0.914, 0.974] | 0.97-1.00 |
| `medium` | **0.000** [0.000, 0.000] | 0.03-0.13 |

Zero on every seed, pooled Wilson [0.000, 0.008]. This is the steepest fall of
any method here: the weld goes 0.973 to 0.582 over the same range, the arm 0.530
to 0.388, and Isaac's demonstration-seeded arm 0.969 to 0.275. Demonstrations
degrade; this collapses.

The grasp column is the mechanism rather than a symptom. The recipe rests
entirely on relabelling manufacturing successes, and a relabelled success needs
the box at a goal, released, with the lift latch set. Under randomised mass,
friction, latency and compliance the policy grasps on at most 13% of episodes
against effectively all of them on the nominal world, so the latch is rarely
set and relabelling starves -- the identical failure mode as having no
curriculum at all, arrived at from the other direction. The curriculum supplies
lifted *start* states, but the policy still has to reproduce the lift itself
once the band anneals away, and under randomisation it never learns to.

**It transfers partially, and fine-tuning destroys it.** "Does not survive
randomisation" is true of *training* there and too strong as a statement about
the policy. The nominal policies evaluated directly on wider worlds, five seeds,
100 episodes:

| the nominal sparse policy, zero-shot | success |
| --- | ---: |
| `none` (where it was trained) | 0.944 [0.914, 0.974] |
| `low` | 0.264 [0.209, 0.319] |
| `medium` | **0.080** [0.054, 0.106] |
| `high` | 0.044 [0.033, 0.055] |

0.080 at `medium` is small and it is not zero, against exactly 0.000 for
training there from scratch. The skill is not unlearnable under randomisation;
it is undiscoverable there.

Which makes fine-tuning the obvious repair, and it fails in an instructive way.
Starting SAC at `medium` from a nominal checkpoint -- whole agent, actor,
critic and entropy coefficient -- begins at 0.200 success and 0.400 grasp
within 300 steps, and is at **0.000** by 50 000. Fine-tuning leaves the policy
*worse than not fine-tuning at all*.

That is the same mechanism this repository already documents for the cloning
anchor: unanchored RL walks away from a good initialisation, and the entropy
term is what walks it. There the fix was to hold the anchor rather than decay
it. The equivalent for a checkpoint is a frozen-policy anchor -- regularise the
actor toward the policy it started from -- and `SACConfig.anchor_coef` with
`SAC.freeze_anchor()` is that, added because this measurement asked for it.

It works, and it is not enough. Five seeds, 100 episodes at `medium`:

| | success |
| --- | ---: |
| training from scratch there | 0.000 [0.000, 0.000] |
| fine-tuning, unanchored | 0.000 [0.000, 0.000] |
| **zero-shot, no training at all** | **0.080** [0.054, 0.106] |
| fine-tuning, anchored | 0.100 [0.054, 0.146] |

The anchor does one thing cleanly: it converts a collapse into a plateau, 0.000
against 0.100 on identical seeds, checkpoints and curriculum, differing only in
whether the anchor is on. That confirms the mechanism a fourth time, alongside
the arm, the wrist and Isaac.

What it does **not** do is beat doing nothing. 0.100 against a 0.080 zero-shot
baseline, on intervals that overlap almost entirely, means 200 000 steps of
anchored fine-tuning at `medium` buys nothing over deploying the nominal policy
unchanged. Every route into `medium` measured here -- from scratch, unanchored,
anchored -- ends at or below the transfer number.

So the scope, stated exactly: **a sparse binary reward chains pick-and-place on
the nominal world, better than demonstrations; the policy transfers to `medium`
at 0.080 rather than zero; and nothing tried recovers more than that. Training
there from scratch reaches 0.000, unanchored fine-tuning destroys the transfer,
and anchored fine-tuning preserves it without improving on it.**

So the scope, stated exactly: **a sparse binary reward chains pick-and-place on
the nominal world, better than demonstrations; the resulting policy transfers to
`medium` at 0.080 rather than zero; training there from scratch reaches 0.000;
and fine-tuning there destroys what transfer there was.**

The relabelling itself is tested: goal entries, the derived goal-minus-object
entries and the recomputed reward, against the simulator's true object position
rather than the noisy observed one.

## The results are honest but small

**Five seeds.** Enough to report a t interval across seeds instead of an
anecdote; not enough for a tight one. Where two conditions overlap, this
repository says they overlap rather than picking the favourable framing.

**200 000 steps per run.** Chosen so the whole grid finishes in a few hours on
eight cores. It is not enough for SAC from scratch under randomisation, and the
results say so rather than quietly extending the budget for the conditions that
needed it.

That was the right call and it left the claim itself unmeasured, so
`experiments/budget_ladder.py` measures it: **600 000 steps, the same three
seeds, everything else held** — same entropy floor, same width, same
update-to-data ratio.

| | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| `medium`, 200k | 0.133 | 0.667 | 0.600 | 0.467 |
| **`medium`, 600k** | **0.833** | **0.833** | **0.767** | **0.811** |
| `high`, 200k | 0.000 | 0.133 | 0.300 | 0.144 |
| **`high`, 600k** | **0.600** | **0.600** | **0.600** | **0.600** |

Paired by seed, which is stronger than comparing across seeds: every one of the
six improves, and the two seeds that were at 0.000 and 0.133 finish at 0.600 and
0.833. On a clean evaluation the 600k runs reach 0.987 [0.972, 1.000] at
`medium` and 0.867 [0.705, 1.000] at `high`.

So "not enough" was correct and understated. These runs had not converged to
something poor — they were still climbing steeply at 200 000 steps, and the
collapsed seeds are collapsed *early* rather than permanently. It also sharpens
what the entropy floor does: it rescues seeds that would otherwise never start
learning, and more budget then rescues most of the rest.

**The headline grid stays at a matched 200 000 steps everywhere**, and these six
runs do not enter it. A table where each cell got as much compute as it needed
to look good is not a table, and the whole point of measuring this separately
was to avoid extending the budget only for the conditions that were losing.

What it does *not* transfer to is `shifted`: 0.030 [0.000, 0.105] and 0.030
[0.000, 0.073]. Tripling the budget triples the own-distribution score and
leaves transfer where it was, which is the same conclusion the entropy floor
produced.

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

**Two tasks now, in two simulators, and the second task is where the
reward-design method stopped working.** `task="place"` is pick-and-place: carry the box to a target patch
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
not**, and it now has seven reward designs, a tripled budget and a task
decomposition behind that sentence rather than a shrug. Every row is five seeds,
200 000 steps, identical settings, and the diagnostic column is behavioural
because the return curve was climbing in all of them:

| what `carry` and `approach` were keyed on | success | peak lift |
| --- | ---: | ---: |
| `carry` on `grasped` only | 0.007 | 0.012 m |
| `carry` on the binary lift latch | 0.000 | 0.010 m |
| `carry` on a clearance ramp, `clear` 0.18/step | 0.000 | 0.011 m |
| `carry` on a clearance ramp, `clear` 0.48/step | 0.007 | 0.011 m |
| + `approach` on horizontal distance (peaks while hovering) | 0.000 | **0.130 m** |
| + `approach` on 3-D distance (peaks at the release) | 0.000 | 0.036 m |
| + `approach` rising monotonically through both | 0.000 | 0.063 m |
| the first design at **600 000 steps** | 0.044 | 0.021 m |

Read down the peak-lift column rather than the success column: that is where the
information is. The first four designs never pick the box up at all — they close
the pads and push it, or grasp it and sit. The fifth is the first thing in the
investigation that produced *carrying*, and it did so because the term is
maximised while holding the object above the target — where it also pays nothing
for finishing, so five seeds carried and stopped. Correcting that so the maximum
sits at the release point stopped the lifting again. Making it rise
monotonically through lift, carry and descent — 1.125, 2.067, 3.000, with
sliding at exactly 0.000 — recovers about half the lifting and still scores
zero.

Two controls rule out the boring explanations. **Budget**: the first design at
600 000 steps, three seeds, more gradient updates than the entire lift grid was
trained with, reaches 0.044 while still sliding. **Task length**: the travel
ladder (`experiments/place_ladder.py`) shrinks the distance between object and
target to nothing, so there is no transport left to do. All three rungs, fifteen
seeds, score 0.000 — including the rung where the target sits where the object
started. The difficulty is not the distance.

The measurement that explains it is a decomposition of what the scripted expert
earns per step on each task:

| | positive reward | share from terms that only pay once the task is done |
| --- | ---: | ---: |
| lift-and-hold | 5.948/step | 51.8% |
| pick-and-place | 3.202/step | **80.7%** |

The lift task's shaping is worth 2.87 a step on its own, and it rises
continuously all the way to the goal — `hold` alone pays 1.73. The place task's
shaping was worth 0.62, and nothing in it paid more as the policy got closer to
finishing. A reward that is 81% terminal is a sparse reward with decorations,
and this repository already has a section showing that the sparse version of the
*first* task scores zero as well.

### A reverse curriculum, and the sharpest version of the result

The literature's answer to "shaping does not chain segments" is to learn the task
backwards (Florensa et al., [Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300),
2017). `GraspEnv(start_progress=p)` sets the world into a partly-finished
episode: at 1.0 the object is in the closed gripper over the target and only the
release remains, at 0.0 it is the ordinary task. Eight stages, 25 000 steps each,
**200 000 in total, matched to every from-scratch arm**, each stage inheriting the
previous stage's actor *and critic*.

| start | what the policy must still do | success |
| ---: | --- | ---: |
| 1.00 | lower, release | **0.888** |
| 0.85 | + a little carry | 0.614 |
| 0.70 | | 0.408 |
| 0.55 | | 0.368 |
| 0.42 | + the lift | 0.274 |
| 0.30 | grasped, on the table | 0.140 |
| **0.15** | **fingers open: + the grasp** | **0.000** |
| 0.00 | + the reach | 0.000 |

Graceful all the way down, then exactly zero the instant the fingers start open.
The obvious reading is stagewise forgetting — 150 000 steps seeing nothing but
states with the object already held — so the next experiment removes staging
entirely: sample the start point **per episode** across the whole range, so every
batch spans the task and no stage can overwrite the last. Five seeds, same
budget, same reward.

It scores **0.580** on the distribution it trains on, and **0.000 on the actual
task**. Both numbers are real and only the second one is the task:

| the same mixed-start policy, evaluated at | success | grasp at end | peak lift |
| --- | ---: | ---: | ---: |
| start 0.00 — the real task | **0.000** | 0.00 | 0.002 m |
| start 0.15 — fingers open | **0.000** | 0.02 | 0.002 m |
| start 0.30 — grasped, on the table | 0.367 | 0.13 | 0.061 m |
| start 0.55 — lifted, carrying | 0.750 | 0.17 | 0.106 m |
| start 1.00 — over the target | 0.750 | 0.15 | 0.096 m |

So it is not forgetting. One policy, trained on all start points at once, does
everything from "already holding the box" onwards at 0.37 to 0.75 — and scores
zero whenever it has to close the fingers itself.

**The two failures are complements.** From-scratch RL under the shaped reward
grasps on 63-90% of steps and never lifts. The curriculum policy lifts, carries,
lowers and releases and never grasps. So the obvious last experiment is to give
one policy both halves: seed the mixed-start curriculum from a shaped
from-scratch checkpoint — actor *and* critic — at the same 200 000 steps.

It starts with the grasp and **loses it**:

| seeded from a policy that grasps at 0.90 | success | peak lift | grasp at end |
| --- | ---: | ---: | ---: |
| start 0.00 — the real task | **0.002** | 0.002 m | 0.03 |
| start 0.15 — fingers open | 0.000 | 0.012 m | 0.08 |
| start 0.30 — grasped, on the table | 0.433 | 0.100 m | 0.45 |
| start 0.55 — lifted, carrying | 0.483 | 0.125 m | 0.48 |
| start 1.00 — over the target | 0.517 | 0.108 m | 0.43 |

The grasp is not merely absent from what the curriculum teaches — it is **actively
overwritten**. About four fifths of mixed-start episodes begin with the object
already held, so the gradient that maintains "close the fingers on a box on the
table" is a minority of every batch, and 200 000 steps of it erases a behaviour
the initialisation arrived with. The policy ends up exactly where the unseeded
curriculum did.

So the complete statement, after seven reward designs, a tripled budget, a travel
ladder, a staged curriculum, a mixed curriculum and a seeded combination, every
one at a matched budget:

**Each half of pick-and-place is learnable by this algorithm on this robot. No
method tried here learns both in sequence, and every attempt to combine them
loses whichever half the current training distribution needs less. Demonstrations
are the only thing that supplied the sequence, and they did it on the first
attempt.**

That is a much sharper claim than "pick-and-place is hard", and earning it took a
second task and six separate controlled experiments.

**The honest conclusion is about the family of designs, not the next weight.**
Hand-shaped dense rewards of this form solved lift-and-hold and did not solve
pick-and-place, and seven attempts is enough to stop reporting the eighth as
imminent. What the shaping *can* be made to do is buy individual segments of the
behaviour — reaching, grasping, lifting, carrying — each time by putting a
maximum where that segment ends, and each time the policy stops there. Chaining
segments is the thing this method does not do, and demonstrations supply exactly
that: the whole sequence, in order, for free.

That is a limitation of the approach the repository demonstrates, and it took a
second task to find it. One task could not have.

**One box shape by default**, and see the shape paragraph above for what happens
with three.

**`shifted` is a proxy, not a robot.** See [sim-to-real.md](sim-to-real.md). No
hardware was involved at any point.

## The Isaac Lab port runs both tasks, and now produces numbers of its own

It was brought up against Isaac Sim 5.1.0 and Isaac Lab 2.3.2 on an RTX 4060.
Both tasks run there: the bring-up checks pass at `none` and `medium` for
lift-and-hold, and all eight pass at both levels for pick-and-place, whose
reward agrees with the shared numpy implementation to 6.9e-08 on the GPU.

**It reproduces the second task's finding on a different robot** — 0.419
demonstration-seeded against 0.000 from scratch, three seeds each, with the
prediction recorded in `experiments/isaac_place_grid.py` before the runs. What it
still does not do is contribute to the README's headline tables, which remain
MuJoCo throughout. That includes the one the port exists for: the reward computed
inside Isaac on the GPU agrees with the shared numpy implementation to about
5e-08 on the same states, so "it is the same task" is measured rather than
asserted. Randomisation is driven through Isaac's event manager from the same
JSON ranges, with the same interval arithmetic.

**The randomisation sweep is now complete on both arms.** Four levels, five
seeds each, twenty runs per arm, 4 000 steps at 512 environments:

| level | from scratch | demonstration-seeded |
| --- | ---: | ---: |
| `none` | 0.000 [0.000, 0.000] | **0.969** [0.902, 1.000] |
| `low` | 0.000 [0.000, 0.000] | 0.519 [0.451, 0.586] |
| `medium` | 0.000 [0.000, 0.000] | 0.275 [0.182, 0.368] |
| `high` | 0.000 [0.000, 0.000] | 0.041 [0.025, 0.056] |

Two readings, and one caution that has to come with them.

**From-scratch RL is 0.000 at every level, on all forty runs.** Not small, not
noisy: zero, with intervals of zero width. This repository's central claim --
that demonstrations are what make this task learnable at this budget -- now
holds in a second simulator across a full randomisation sweep rather than at one
operating point.

**Demonstration-seeding degrades monotonically and then falls off a cliff.**
0.969 to 0.519 to 0.275 to 0.041. The `high` column is the interesting one: the
grasp rate there is 0.08-0.10, so the policy is failing to *close on the box*
rather than failing to lift it. That is consistent with the contact finding
above -- Isaac's grip is the fragile part -- though the link is asserted from
two measurements sitting next to each other, not established.

The caution is the budget. Isaac runs 4 000 steps at 512 environments against
MuJoCo's 200 000, matched on **gradient updates** rather than transitions,
because matching updates is what made the two agree at `none` in the first
place. MuJoCo over the comparable range goes 0.973 to 0.582 at `medium`, so
Isaac's collapse is steeper -- but that compares two configurations, not two
simulators, and this document has already recorded three cross-condition
comparisons that turned out to be measuring something other than what they
claimed. Treat the shape as real and the rate as configuration-specific.

`medium` appears twice in this repository at different budgets: 0.275 at 4 000
steps and 0.131 at 15 000. That is not an inconsistency, it is the documented
finding that randomised Isaac runs get *worse* with more training, and the table
above quotes the 4 000-step column throughout for comparability.

Every success rate in the README's headline tables was still produced in MuJoCo.
The port also carries the entropy floor against its control (0.194 against
0.463, t = 1.01 — not separated) and two floor sweeps. The randomised runs get *worse* with more training, and the
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

This list has been rewritten four times as items came off it, and one thing is
left on it.

1. **Measured randomisation ranges from real hardware, and a real robot to check
   any of this against.** Everything else that was on this list has been done and
   is reported above, including the items that came back as negative results. The
   ranges here have been audited against published measurements
   ([randomisation-sources.md](randomisation-sources.md)) and against this
   repository's own perception stack, and both say the same thing: they are
   optimistic. A survey is not a robot, `shifted` is a proxy, and no amount of
   further simulation closes that.

Everything that *was* on this list is now a finding rather than a plan, and is
written up above rather than promised here. Five changed meaning when they were
finally measured, and in every case the cause was different from the one this
document had blamed:

* **Chaining segments without demonstrations** — **solved on the nominal world,
  and the old explanation was wrong.** Seven shaped designs, a tripled budget and
  two curricula all scored 0.000, and this document concluded that shaping buys
  segments without chaining them, and that hindsight was *structurally
  inapplicable*. Neither held. The relabeller always carried the lift latch
  correctly; the zero came from exploration, with the latch set on 0.00 of frames
  so there was nothing to relabel. A sparse binary reward with hindsight and a
  start curriculum that **anneals back to the true start** reaches **0.944**
  [0.914, 0.974] — above the 0.916 of the demonstration-seeded pipeline, with no
  demonstrations and no shaping. It scores **0.000** at `medium`, so the scope is
  the nominal world only.
* **Why cross-simulator transfer fails** — narrowed past every action-space
  explanation to the contact model, measured at the contact level, and then
  tested for tunability: friction, position and velocity solver iterations,
  collision rest and contact offset, and finger drive stiffness. Contact is made
  in every cell and the grip is retained in none. The difference is structural,
  not a parameter waiting to be found.
* **Real sensing ranges** — measured, the consequence measured, then largely
  **un-measured**: the noise model turned out to be harsher than the estimator it
  was calibrated from. With dynamics held identical, the real CNN in the loop
  gives **0.728** [0.677, 0.779] where injected random noise of the same
  magnitude gives **0.406** [0.345, 0.467]. A CNN returns the same wrong pose for
  the same scene, and a policy learns to invert a repeatable distortion;
  `measured_camera` matched the magnitude and discarded the structure.
  Unfreezing the estimator changes nothing (0.734), because a trained policy
  visits *easier* states, not harder ones.
* **Grasp-point selection** — built and learned: 0/30 for the naive strategy,
  0.896 cloned, 0.996 demonstration-seeded.

Three further items were closed by finding bugs rather than by running more
compute, which is worth recording as a pattern. The arm's collapse under
randomisation was a position servo whose gain was scaled without its bias. Every
wrist-versus-no-wrist number in this repository's history compared two different
object-size distributions, because the cap moved with the wrist flag. And
from-scratch RL through the camera was declined on cost -- 75.9 ms a step, four
hours for 200 000 -- when the actual obstacle was that `train_rl.py` had no
`--perception` flag; with one, it reaches 0.950.

What is left is a boundary rather than a queue. Three failure modes are located
and unfixed, and they look identical in a success column: the arm cannot
**reach** (it stalls 116 mm above the box, with six causes eliminated), the
sparse recipe at `medium` cannot **grasp** (0.13 at best, so relabelling
starves), and the camera at `medium` grasps at 0.55-0.65 and cannot **hold**.
None of them is the reward. Each would need a different fix, and none of those
fixes is more simulation.
