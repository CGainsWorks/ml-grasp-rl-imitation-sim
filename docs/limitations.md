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

**A six-jointed arm variant exists and does not work yet.**
`envs/mujoco/assets/grasp_scene_arm.xml` carries a generic 6R arm with joint
limits and self-collision, and `make_env(..., arm=True)` drives it through
damped-least-squares IK so the policy keeps its Cartesian action space — the
wrist experiment above is why the joints are not exposed to the policy directly.

What works: the model loads, the IK converges, and 25 of 27 workspace corners
are reachable to within 15 mm.

What does not: the IK is collision-blind, as IK generally is, and it reaches the
table by folding the arm *through* it. Twenty resets produce about a hundred
penetrating contacts, up to 120 mm deep, and the first physics step of an
episode resolves them explosively — the scripted expert scores 0.000 with a mean
peak object height of 13.9 m, which is the box being thrown across the room.

Four fixes were tried and measured. None worked, and the reason is a design
problem rather than a bug:

| attempt | penetrating contacts / 20 resets | workspace corners unreachable |
| --- | ---: | ---: |
| as built | 105 | 2 of 27 |
| restrict the elbow's joint range | 105 | — |
| nullspace posture bias | 90 | 9 of 27 |
| IK restarts with a collision check (40 tries) | 113 | — |
| mount the arm on a pedestal above the table | 41 | 24 of 27 |

The nullspace attempt deserves recording because it is *mathematically vacuous*
here and looked reasonable: six joints against a six-dimensional pose target
leave no redundancy, so `I − J⁺J` is zero except at singularities. A nullspace
posture bias needs a seventh joint, or a task that does not constrain all six
degrees of freedom.

The restart attempt is the informative one. Forty random restarts per reset,
keeping any solution that is contact-free, found **zero** in twenty resets — so
contact-free solutions reaching the start pose are not merely hard for the
solver to find, they are close to absent for this arm in this scene. A separate
search over 400 000 random configurations did find contact-free postures near
the workspace centre with the pads facing down, which means the set is not empty
but is small and awkwardly placed.

And the last row is the trade laid bare: raising the base cuts penetration by
more than half and costs almost all the reach.

What this actually needs is kinematic *design*, not debugging: the links here
are a naive serial stack of collinear capsules, where real arms use link offsets
precisely so the elbow can clear the workspace it reaches over. Choosing link
lengths, offsets and a base placement that give both reach and clearance is a
half-day of geometry with a reachability map to check against — not something to
converge on by adjusting one number at a time, which is what the table above is
a record of.

Until then the flag is off by default, no training script uses it, and no number
in this repository comes from it. It is recorded here rather than deleted
because a half-built arm that is honestly labelled is more useful than the
absence of one.

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

Learned, on the mixed distribution at 200 000 steps against the box-only control
at the same budget, the variance swallows the comparison: 0.167 [0.000, 0.884]
across three seeds (0.50, 0.00, 0.00) against 0.407 for boxes alone across five.
Adding shape variety plausibly makes learning harder at a fixed budget, in the
same way every other widening of the distribution does here, but three seeds
with that spread cannot establish it.

No bottles, no bags, no clutter, no bin, and still nothing here demonstrates
grasp *selection* — choosing where to grasp an unfamiliar shape — which is what
most of the grasping literature is actually about.

**The policy is handed the object pose.** No camera, no point cloud, no
detector. The sensing randomisation adds Gaussian noise to a ground-truth pose,
which is a weak model of a real pose estimator: its error is correlated across
steps, biased by viewpoint, and worst when the gripper occludes the object.

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

**One task.** Lift-and-hold, one hold point, one box shape.

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

1. **A reward term for yaw alignment.** The wrist degree of freedom now exists
   and the scripted expert uses it, but a policy trained with it scores 0.000
   against 0.122 without it: `w_align` penalises lateral offset and nothing
   rewards turning the wrist, so the extra dimension is exploration cost with no
   gradient. The joint was the easy half.
2. Measured randomisation ranges from real hardware. The guessed ones have now
   been checked against published measurements rather than defended
   ([randomisation-sources.md](randomisation-sources.md)): they are optimistic
   on latency by 2-5x, optimistic on sensing, and omitted orientation error
   altogether. Real hardware would still be better than a literature survey.
3. Perception: a wrist camera and a pose estimator in the loop, with its real
   error model rather than additive Gaussian noise.
4. More object shapes, and grasp-point selection.
5. **Why cross-simulator transfer fails.** The evaluation exists and the answer
   is not a control-gain constant — seven scalings of the action, in both
   directions, none of which clears its interval. Halving lateral commands alone
   raises peak lift to what a successful MuJoCo policy reaches while success
   does not move, which points at contact rather than reaching, with one
   supporting number.
6. A sparse-reward variant with hindsight experience replay, to show the task
   can be learned without hand-designed shaping.
