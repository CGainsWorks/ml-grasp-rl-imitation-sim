# Limitations

Written to be read before the results, not after them.

## The environment is a simplification, in specific ways

**There is no arm.** The MuJoCo hand is a free body dragged by a mocap weld.
No joint limits, no self-collision, no arm inertia, no singularities, no
reachability constraint. Every one of those is a real failure mode in a real
cell. A policy trained here needs a reachability check and a joint-limit clamp
between it and any servo loop.

**The hand cannot rotate.** The pads always close along world *x*. A square box
at 45° of yaw presents √2 times its side, so above roughly 27 mm of half-size it
is ungraspable at the worst yaw regardless of the policy. Object sizes are
therefore capped at 24 mm half-size everywhere. Adding a wrist yaw degree of
freedom is the single change that would most improve this task's realism: it
turns "close the fingers" into "align, then close", which is where the real
difficulty in grasping lives.

**Objects are boxes.** One shape, randomised in size, mass and friction. No
cylinders, no bottles, no bags, no clutter, no bin. Nothing here demonstrates
grasp *selection* — choosing where to grasp an unfamiliar shape — which is what
most of the grasping literature is actually about.

**The policy is handed the object pose.** No camera, no point cloud, no
detector. The sensing randomisation adds Gaussian noise to a ground-truth pose,
which is a weak model of a real pose estimator: its error is correlated across
steps, biased by viewpoint, and worst when the gripper occludes the object.

**The reward is dense and hand-designed.** Nothing here shows RL discovering
grasping from a sparse signal; a sparse version of this task wants hindsight
experience replay and a much larger compute budget than a CPU afternoon.
[docs/reward-design.md](reward-design.md) is explicit about the two shapings
that failed before the current one worked.

## The results are honest but small

**Five seeds.** Enough to report a t interval across seeds instead of an
anecdote; not enough for a tight one. Where two conditions overlap, this
repository says they overlap rather than picking the favourable framing.

**200 000 steps per run.** Chosen so the whole grid finishes in a few hours on
eight cores. It is not enough for SAC from scratch under randomisation, and the
results say so rather than quietly extending the budget for the conditions that
needed it.

**The from-scratch runs have an entropy-collapse failure mode, and it is now
fixed — but the headline tables predate the fix.** The seeds that stall settle
at an entropy coefficient around 0.025 while the seeds that solve the task sit
near 0.17. A floor under that coefficient rescues all five seeds at half the
budget (0.993 mean, 95% t [0.975, 1.000], against 0.400 and [0.000, 1.000]
without it). The investigation, including the hypothesis that turned out to be
wrong, is in [exploration.md](exploration.md).

The floor fixes the nominal world only. Under `medium` randomisation it moves
five seeds from 0.120 to 0.160, which does not survive its own confidence
interval, so whatever stops from-scratch SAC learning under randomisation is a
different problem and is still open.

The grid in the README was run before this and is deliberately left alone: it is
a fair record of what a standard SAC configuration does here, and rerunning
three hours of compute to replace an honest result with a flattering one would
not change either conclusion it supports. Read the from-scratch rows as a
demonstration of seed variance and of what demonstrations buy, not as the best
this algorithm can do on this task.

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

What it does *not* do is produce any number quoted in this repository. Every
success rate in the README was produced in MuJoCo. `scripts/isaac_train.py`
runs the same SAC implementation against the vectorised Isaac environment and a
short run does learn to grasp, but no full grid has been trained there, so
there is no Isaac column in any table.

Four of the eleven randomisation parameters have no Isaac equivalent yet:
object size needs a pre-startup scale term, hand compliance is a property of
the MuJoCo weld with no analogue, action latency would need a command queue in
the task, and gravity is per-scene rather than per-environment. Those four are
listed in `envs/isaac/README.md` rather than quietly omitted.

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

1. A wrist yaw degree of freedom, and object yaw alignment as part of the task.
2. Measured randomisation ranges from real hardware, replacing the guessed ones.
3. Perception: a wrist camera and a pose estimator in the loop, with its real
   error model rather than additive Gaussian noise.
4. More object shapes, and grasp-point selection.
5. Isaac bring-up, then a cross-simulator evaluation, which is the closest thing
   to a sim-to-real test available without a robot.
6. A sparse-reward variant with hindsight experience replay, to show the task
   can be learned without hand-designed shaping.
