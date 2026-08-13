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

**The from-scratch runs have an entropy-collapse failure mode, and it is not
fixed.** The seeds that stall settle at an entropy coefficient around 0.025
while the seeds that solve the task sit near 0.17 — an order of magnitude more
exploration. The stalled policies grasp the box reliably and hold it on the
table, which the reward pays 0.73 per step for against 9.75 at the hold point:
a local optimum a nearly deterministic policy has no way out of.
`docs/plots/entropy_collapse.png` plots it, one point per run.

The obvious remedy was tried and did not work. Three runs on the nominal world,
using the three seeds that stalled, with the target entropy raised from
-dim(A) to -dim(A)/2 (`--target-entropy-scale 0.5`), 100 000 steps each:

| run | final entropy coefficient | success |
| --- | ---: | ---: |
| `probe_entropy_s2` | 0.051 | 0.00 |
| `probe_entropy_s3` | 0.050 | 0.00 |
| `probe_entropy_s4` | 0.052 | 0.00 |

The coefficient roughly doubled, against about 0.025 in the stalled baseline
runs, and none of the three escaped. So the entropy coefficient is a reliable
*marker* of the failure but raising it by this much is not a cure, and the
honest state of the diagnosis is: mechanism identified, fix not found. The raw
curves are in `experiments/runs/probe_entropy_s*/progress.csv`; reproduce with

```bash
python src/train_rl.py --steps 100000 --seed 2 --randomisation none     --hidden 128 --target-entropy-scale 0.5 --output experiments/runs/probe_entropy_s2
```

**One task.** Lift-and-hold, one hold point, one box shape.

**`shifted` is a proxy, not a robot.** See [sim-to-real.md](sim-to-real.md). No
hardware was involved at any point.

## The Isaac Lab port runs, but not completely

It was brought up against Isaac Sim 5.1.0 and Isaac Lab 2.3.2 on an RTX 4060.
Four of its five bring-up checks pass, including the one the port exists for:
the reward computed inside Isaac on the GPU agrees with the shared numpy
implementation to 5.1e-08 on the same states, so "it is the same task" is now
measured rather than asserted.

The check that fails is the scripted expert grasping reliably. The cause is
measured, not guessed: the Franka's implicit PD leaves a standing 70-115 mm
error between the commanded IK setpoint and the achieved pose, growing as the
arm extends, while the expert is a state machine with 12 mm phase tolerances
written for a MuJoCo hand that tracks its setpoint to the millimetre. The grasp
mechanism itself works — `envs/isaac/README.md` has a trace of the gripper
closing on the box and lifting it to the hold point — but not across arbitrary
spawns. Fixing it needs a control path that reaches its setpoint (gravity
compensation, stiffer gains, or operational-space control), which is real work
and is not done.

Also not done in Isaac: the randomisation ranges are loaded from the shared
configs but not yet applied through Isaac Lab's event manager, so episodes run
at nominal; the grasp test is geometric rather than contact-based; and **no
policy has been trained there**. Every learned number in this repository comes
from MuJoCo.

**Cross-simulator transfer has not been attempted.** The control paths differ
(mocap weld against differential IK on a Franka), so a MuJoCo policy is not
expected to run in Isaac unchanged.

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
