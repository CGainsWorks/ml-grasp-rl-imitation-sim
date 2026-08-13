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

## The Isaac Lab port has never been run

`envs/isaac/grasp_task.py` is written and reviewed; there is no Isaac Sim
installation here, so it has never been executed. It shares the reward and the
randomisation ranges with the MuJoCo task through imports rather than copies,
and `tests/test_reward_parity.py` verifies numpy and torch agree — which is the
strongest claim available without a simulator to run it in. Its grasp test is
currently geometric rather than contact-based, which is weaker than the MuJoCo
version. `envs/isaac/README.md` lists the bring-up order.

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
