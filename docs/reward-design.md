# Reward design

The reward lives in one file, [`src/rewards/grasp_reward.py`](../src/rewards/grasp_reward.py),
and both simulators call it. This document is the reasoning behind it, including
the two shapings that were tried, failed, and were replaced — because on this
task the reward *was* the hard part, and a weights table with no history is not
much use to anyone.

## The task, stated as an objective

Close a parallel-jaw hand on a box, lift it to a hold point 0.15 m above the
table, and still be holding it there when the episode ends. Success is read at
the **final** step:

```
success = (|object - hold point| < 0.05 m) AND (both pads in contact)
```

Reading it at the final step rather than "at any point" is not a detail. A
policy that flings the box upward passes through the goal sphere and, under the
any-point definition, scores 100%. Under this one it scores zero, which is what
a gripper that has thrown its payload deserves.

## Terms

| Term | Weight | What it is for |
| --- | ---: | --- |
| `reach` | 1.0 | `-(1 - exp(-d_grip_obj / 0.10))`. Saturating, so a box on the far side of the table does not dominate the sum. |
| `align` | 0.3 | Extra penalty on *horizontal* offset only. Vertical error is recoverable; lateral error tips the box over. |
| `grasp` | 0.5 | Flat bonus while both pads are in contact. Deliberately small — it is a means, not the goal. |
| `lift` | 4.0 | Height gained above the resting height, clipped at 0.12 m. Bootstraps the behaviour. |
| `place` | 6.0 | *Progress* from the resting place towards the hold point, gated on `grasped`. |
| `hold` | 3.0 | `exp(-d_goal / 0.05)`, gated on `grasped`. A smooth hill centred on the hold point. |
| `success` | 5.0 | Paid on every step the success condition holds. Rewards holding, not touching. |
| `drop` | −5.0 | One-off, when the box leaves the table. |
| `action` | 0.02 | Squared-action penalty. Discourages bang-bang commands. |
| `time` | 0.0 | Off. The horizon is fixed, so a per-step cost only rescales everything. |

Resulting reward per step, on the nominal world, with the hand at the box:

| State | Grasped | Not grasped |
| --- | ---: | ---: |
| Box on the table | 0.73 | 0.00 |
| Lifted 5 cm | 1.63 | 0.20 |
| Lifted 8 cm | 7.45 | 0.32 |
| **At the hold point** | **9.75** | 0.48 |
| 5 cm above the hold point | 2.55 | 0.48 |
| At the ceiling of the workspace | 1.70 | 0.48 |

The jump between 5 cm and 8 cm is the success bonus switching on: at 8 cm of
lift the box is 4.8 cm from the hold point, just inside the 5 cm tolerance.

That table is the design. Every state transition a good policy makes is uphill,
the peak is at the hold point, and it is a peak rather than a plateau.

Reproduce it with:

```bash
python scripts/reward_landscape.py
```

## Two shapings that did not work

**Sparse reward only.** Never tried seriously, and the honest reason is compute:
success requires a specific 4-D action sequence roughly forty steps long, and
finding it by chance inside a few hundred thousand steps on a CPU is not
plausible. A sparse formulation on this task wants hindsight experience replay
and a much larger budget. That is a real limitation of what is demonstrated
here: the result shows dense-reward RL working, not RL discovering grasping
from scratch.

**A cliff at the goal instead of a hill.** The first version had `lift` at 8.0,
`place` as a plain distance penalty, and no `hold` term. Every seed learned to
grasp (grasp rate 0.93 by 30k steps) and then hoisted the box to the top of the
workspace and held it there: mean peak lift 0.22 m against a 0.128 m hold point,
success 0.000 at 50k steps and still climbing in return. The reason is visible
in the numbers: with `lift` clipped at 0.12 m, everything above the hold point
paid the same, and the only thing marking the hold point was a binary +5 inside
a 5 cm ball. A policy travelling upward crosses that ball in one or two steps
out of a hundred, so the critic almost never saw the bonus. Adding `hold` — an
exponential hill with a 5 cm width — turned a cliff the policy skipped over into
a gradient it could climb.

**A `place` term that charged for lifting.** The fix above came with `place`
raised to 6.0 as an absolute distance penalty, and that made things worse rather
than better: with the box grasped on the table, `place` paid −0.77 per step
while `grasp` paid +0.5, so *picking the box up at all* was worse than leaving
it alone until the policy could also lift it. Grasp rates collapsed from 0.93 to
0.40 and lift heights to under a centimetre. Rewriting `place` as progress from
the resting position — zero on the table, rising as the box comes up — removed
the barrier without changing what the term is for.

Both failures share a shape: a term that is correct at the optimum and wrong on
the path to it. Checking the reward at the optimum is not enough; the table
above exists because the useful check is the whole path.

## The same failure again, on a different task

Everything above was learned on one task, which makes it a story about this
reward rather than about designing rewards. `src/rewards/place_reward.py` is a
second task -- pick the box up, carry it somewhere else on the table, put it
down -- and it was written by someone who had just written all of the above.

It failed in the same way on the first attempt.

`carry` pays for progress across the table towards the target and was gated on
`grasped`, by direct analogy with `place` in the lift reward. `grasped` means
both pads in contact with the object, and both pads can be in contact with an
object that is being **pushed**. Five from-scratch seeds found that: grasp rate
0.63-0.83, mean peak lift 0.010 m against a 0.04 m latch, 0/20 episodes ever
picking the box up, and 0.002 [0.000, 0.008] success. The largest dense term in
the reward was payable without doing the task, so it was collected without doing
the task.

The fix is one line -- `carry` is multiplied by the lift latch, so transport
pays nothing until the object has actually been picked up -- and it is the same
fix as the yaw term's proximity gate in the lift reward, arrived at
independently.

What this costs to learn is worth stating plainly: three separate shaping terms,
across two tasks, each correct at the optimum, each wrong on the path, each
found only by training on it and reading what the policy actually did. The
diagnosis in every case came from a behavioural quantity (peak lift height,
grasp rate) rather than from the return curve, which in the sliding case was
climbing perfectly happily.

## Termination

* **Drop** — the box falls below 0.34 m (6 cm under the table top). Terminal,
  and bootstrapping stops there, because it is a genuine absorbing state.
* **Time limit** — 100 steps of 40 ms, i.e. 4 s. Truncation, not termination:
  the value function bootstraps through it. Treating a time limit as terminal
  teaches the policy that the world ends at four seconds, which shows up as a
  policy that gets careless near the end of an episode.

## Changing the weights

`GraspRewardConfig` loads from JSON:

```bash
python src/train_rl.py --reward-config my_weights.json ...   # via make_env(reward_config=...)
```

Unknown keys are rejected rather than ignored, so a typo in a weight name fails
loudly instead of silently training against the defaults.
