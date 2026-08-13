# Imitation: demonstrations, cloning, DAgger, and RL on top

## Where the demonstrations come from

A four-phase state machine,
[`src/policies/scripted_expert.py`](../src/policies/scripted_expert.py):

```
APPROACH   move to a waypoint 85 mm above the box, fingers open
DESCEND    drop onto the grasp height, fingers still open
CLOSE      squeeze, and wait six steps for the pads to load
LIFT       rise to the hold point and stay there
```

It reads the same 32-dimensional observation the policies see and nothing else,
so it can label states a *learner* visited — which is what makes DAgger possible
here without a human.

Three details in it are not decoration, and each was added to fix an observed
failure:

* **A low-pass filter on the pose channels.** Under the noisier worlds a raw
  reading jitters by more than the phase tolerances, and an unfiltered state
  machine never leaves APPROACH. Any real system filters its pose estimate.
* **Per-phase speed caps.** With command latency in the world, a saturated
  lateral command during the descent sweeps the box off the table before the
  fingers ever close. Approach is fast, descent is a third of that.
* **A lift phase that closes the loop on the box, not on the hand.** The weld
  that drags the hand is compliant, so under load the hand hangs below where it
  was told to go. A heavy box and a soft weld put that sag outside the goal
  tolerance. Commanding "move by whatever error the *object* still has" cancels
  it without knowing the compliance.

Recording:

```bash
python src/record_demos.py --episodes 200 --randomisation low \
    --expert-noise 0.02 --output demonstrations/expert_low.npz
```

Failed episodes are dropped by default: cloning an expert's failures teaches a
policy to fail in the same places. A little action noise (σ = 0.02) is added
during recording, which widens the state distribution the clone sees and is the
cheapest available mitigation for compounding error.

The shipped set is 200 episodes recorded at the `low` level, 20 000
transitions, and the expert succeeded on 99.5% of its attempts while recording.

## Behaviour cloning

Supervised regression from observation to expert action, mean squared error on
the deterministic mode of the same actor network SAC uses — so a cloned actor
can be loaded straight into SAC.

```bash
python src/train_il.py --demos demonstrations/expert_low.npz --epochs 60 \
    --seed 0 --output experiments/runs/bc_s0
```

With the full 200 episodes the clone matches the expert on the distribution it
was recorded on. That is not a surprising result and it is not the interesting
one; the interesting parts are what happens with fewer demonstrations, and what
happens off-distribution.

![behaviour cloning data efficiency](plots/bc_data_efficiency.png)

The characteristic failure is compounding error: the clone is trained on states
the expert visits, so the first small error puts it somewhere the expert never
was, its next action is an extrapolation, and the deviation grows. The
signature is a *small action error and a poor success rate at the same time* —
which is exactly what the low-demonstration end of that curve shows.

## DAgger

The fix that keeps the supervised setup: roll out the learner, label the states
it actually visited with the expert's action, aggregate, retrain.

```bash
python src/train_il.py --demos demonstrations/expert_low.npz --dagger \
    --dagger-rounds 5 --randomisation shifted --output experiments/runs/dagger_s0
```

β, the probability of executing the expert's action instead of the learner's,
anneals from 0.5 to 0 across rounds: early rounds stay near states where the
expert is competent, later rounds go wherever the learner actually goes.

DAgger here is run against the **shifted** worlds, because that is where plain
cloning has room to improve. It costs expert queries at run time, which are free
in this repository because the expert is a function. On a real robot they are a
human with a joystick, and that is why DAgger is less used in practice than its
results deserve.

![dagger rounds](plots/dagger_rounds.png)

## Imitation plus reinforcement learning

The combined run is `src/train_rl.py` with three extra flags, and each does
something different:

| Flag | Effect |
| --- | --- |
| `--init-actor experiments/runs/bc_s0/policy.pt` | starts from the cloned policy instead of a random one |
| `--demos demonstrations/expert_low.npz` | writes the demonstrations into a *pinned* prefix of the replay buffer, which the ring never overwrites |
| `--demo-fraction 0.25` | draws a quarter of every batch from that prefix |
| `--bc-coef 50 --bc-decay-steps 100000` | behaviour-cloning term on the demonstration slice, decaying to zero over half of training |
| `--critic-warmup 3000` | critic-only gradient steps before the actor is allowed to move |
| `--target-entropy-scale 2.0 --init-alpha 0.02` | keeps the fine-tuned policy closer to deterministic than a from-scratch run |

### Three things that had to be fixed before this worked at all

Loading a cloned actor into SAC and pressing go destroys the clone. Measured on
the `medium` distribution, starting from a clone that scores 0.70: it was at
0.00 within two thousand actor updates. Three separate causes, each worth
naming because each is invisible from the training curve alone.

**The critic starts random, and the actor loss is −Q.** For the first few
thousand updates the critic is noise, so "improve the policy" means "move the
policy towards whatever the noise prefers". `--critic-warmup` holds the actor
still while the critic fits the demonstrations. Verified directly: with the
actor frozen the clone keeps scoring 0.70 indefinitely, so the loading path is
not the problem.

**The Q term and the BC term are not on the same scale.** Returns here run to
several hundred, and a behaviour-cloning MSE is of order 0.01, so summing them
is not a trade-off — it is the Q term with a rounding error attached. The Q
term is therefore divided by its own mean absolute value, as
[TD3+BC](https://arxiv.org/abs/2106.06860) does, and the coefficient is 50
rather than 1. At 5 the clone still collapsed to 0.20; at 20 it dipped to 0.55;
at 50 it held at 0.90 and improved.

**The Q-filter has to be off early.** Applying the BC term only where the critic
prefers the expert action is a well-known refinement, and it is exactly wrong at
the start of fine-tuning: an untrained critic systematically overrates the
actions the policy is already taking, so the mask is mostly zero and the anchor
disappears at the moment it is holding the clone together. It is available as
`--bc-q-filter` and defaults to off, with the reasoning recorded in
[`src/policies/sac.py`](../src/policies/sac.py).

None of this is exotic; all three are standard failure modes of offline-to-online
fine-tuning. They are written down here because a repository that shows only the
working configuration teaches nobody why it is that configuration.

Pinning matters more than it sounds. With a 400 000-transition ring and 20 000
demonstration transitions, an unpinned buffer loses every demonstration inside
the first two hundred thousand steps, which is precisely when the BC term is
still switched on. `tests/test_learning.py::test_replay_buffer_never_overwrites_pinned_demonstrations`
holds that down.

The comparison against SAC from scratch is in the README's headline table and in
the right-hand panel of the training curves.

### The coefficient schedule leaves a visible scar

The seeded training curves drop sharply at exactly 100 000 steps, which is where
the coefficient reaches zero: at medium randomisation, from about 0.95 to about
0.5, recovering to about 0.75 by the end of training. The anchor is doing more
work than the decay schedule assumed. A slower decay, or one conditioned on the
critic loss having settled rather than on a step count, is the obvious fix and
is not implemented here. The numbers in the tables are taken after the drop,
because they are what the configuration in this repository actually produces.
