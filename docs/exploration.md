# Why three seeds in five stalled, and what fixed it

This is the one genuinely open question this repository had, and it is now
closed. The write-up is kept because the route to the answer is more useful than
the answer: two plausible hypotheses, both with literature behind them, one
decisive and one not.

## The symptom

SAC from scratch on the nominal world, five seeds, 200 000 steps each:

| seed | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| success | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |

Mean 0.40, 95% t interval across seeds **[0.000, 1.000]** — an interval so wide
it carries no information. The three failures are not noisy versions of the
successes; they are a different behaviour. The stalled policies grasp the box on
about 90% of episodes and hold it on the table for the rest of the episode,
never lifting. The reward pays 0.73 per step for exactly that, against 9.75 at
the hold point.

Two measurements narrowed it down:

* The stalled runs settle at an entropy coefficient near **0.025**; the
  successful ones sit near **0.17**, an order of magnitude apart
  (`docs/plots/entropy_collapse.png`).
* The same basin appears in Isaac Lab — a different physics engine, contact
  solver and embodiment — where SAC from scratch reached grasp rate 1.00 and
  success 0.00 over 480 000 transitions. So it is a property of the task and the
  reward shaping, not of MuJoCo.

## Two hypotheses

**H1, the policy stopped exploring.** SAC tunes its entropy coefficient
automatically against a target entropy. That drives the coefficient towards zero
once the policy is confident, which is right when the policy is confident about
the correct behaviour and fatal when it has settled into a local optimum: at
alpha ≈ 0.02 the policy is effectively deterministic and simply never tries
anything else. Premature entropy collapse is a documented SAC failure mode, and
the usual remedy is to stop the coefficient falling that far.

**H2, the missing behaviour is temporally extended.** Lifting the box is roughly
twenty consecutive upward commands. SAC explores by sampling actions
independently at each step — white noise — and independent samples average out
to almost no net displacement, so a sustained lift is essentially never
sampled. Grasping, by contrast, is a *positional* behaviour that white noise
finds easily, which fits the symptom exactly: these policies grasp well and
lift never. Eberhard, Hollenstein, Pinneri and Martius, [*Pink Noise Is All You
Need: Colored Noise Exploration in Deep Reinforcement Learning*](https://openreview.net/forum?id=hQ9V5QN27eS)
(ICLR 2023), evaluate the colored-noise family on SAC and MPO and find pink
noise — halfway between white and Brownian — beats white noise, OU noise and the
rest across continuous control, and recommend it as the default.

An earlier, weaker version of H1 was tried first and failed: raising the target
entropy from −dim(A) to −dim(A)/2 roughly doubled the coefficient, to about
0.05, and none of the three stalled seeds escaped within 100 000 steps. That
failure is what motivated testing a hard floor rather than a nudge.

## The experiment

One variable at a time, on exactly the three seeds that stalled, nominal world,
**100 000 steps — half the baseline budget**, same evaluation protocol
(`experiments/exploration_ablation.py`):

| arm | seed 2 | seed 3 | seed 4 | solved |
| --- | ---: | ---: | ---: | ---: |
| baseline (200k steps) | 0.00 | 0.00 | 0.00 | 0/3 |
| **entropy floor, alpha ≥ 0.15** | **1.00** | **1.00** | **1.00** | **3/3** |
| pink exploration noise | 0.00 | 0.00 | 0.90 | 1/3 |
| both | 1.00 | 0.93 | 1.00 | 3/3 |

**H1 is the answer.** A floor under the entropy coefficient rescues every
stalled seed, at half the budget that failed without it. Pink noise rescues one
of three — real but not sufficient — and combining the two adds nothing over the
floor alone, which is what you would expect if the floor is the active
ingredient and correlated noise is a second-order help.

Extended to all five seeds, again at 100 000 steps:

| | per-seed | mean | 95% t across seeds | solved |
| --- | --- | ---: | --- | ---: |
| baseline, 200k steps | 1.0, 1.0, 0.0, 0.0, 0.0 | 0.400 | [0.000, 1.000] | 2/5 |
| **entropy floor, 100k steps** | 0.97, 1.0, 1.0, 1.0, 1.0 | **0.993** | **[0.975, 1.000]** | **5/5** |

The interval goes from carrying no information to being tight, and the two seeds
that already worked are not harmed by the floor.

## Reproducing

```bash
python experiments/exploration_ablation.py --jobs 6
python src/train_rl.py --steps 100000 --seed 2 --randomisation none \
    --hidden 128 --alpha-floor 0.15 --output experiments/runs/explore_alphafloor_s2
```

Results in `experiments/results/exploration.json`; per-run curves in
`experiments/runs/explore_*/progress.csv`.

## What this does and does not change

The headline tables elsewhere in this repository were produced **before** this
fix and are left as they are. They are a fair record of what SAC does on this
task with a standard configuration, and the from-scratch rows should be read
that way: as a demonstration of seed variance and of what demonstrations buy,
not as the best this algorithm can do here. Rerunning the full 50-run grid with
the floor would be about three hours of compute and would replace an honest
result with a better-looking one, without changing either conclusion the grid
supports.

What it does change is the standing of the finding itself. "Three of five seeds
stall and we do not know why" is now "three of five seeds stall because the
entropy coefficient collapses, and a floor of 0.15 fixes it in every seed at
half the budget".

## Under randomisation it needs the floor *and* a longer budget

The obvious next question — is this also why from-scratch SAC fails under
randomisation? — took two rounds to answer, and the first round was misleading.

At a matched budget the floor appears to do nothing. Five seeds at `medium`,
100 000 steps:

| condition | per-seed | mean | 95% t |
| --- | --- | ---: | --- |
| medium, no floor (200k) | 0.2, 0.03, 0.0, 0.27, 0.1 | 0.120 | [0.000, 0.259] |
| medium, floor (100k) | 0.0, 0.03, 0.37, 0.0, 0.4 | 0.160 | [0.000, 0.414] |

Two of those seeds were still climbing steeply when the run ended, which is the
tell. Extending three seeds to 300 000 steps:

| condition | per-seed | mean | 95% t |
| --- | --- | ---: | --- |
| medium, no floor, 200k (seeds 0, 2, 4) | 0.2, 0.0, 0.1 | 0.100 | [0.000, 0.348] |
| **medium, floor, 300k** | 0.60, 0.63, 0.77 | **0.667** | [0.448, 0.886] |

The intervals do not overlap. The floor works under randomisation too; it simply
needs about three times the budget to show it, because learning is slower when
every episode is a different world. The trajectory is visible in the curves —
seed 0 sat at 0.00 through 175 000 steps and reached 0.60 by 300 000.

So the honest conclusion is not "a different mechanism" but "the same fix, and
the nominal budget was never enough". Stopping at 100 000 steps would have
recorded the opposite, which is worth remembering next time an intervention
looks inert.

## How sensitive is it to the floor value?

0.15 was originally taken from what the successful seeds settle at (≈0.17)
rather than searched, which is a weak reason to trust a number. Swept over three
stalled seeds, 100 000 steps each. Final success saturates once the fix works at
all, so the informative column is how quickly a run first clears 0.5:

| floor | final per seed | mean | steps to >0.5 |
| ---: | --- | ---: | --- |
| 0.00 (baseline, 200k) | 0.00, 0.00, 0.00 | 0.000 | never |
| 0.05 | 0.87, 0.00, 0.00 | 0.289 | 70k, 1 of 3 |
| 0.10 | 1.00, 0.97, 1.00 | 0.989 | 43k, 3 of 3 |
| 0.15 | 1.00, 1.00, 1.00 | 1.000 | 40k, 3 of 3 |
| **0.30** | 0.97, 1.00, 1.00 | 0.989 | **23k**, 3 of 3 |
| 0.50 | 1.00, 1.00, 1.00 | 1.000 | 63k, 3 of 3 |

It is a **broad basin**, not a lucky point: anything from 0.10 to 0.50 rescues
all three seeds. Only 0.05 fails, and it is the value closest to the 0.025 the
collapsed runs settle at — which is the result the mechanism predicts.

There is a shallow speed optimum near 0.30, roughly 1.7 times faster to
threshold than 0.15 and nearly three times faster than 0.50. Both edges cost
something: too low and the policy still stops exploring, too high and it keeps
exploring when it should be exploiting.

The grid below was nevertheless run at **0.15**, not 0.30. The speed advantage
was measured on the nominal world only, and 0.15 is the value validated at five
seeds there *and* at `medium` over 300 000 steps. Choosing the faster value
would have been extrapolating a nominal-world result into the randomised
setting, and it would have discarded eight runs already on disk. Whether 0.30
keeps its advantage under randomisation is untested.

Two caveats worth keeping:

* Everything here is one task, one reward, one algorithm. A floor is not a
  general cure for premature convergence; it is a fix for this failure.
* 0.667 at `medium` is still short of what demonstrations achieve under the same
  randomisation (0.726, reached within 30 000 steps rather than 300 000).
