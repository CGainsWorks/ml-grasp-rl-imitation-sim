# Evaluation protocol

Every success rate in this repository comes from
[`src/utils/rollout.py:evaluate_policy`](../src/utils/rollout.py). There is one
definition, deliberately, so the number in a training curve, the number in the
ablation table and the number in the README cannot drift apart.

## The protocol

* **N episodes with explicit seeds.** Episode *i* uses seed `900000 + i`, so two
  policies are compared on exactly the same worlds — same box size, same mass,
  same friction, same starting pose. Comparing policies on independently drawn
  worlds throws away a large amount of statistical power for nothing.
* **Deterministic actions.** The mode of the policy, not a sample. Sampling at
  evaluation measures the exploration noise as well as the policy.
* **Success is read at the final step.** Not "at any point during the episode".
* **The seed block is disjoint from training.** Training resets draw from a
  block based at 0, training-time evaluation from 500 000, and final evaluation
  from 900 000.

## Reporting: two intervals, two questions

Both are computed, in [`src/utils/stats.py`](../src/utils/stats.py), and both
appear in the results files.

**Within one seed — Wilson.** "This policy succeeded 84 times in 100 episodes;
what is *this policy's* success rate?" That is a binomial proportion. The Wilson
score interval is used rather than the normal approximation because it does not
run past 0 or 1 and stays sensible near the extremes: at 0 successes out of 30
the normal interval has zero width, which is obviously wrong.

**Across seeds — Student t.** "Five independently trained policies scored 0.84,
0.61, 0.79, 0.88 and 0.55; what will the *next* training run score?" That is a
question about the training procedure, and it is the one a reader of a
reinforcement-learning result actually cares about. The interval is a t interval
on the per-seed rates.

The headline numbers in the README are the across-seed t intervals. The pooled
Wilson intervals are printed next to them in the results JSON, and they are
always narrower — often by a factor of two or three. Quoting only the pooled
interval is the standard way to make an RL result look more certain than it is:
it answers a question nobody asked, because pooling treats five different
policies as one.

A bootstrap interval on the per-seed means is also reported, since five samples
is few enough that normality is worth not assuming.

## Why five seeds

Because one is an anecdote and twenty was not affordable. `src/evaluate.py`
attaches an explicit warning to any single-seed result:

```json
"single_seed_warning": "one seed only: this is an anecdote, not a result"
```

Five seeds with 100 episodes each is 500 episodes per condition. The
seed-to-seed spread on this task is several times the binomial spread within a
seed, which is exactly why the two intervals are reported separately.

## Comparisons between methods

`experiments/summarise.py` reports Welch's t statistic and the difference in
means for the comparisons that matter (imitation-plus-RL against SAC alone, wide
randomisation against none on the shifted worlds). The statistic is reported
without a significance verdict attached: with five seeds per arm, a p-value is
decoration, and the honest summary is the effect size next to the spread.

## Reproducing

```bash
make expert-baseline      # scripted expert on every level
make evaluate             # the headline table
make ablation             # the randomisation ablation
```

Each writes a JSON into `experiments/results/` containing the per-seed numbers,
not just the aggregate, so any interval in this repository can be recomputed
from the file it came from.
