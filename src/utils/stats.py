"""Confidence intervals for success rates.

There are two different questions in this repository and they need two
different intervals, which is why both are implemented here.

**Within one seed**: "this policy succeeded 84 times out of 100 episodes, what
is the success rate of *this policy*?" That is a binomial proportion, and the
Wilson score interval is the right one: unlike the textbook normal
approximation it does not produce intervals that run past 0 or 1, and it stays
sensible at small counts and at rates near the extremes.

**Across seeds**: "five independently trained policies scored 0.84, 0.61, 0.79,
0.88, 0.55, what will the *next* training run score?" That is a question about
the training procedure, not about one network, and the variance between seeds
is usually far larger than the binomial variance within a seed. A t interval on
the per-seed rates is the honest answer, and it is the one quoted in the README.

Reporting only the within-seed interval is the standard way to make a
reinforcement-learning result look more certain than it is: it answers a
question nobody asked.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Sequence

import numpy as np

# Two-sided t critical values at 95% for small samples, indexed by dof.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def t_critical(dof: int, confidence: float = 0.95) -> float:
    """Two-sided t critical value. Falls back to the normal value for large dof."""
    if confidence != 0.95:
        raise ValueError("only the 95% level is tabulated; add values if needed")
    if dof <= 0:
        return float("inf")
    return _T95.get(dof, 1.96)


@dataclasses.dataclass
class Interval:
    """A point estimate with a two-sided interval."""

    point: float
    low: float
    high: float
    n: int
    method: str

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        return "{:.3f} [{:.3f}, {:.3f}]".format(self.point, self.low, self.high)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "wilson")
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), trials, "wilson")


def t_interval(values: Sequence[float], confidence: float = 0.95) -> Interval:
    """Student-t interval on the mean of per-seed rates."""
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "t")
    mean = float(arr.mean())
    if n == 1:
        return Interval(mean, float("nan"), float("nan"), 1, "t")
    sd = float(arr.std(ddof=1))
    half = t_critical(n - 1, confidence) * sd / math.sqrt(n)
    return Interval(mean, max(0.0, mean - half), min(1.0, mean + half), n, "t")


def bootstrap_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap on the mean, for when five seeds look non-normal."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(means, 100 * (1 - (1 - confidence) / 2)))
    return Interval(float(arr.mean()), lo, hi, arr.size, "bootstrap")


def summarise_seeds(
    per_seed_successes: Sequence[int],
    per_seed_trials: Sequence[int],
    confidence: float = 0.95,
) -> Dict[str, object]:
    """Full summary of a condition evaluated over several seeds.

    Returns the across-seed t interval (the headline number), the pooled Wilson
    interval (what you would get if you wrongly treated all episodes as one
    sample), and the per-seed Wilson intervals.
    """
    rates = [s / t for s, t in zip(per_seed_successes, per_seed_trials) if t > 0]
    pooled = wilson_interval(
        int(sum(per_seed_successes)), int(sum(per_seed_trials)), confidence
    )
    across = t_interval(rates, confidence)
    boot = bootstrap_interval(rates, confidence)
    per_seed: List[Dict] = [
        wilson_interval(int(s), int(t), confidence).as_dict()
        for s, t in zip(per_seed_successes, per_seed_trials)
    ]
    return {
        "across_seeds": across.as_dict(),
        "across_seeds_bootstrap": boot.as_dict(),
        "pooled_wilson": pooled.as_dict(),
        "per_seed": per_seed,
        "per_seed_rates": rates,
        "n_seeds": len(rates),
        "episodes_per_seed": list(per_seed_trials),
    }


def welch_t(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Welch's t statistic between two small samples of per-seed rates.

    Reported, not thresholded. With five seeds a p-value is a decoration; the
    statistic and the difference in means are what carry information.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size < 2 or y.size < 2:
        return {"diff": float(x.mean() - y.mean()), "t": float("nan"), "dof": float("nan")}
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    denom = math.sqrt(vx + vy)
    t_stat = float((x.mean() - y.mean()) / denom) if denom > 0 else float("inf")
    if denom > 0:
        dof = float((vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1)))
    else:
        dof = 0.0
    return {"diff": float(x.mean() - y.mean()), "t": t_stat, "dof": dof}
