"""Every figure in the README, generated from the files in ``experiments/``.

    python analysis/plots.py --all

Each function reads results that a training or evaluation script wrote and
writes a PNG into ``docs/plots``. Nothing here recomputes a number: if a figure
and the README disagree, the figure is stale and ``make plots`` fixes it.

Matplotlib only, no seaborn, no style sheets to install.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.stats import t_interval  # noqa: E402

PLOT_DIR = os.path.join("docs", "plots")
RUN_DIR = os.path.join("experiments", "runs")
RESULT_DIR = os.path.join("experiments", "results")

LEVEL_COLOURS = {
    "none": "#c0392b",
    "low": "#e67e22",
    "medium": "#2980b9",
    "high": "#27ae60",
    "shifted": "#8e44ad",
    "bc": "#7f8c8d",
    "bc_rl": "#16a085",
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _read_progress(run: str) -> Optional[Dict[str, np.ndarray]]:
    path = os.path.join(run, "progress.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        rows = [line.strip().split(",") for line in fh if line.strip()]
    if not rows:
        return None
    data = np.asarray(rows, dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def _group_curves(pattern: str) -> Optional[Dict[str, np.ndarray]]:
    """Mean and 95% t interval of the success curve over every seed matching."""
    runs = sorted(glob.glob(pattern))
    curves = [_read_progress(r) for r in runs]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    length = min(len(c["step"]) for c in curves)
    steps = curves[0]["step"][:length]
    success = np.stack([c["success_rate"][:length] for c in curves])
    means, lows, highs = [], [], []
    for col in range(length):
        interval = t_interval(success[:, col])
        means.append(interval.point)
        lows.append(interval.low if np.isfinite(interval.low) else interval.point)
        highs.append(interval.high if np.isfinite(interval.high) else interval.point)
    return {
        "step": steps, "mean": np.asarray(means),
        "low": np.asarray(lows), "high": np.asarray(highs),
        "n_seeds": len(curves), "per_seed": success,
    }


# --------------------------------------------------------------------------
def plot_training_curves(out: str = "training_curves.png") -> Optional[str]:
    """Success against environment steps: from scratch, seeded, and side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    plotted = False

    for level in ("none", "low", "medium", "high"):
        group = _group_curves(os.path.join(RUN_DIR, "sac_{}_s*".format(level)))
        if group is None:
            continue
        plotted = True
        colour = LEVEL_COLOURS[level]
        axes[0].plot(group["step"], group["mean"], color=colour,
                     label="{} (n={})".format(level, group["n_seeds"]), linewidth=1.8)
        axes[0].fill_between(group["step"], group["low"], group["high"],
                             color=colour, alpha=0.15, linewidth=0)

    for level in ("none", "low", "medium", "high"):
        group = _group_curves(os.path.join(RUN_DIR, "bcrl_{}_s*".format(level)))
        if group is None:
            continue
        plotted = True
        colour = LEVEL_COLOURS[level]
        axes[1].plot(group["step"], group["mean"], color=colour,
                     label="{} (n={})".format(level, group["n_seeds"]), linewidth=1.8)
        axes[1].fill_between(group["step"], group["low"], group["high"],
                             color=colour, alpha=0.15, linewidth=0)

    for label, pattern, colour in (
        ("SAC from scratch", "sac_medium_s*", LEVEL_COLOURS["none"]),
        ("BC + SAC", "bcrl_medium_s*", LEVEL_COLOURS["bc_rl"]),
    ):
        group = _group_curves(os.path.join(RUN_DIR, pattern))
        if group is None:
            continue
        plotted = True
        axes[2].plot(group["step"], group["mean"], color=colour,
                     label="{} (n={})".format(label, group["n_seeds"]), linewidth=1.8)
        axes[2].fill_between(group["step"], group["low"], group["high"],
                             color=colour, alpha=0.15, linewidth=0)

    if not plotted:
        plt.close(fig)
        return None

    _style(axes[0], "SAC from scratch", "environment steps", "success rate")
    _style(axes[1], "SAC seeded with demonstrations", "environment steps", "success rate")
    _style(axes[2], "Both, at medium randomisation", "environment steps", "success rate")
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Training curves: mean over seeds, band is the 95% t interval across seeds",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def plot_ablation(out: str = "randomisation_ablation.png") -> Optional[str]:
    """Own-distribution against held-out success, from scratch and demonstration-seeded."""
    panels = []
    for name, title in (("ablation.json", "SAC from scratch"),
                        ("ablation_bcrl.json", "SAC seeded with demonstrations")):
        path = os.path.join(RESULT_DIR, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                panels.append((title, json.load(fh)))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.3), squeeze=False)
    width = 0.36

    for ax, (title, blob) in zip(axes[0], panels):
        levels = [row["level"] for row in blob["rows"]]
        x = np.arange(len(levels))
        for offset, key, label, colour in (
            (-width / 2, "train_level", "its own training distribution", "#2980b9"),
            (width / 2, "shifted", "held-out shifted worlds", "#8e44ad"),
        ):
            means = [row[key]["across_seeds"]["point"] for row in blob["rows"]]
            lows = [max(0.0, row[key]["across_seeds"]["point"] - row[key]["across_seeds"]["low"])
                    for row in blob["rows"]]
            highs = [max(0.0, row[key]["across_seeds"]["high"] - row[key]["across_seeds"]["point"])
                     for row in blob["rows"]]
            ax.bar(x + offset, means, width, label=label, color=colour, alpha=0.85)
            ax.errorbar(x + offset, means, yerr=[lows, highs], fmt="none",
                        ecolor="#2c3e50", capsize=3, linewidth=1.0)
        _style(ax, "{}  ({} seeds, {} episodes each)".format(
            title, blob.get("n_seeds", "?"), blob.get("episodes", "?")),
            "randomisation used during training", "success rate")
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Domain randomisation ablation: bars are the mean over seeds, "
                 "whiskers the 95% t interval", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def plot_reward_terms(out: str = "reward_terms.png") -> Optional[str]:
    """Per-term reward along one expert rollout, to show what the shaping does."""
    from envs.mujoco.grasp_env import make_env
    from src.policies.scripted_expert import ScriptedExpert
    from src.rewards.grasp_reward import RewardTerms

    env = make_env("none", seed=3, max_steps=100)
    expert = ScriptedExpert()
    obs, _ = env.reset(seed=900_000)
    names = [n for n in RewardTerms.names() if n != "time"]
    series: Dict[str, List[float]] = {n: [] for n in names}
    total: List[float] = []
    phases: List[int] = []

    while True:
        action = expert.act(obs)
        phases.append(expert.phase)
        obs, reward, terminated, truncated, info = env.step(action)
        for name in names:
            series[name].append(info["reward_terms"][name])
        total.append(reward)
        if terminated or truncated:
            break
    env.close()

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    steps = np.arange(len(total))
    for name in names:
        values = np.asarray(series[name])
        if np.allclose(values, 0.0):
            continue
        ax.plot(steps, values, linewidth=1.4, label=name)
    ax.plot(steps, total, color="black", linewidth=2.0, label="total", zorder=5)

    boundaries = [i for i in range(1, len(phases)) if phases[i] != phases[i - 1]]
    for boundary in boundaries:
        ax.axvline(boundary, color="#95a5a6", linestyle=":", linewidth=1.0)
    # Headroom for the phase labels, so they sit inside the axes rather than
    # colliding with the title.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.18 * (high - low))
    labels = ["approach", "descend", "close", "lift"]
    edges = [0] + boundaries + [len(total)]
    for i in range(min(len(labels), len(edges) - 1)):
        ax.text((edges[i] + edges[i + 1]) / 2, ax.get_ylim()[1], labels[i],
                ha="center", va="top", fontsize=8, color="#7f8c8d")

    _style(ax, "Reward terms along one expert rollout", "environment step", "reward contribution")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    return _save(fig, out)


def plot_bc_data_efficiency(out: str = "bc_data_efficiency.png") -> Optional[str]:
    path = os.path.join(RESULT_DIR, "bc_data_efficiency.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)

    sizes = [row["demo_episodes"] for row in blob["rows"]]
    means = [row["success"]["point"] for row in blob["rows"]]
    lows = [row["success"]["low"] for row in blob["rows"]]
    highs = [row["success"]["high"] for row in blob["rows"]]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(sizes, means, "o-", color="#7f8c8d", linewidth=1.8, label="behaviour cloning")
    ax.fill_between(sizes, lows, highs, color="#7f8c8d", alpha=0.18, linewidth=0)
    if blob.get("expert_success") is not None:
        ax.axhline(blob["expert_success"], color="#2c3e50", linestyle="--", linewidth=1.2,
                   label="scripted expert")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(-0.03, 1.03)
    _style(ax, "Behaviour cloning against demonstration count", "demonstration episodes",
           "success rate")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def plot_dagger(out: str = "dagger_rounds.png") -> Optional[str]:
    runs = sorted(glob.glob(os.path.join(RUN_DIR, "dagger_s*")))
    histories = []
    for run in runs:
        path = os.path.join(run, "result.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            histories.append(json.load(fh)["history"])
    if not histories:
        return None

    length = min(len(h) for h in histories)
    rounds = np.arange(length)
    success = np.stack([[h[i]["success_rate"] for i in range(length)] for h in histories])
    means, lows, highs = [], [], []
    for col in range(length):
        interval = t_interval(success[:, col])
        means.append(interval.point)
        lows.append(interval.low if np.isfinite(interval.low) else interval.point)
        highs.append(interval.high if np.isfinite(interval.high) else interval.point)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(rounds, means, "o-", color="#16a085", linewidth=1.8,
            label="DAgger (n={})".format(len(histories)))
    ax.fill_between(rounds, lows, highs, color="#16a085", alpha=0.18, linewidth=0)
    ax.set_xticks(rounds)
    ax.set_xticklabels(["BC"] + ["round {}".format(i) for i in range(1, length)])
    ax.set_ylim(-0.03, 1.03)
    _style(ax, "DAgger rounds on the shifted worlds", "", "success rate")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def plot_randomisation_ranges(out: str = "randomisation_ranges.png") -> Optional[str]:
    """What each level actually perturbs, drawn as ranges around the nominal value."""
    from src.randomisation.domain_rand import NOMINAL, load_randomisation

    levels = ["low", "medium", "high", "shifted"]
    configs = {level: load_randomisation(level) for level in levels}
    params = sorted({k for cfg in configs.values() for k in cfg.params})

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    offsets = {"low": 0.24, "medium": 0.08, "high": -0.08, "shifted": -0.24}

    for level in levels:
        cfg = configs[level]
        colour = LEVEL_COLOURS[level]
        drawn = False
        for i, name in enumerate(params):
            spec = cfg.params.get(name)
            if spec is None:
                continue
            nominal = getattr(NOMINAL, name)
            if spec.mode == "scale":
                lo = 1.0 + (spec.low - 1.0) * cfg.scale
                hi = 1.0 + (spec.high - 1.0) * cfg.scale
            else:
                mid = 0.5 * (spec.low + spec.high)
                half = 0.5 * (spec.high - spec.low) * cfg.scale
                scale_base = nominal if nominal else max(abs(spec.high), 1e-6)
                lo = (mid - half) / scale_base if nominal else 0.0
                hi = (mid + half) / scale_base if nominal else 1.0
            y = i + offsets[level]
            ax.plot([lo, hi], [y, y], color=colour, linewidth=3.0, solid_capstyle="butt",
                    label=level if not drawn else None)
            drawn = True

    ax.axvline(1.0, color="#2c3e50", linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=8)
    ax.set_xlabel("multiple of the nominal value (additive parameters shown as an "
                  "absolute band)", fontsize=8)
    _style(ax, "Randomisation ranges by level", ax.get_xlabel(), "")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    return _save(fig, out)


def plot_entropy_collapse(out: str = "entropy_collapse.png") -> Optional[str]:
    """Why some from-scratch seeds never take off.

    One point per SAC run: the entropy coefficient it settled at against the
    success rate it reached. The seeds that solve the task are the ones whose
    entropy coefficient stayed an order of magnitude higher; the ones that
    stall have collapsed to a nearly deterministic policy sitting in the
    "grasp the box and hold it on the table" optimum, which the reward pays
    0.73 per step for against 9.75 at the hold point.
    """
    points = []
    for level in ("none", "low", "medium", "high"):
        for run in sorted(glob.glob(os.path.join(RUN_DIR, "sac_{}_s*".format(level)))):
            curve = _read_progress(run)
            if curve is None or len(curve["step"]) < 2:
                continue
            points.append((level, float(curve["alpha"][-1]), float(curve["success_rate"][-1]),
                           float(curve["grasp_rate"][-1])))
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for level in ("none", "low", "medium", "high"):
        xs = [p[1] for p in points if p[0] == level]
        ys = [p[2] for p in points if p[0] == level]
        if not xs:
            continue
        ax.scatter(xs, ys, s=46, color=LEVEL_COLOURS[level], alpha=0.8,
                   edgecolor="white", linewidth=0.8, label=level)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Entropy coefficient at the end of training against success",
           "final entropy coefficient (log scale)", "final success rate")
    ax.legend(fontsize=8, frameon=False, title="randomisation", title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def plot_floor_control(out: str = "entropy_floor.png") -> Optional[str]:
    """What the entropy floor actually buys, once the budgets are matched.

    Left: success by randomisation level, with and without the floor, both arms
    at 300 000 steps and five seeds. Bars are the mean, whiskers the 95% t
    interval, dots the individual seeds -- the dots matter here, because the
    floor's real effect at `medium` is that they stop being spread from 0 to
    0.77.

    Right: the `low` regression, which is the reason the left panel is not the
    whole story. Success and grasp rate against the floor value, on the level
    where a floor of 0.15 stops the policy learning to close on the box at all.
    """
    control_path = os.path.join(RESULT_DIR, "floor_control.json")
    if not os.path.exists(control_path):
        return None
    with open(control_path, "r", encoding="utf-8") as fh:
        control = json.load(fh)

    low_path = os.path.join(RESULT_DIR, "low_anomaly.json")
    low = None
    if os.path.exists(low_path):
        with open(low_path, "r", encoding="utf-8") as fh:
            low = json.load(fh)

    fig, axes = plt.subplots(1, 2 if low else 1, figsize=(11.6 if low else 6.4, 4.3),
                             squeeze=False)
    ax = axes[0][0]
    levels = [row["level"] for row in control["rows"]]
    x = np.arange(len(levels))
    width = 0.36
    for offset, key, label, colour in (
        (-width / 2, "control_300k", "no floor", "#7f8c8d"),
        (width / 2, "with_floor_300k", "entropy floor 0.15", "#2980b9"),
    ):
        means = [row[key]["point"] for row in control["rows"]]
        lows = [max(0.0, row[key]["point"] - row[key]["low"]) for row in control["rows"]]
        highs = [max(0.0, row[key]["high"] - row[key]["point"]) for row in control["rows"]]
        ax.bar(x + offset, means, width, label=label, color=colour, alpha=0.85)
        ax.errorbar(x + offset, means, yerr=[lows, highs], fmt="none",
                    ecolor="#2c3e50", capsize=3, linewidth=1.0)
        for i, row in enumerate(control["rows"]):
            seeds = row[key]["per_seed"]
            ax.scatter(np.full(len(seeds), x[i] + offset), seeds, s=16, zorder=3,
                       color="#2c3e50", alpha=0.7, linewidth=0)
    _style(ax, "Matched at 300 000 steps, five seeds",
           "randomisation level", "success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, frameon=False)

    if low:
        ax = axes[0][1]
        floors = [row["alpha_floor"] for row in low["rows"]]
        pos = np.arange(len(floors))
        success = [row["across_seeds"]["point"] for row in low["rows"]]
        grasp = [row["mean_grasp"] for row in low["rows"]]
        ax.plot(pos, grasp, marker="o", color="#e67e22", label="grasp rate")
        ax.plot(pos, success, marker="s", color="#2980b9", label="success rate")
        for i, row in enumerate(low["rows"]):
            seeds = row["success_per_seed"]
            ax.scatter(np.full(len(seeds), pos[i]), seeds, s=16, zorder=3,
                       color="#2c3e50", alpha=0.7, linewidth=0)
        _style(ax, "`low` randomisation: the floor value matters, and 0.15 is wrong",
               "entropy coefficient floor", "rate")
        ax.set_xticks(pos)
        ax.set_xticklabels(["{:.2f}".format(f) for f in floors])
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("The entropy floor rescues the nominal world and does not "
                 "generalise across randomisation levels", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def _save(fig, name: str) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote " + path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of: curves ablation reward bc dagger ranges entropy floor")
    args = parser.parse_args()

    jobs = {
        "curves": plot_training_curves,
        "ablation": plot_ablation,
        "reward": plot_reward_terms,
        "bc": plot_bc_data_efficiency,
        "dagger": plot_dagger,
        "ranges": plot_randomisation_ranges,
        "entropy": plot_entropy_collapse,
        "floor": plot_floor_control,
    }
    chosen: Sequence[str] = args.only or list(jobs)
    for name in chosen:
        if name not in jobs:
            raise SystemExit("unknown figure: " + name)
        if jobs[name]() is None:
            print("skipped {}: inputs not present yet".format(name))


if __name__ == "__main__":
    main()
