"""Regenerate the README's results tables from the files in ``experiments/results``.

    python analysis/readme_tables.py            # rewrite README.md in place
    python analysis/readme_tables.py --stdout   # just print them

The tables live between HTML comment markers in README.md and are replaced
wholesale. Nothing here recomputes a success rate: it reads the JSON that
``make evaluate`` and ``make ablation`` wrote. If a table looks stale, the fix
is to rerun those, not to edit the README.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "experiments", "results")
README = os.path.join(REPO, "README.md")

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _load(name: str) -> Optional[Dict]:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cell(interval: Dict) -> str:
    if interval is None or interval.get("point") is None:
        return "--"
    point = interval["point"]
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None or low != low:  # NaN check
        return "{:.3f}".format(point)
    # Three decimals: on the shifted column the difference between 0.002 and
    # 0.072 is the whole result, and two decimals rounds both to "0.00".
    return "**{:.3f}** [{:.3f}, {:.3f}]".format(point, low, high)


def headline_table(summary: Dict) -> List[str]:
    levels = summary["levels"]
    lines = [
        "### Success rate by method",
        "",
        "{} episodes per seed, deterministic actions, success read at the final step. "
        "Cells are the mean over seeds with a 95% t interval across seeds.".format(
            summary["episodes_per_seed"]),
        "",
        "| method | seeds | " + " | ".join("eval: `{}`".format(lv) for lv in levels) + " |",
        "| --- | ---: | " + " | ".join("---" for _ in levels) + " |",
    ]

    expert = summary["expert"]["levels"]
    lines.append("| scripted expert (reference) | -- | " + " | ".join(
        "{:.3f}".format(expert[lv]["success_rate"]) for lv in levels) + " |")

    for key, method in summary["methods"].items():
        cells = [_cell(method["levels"][lv]["across_seeds"]) for lv in levels]
        lines.append("| {} | {} | {} |".format(
            method.get("label", key), method["n_seeds"], " | ".join(cells)))

    lines += [
        "",
        "`none` is the nominal world, `medium` a training-like distribution, and "
        "`shifted` the held-out worlds that stand in for a real robot "
        "([why that is a proxy](docs/sim-to-real.md)).",
    ]
    return lines


def ablation_table(ablation: Dict, title: str, note: str) -> List[str]:
    lines = [
        "### " + title,
        "",
        note,
        "",
        "| trained with | seeds | on its own distribution | on `shifted` | gap |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in ablation["rows"]:
        lines.append("| `{}` | {} | {} | {} | {:+.3f} |".format(
            row["level"], row["n_seeds"],
            _cell(row["train_level"]["across_seeds"]),
            _cell(row["shifted"]["across_seeds"]),
            row["gap"]))
    comparisons = ablation.get("comparisons", {})
    if comparisons:
        lines.append("")
        for name, stat in comparisons.items():
            lines.append("* `{}`: difference {:+.3f} in mean success, Welch t = {:.2f}".format(
                name, stat["diff"], stat["t"]))
    return lines


def expert_table(baseline: Dict) -> List[str]:
    lines = [
        "### Scripted expert, for reference",
        "",
        "{} episodes per level, 95% Wilson intervals (one policy, so the binomial "
        "interval is the right one).".format(baseline["episodes"]),
        "",
        "| level | success | grasp rate | mean peak lift |",
        "| --- | --- | ---: | ---: |",
    ]
    for level, row in baseline["levels"].items():
        lines.append("| `{}` | **{:.3f}** [{:.3f}, {:.3f}] | {:.2f} | {:.3f} m |".format(
            level, row["success_rate"], row["wilson_low"], row["wilson_high"],
            row["grasp_rate"], row["mean_max_lift"]))
    return lines


def build() -> str:
    blocks: List[str] = []

    summary = _load("summary.json")
    if summary:
        blocks.append("\n".join(headline_table(summary)))

    ablation = _load("ablation.json")
    if ablation:
        blocks.append("\n".join(ablation_table(
            ablation,
            "Randomisation ablation: SAC from scratch",
            "Every policy evaluated twice: on the distribution it trained on, and on "
            "the held-out `shifted` worlds. The gap is what matters -- a policy that "
            "scores well on its own distribution and badly on `shifted` has learned "
            "the simulator rather than the task.",
        )))

    ablation_floor = _load("ablation_sacfloor.json")
    if ablation_floor:
        blocks.append("\n".join(ablation_table(
            ablation_floor,
            "Randomisation ablation: SAC from scratch, with a tuned entropy floor",
            "The same runs as the table above with one line changed -- a floor under "
            "the entropy coefficient, at the value that works for each level "
            "([docs/exploration.md](docs/exploration.md)). Same 200 000-step budget. "
            "Fixing the collapse roughly triples the own-distribution column and "
            "leaves the gap where it was, which is worth knowing: the poor transfer "
            "is not an artefact of an undertrained baseline.",
        )))

    ablation_bcrl = _load("ablation_bcrl.json")
    if ablation_bcrl:
        blocks.append("\n".join(ablation_table(
            ablation_bcrl,
            "Randomisation ablation: imitation-seeded SAC",
            "The same ablation for the imitation-plus-RL variant, which starts from a "
            "cloned policy and keeps the demonstrations pinned in the replay buffer.",
        )))

    baseline = _load("expert_baseline.json")
    if baseline:
        blocks.append("\n".join(expert_table(baseline)))

    if not blocks:
        return "*No results yet. Run `make experiments`.*"
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    tables = build()
    if args.stdout:
        print(tables)
        return

    with open(README, "r", encoding="utf-8") as fh:
        text = fh.read()
    if START not in text or END not in text:
        raise SystemExit("README.md is missing the {} / {} markers".format(START, END))
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    with open(README, "w", encoding="utf-8") as fh:
        fh.write(head + START + "\n\n" + tables + "\n\n" + END + tail)
    print("rewrote the results section of README.md")


if __name__ == "__main__":
    main()
