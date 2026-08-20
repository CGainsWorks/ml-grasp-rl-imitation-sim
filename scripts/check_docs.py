"""Do the documents still agree with the results they quote?

    python scripts/check_docs.py

This exists because of a specific, repeated failure. Twice now a number in this
repository has been superseded by a new run, the section reporting it has been
rewritten, and a *summary* of that section somewhere else has gone on asserting
the old conclusion -- once in the README while the tables above it were current,
once in the next-steps list of `limitations.md` two hundred lines below the body
that contradicted it. Both times the numbers were right and the prose was wrong,
and both times a human reading the published page caught it rather than the
diff, because a diff only shows the part that changed.

So this checks the part that did *not* change. Two rules:

``CLAIMS``     a headline number that must still appear, in the documents that
               quote it, at the value the evaluation JSON currently reports.
               Recomputed from `experiments/results/*_eval.json` on every run,
               so a rerun that moves a number fails the build until the prose
               moves with it.
``RETIRED``    a phrasing that used to be true and is not. This is the half that
               catches stale summaries: a claim can be absent from the results
               and still sit in a digest, where nothing recomputes it.

Neither rule is clever. Both encode the thing that actually went wrong, which
is worth more here than a general-purpose consistency checker that would need
to parse English.
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import re
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Student-t two-sided 95% multipliers, matching src/utils/stats.py.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262}

# label, level, the documents that quote it, and what it is called in prose.
CLAIMS = [
    # The README now summarises the wrist as four deltas and keeps the
    # per-cell numbers in limitations.md, so only that file quotes this.
    ("wristhold", "wrist_bench", ["docs/limitations.md"],
     "the wrist with a held cloning anchor"),
    ("campriv", "measured_camera", ["README.md", "docs/limitations.md"],
     "privileged distillation at camera-grade sensing"),
    ("camhist", "measured_camera", ["docs/limitations.md"],
     "a four-frame observation window at camera-grade sensing"),
    # The two controls. These are the rows whose absence let a wrong
    # conclusion ship, so they are the rows most worth pinning.
    # Matched-cap cells. nowristbc is NOT a control for the wrist rows -- it
    # runs at a different object size cap, which is the confound that produced
    # two wrong conclusions -- so each cap is pinned as its own pair.
    ("nowristbig", "wrist_bench", ["docs/limitations.md"],
     "no wrist at cap 0.034, matched against the wrist rows"),
    ("wristsmall", "wrist_bench", ["docs/limitations.md"],
     "wrist at cap 0.024, matched against the no-wrist rows"),
    ("wristsmallhold", "wrist_bench", ["docs/limitations.md"],
     "held-anchor RL with the wrist at cap 0.024"),
    ("nowristbc", "wrist_bench", ["docs/limitations.md"],
     "no wrist at cap 0.024"),
    ("heranchor", "medium", ["README.md", "docs/limitations.md"],
     "anchored fine-tuning at medium"),
    ("hermed", "medium", ["docs/limitations.md"],
     "the sparse recipe under medium randomisation, where it collapses"),
    ("herfinal", "none", ["README.md", "docs/limitations.md"],
     "sparse reward with hindsight and an annealed start curriculum"),
    ("percmed", "medium", ["docs/limitations.md"],
     "camera-in-the-loop RL under medium randomisation"),
    ("percscratch", "none", ["README.md", "docs/limitations.md"],
     "from-scratch RL with the camera in the loop"),
    ("percbc", "none", ["README.md", "docs/limitations.md"],
     "cloning with the pose estimator in the loop"),
    ("wristcam", "none", ["README.md", "docs/limitations.md"],
     "the wrist camera with clutter, nominal dynamics"),
    ("realsensor", "measured_camera_realsensor",
     ["README.md", "docs/limitations.md"],
     "the wrist camera with clutter at measured_camera's dynamics -- the "
     "controlled half of the noise-model comparison"),
    # The arm2* runs are the grid retrained on the corrected position servo.
    # The armgrid* runs are kept on disk but no longer quoted: they measured a
    # servo whose gain was scaled without its bias.
    ("arm2nonehold", "none", ["README.md", "docs/limitations.md"],
     "the arm with a held anchor at level none, corrected servo"),
    ("arm2mediumhold", "medium", ["docs/limitations.md"],
     "the arm with a held anchor at level medium, corrected servo"),
    ("arm2highhold", "high", ["README.md", "docs/limitations.md"],
     "the arm with a held anchor at level high, corrected servo"),
    ("fsns", "wrist_bench", ["docs/limitations.md"],
     "from scratch, no wrist, cap 0.024"),
    ("fsws", "wrist_bench", ["docs/limitations.md"],
     "from scratch, wrist, cap 0.024"),
    ("camord", "measured_camera", ["README.md", "docs/limitations.md"],
     "the ordinary-demonstrator control at camera-grade sensing"),
]

# Phrasings that were true once. Each is a regex, the files it must not appear
# in, and the result that retired it.
RETIRED = [
    (r"randomised levels have not been tried through the camera",
     ["README.md", "docs/limitations.md"],
     "they have: 0.000 at `medium` with grasp 0.55-0.65"),
    (r"from-scratch RL through the camera is left undone",
     ["README.md", "docs/limitations.md"],
     "it reaches 0.950 [0.875, 1.000]; the obstacle was a "
     "missing --perception flag on train_rl, not the cost"),
    (r"Shaping here buys segments and does not chain them",
     ["README.md", "docs/limitations.md"],
     "sparse reward plus hindsight plus an annealed start curriculum "
     "reaches 0.944, above the demonstration-seeded 0.916"),
    (r"hindsight experience replay inapplicable",
     ["docs/limitations.md"],
     "the relabeller always carried the latch; the zero was "
     "exploration, and with a curriculum it fires on half of all "
     "transitions"),
    (r"[Pp]erception is a pipeline, on the easy camera",
     ["README.md"],
     "the wrist camera with clutter reaches 0.960 at nominal dynamics "
     "and 0.728 at measured_camera's"),
    # Anchored to the arm specifically. The bare phrase "does not survive
    # randomisation" is *true* of the sparse hindsight recipe, which collapses
    # to 0.000 at `medium`, so retiring the phrase alone flagged a correct
    # sentence about a different subject. A retirement has to name the claim,
    # not the wording.
    (r"grid of its own,? and it does not survive randomisation",
     ["README.md", "docs/limitations.md"],
     "the arm's collapse to 0.052 was a position-servo bug; corrected, it "
     "declines mildly from 0.530 to 0.354 across the range"),
    (r"Isaac Lab runs both tasks and still supplies no headline number",
     ["README.md"],
     "the Isaac port now carries a four-level, five-seed sweep on both "
     "arms: from scratch 0.000 everywhere, demonstration-seeded 0.969 "
     "to 0.041"),
    (r"Perception is a check, not a pipeline",
     ["README.md"],
     "a policy trained and evaluated through the estimator reaches "
     "0.934 [0.900, 0.968]; substituting it costs 18 points, training "
     "through it costs about four"),
    (r"[Ee]very policy here scores ~0\.00",
     ["README.md", "docs/limitations.md"],
     "privileged distillation reaches 0.406 at measured_camera"),
    # NOTE: an earlier version of this file retired "the wrist does not help".
    # The control reinstated it -- 0.838 without the wrist against 0.478 with
    # it -- so the entry was removed rather than kept and inverted. A registry
    # of retired claims can itself go stale, and this is what that looks like.
    (r"wrist is learnable but not discoverable",
     ["README.md", "docs/limitations.md"],
     "superseded by the matched-cap grid"),
    (r"scores \*\*0\.000\*\* against 0\.122 without it",
     ["README.md"],
     "matched, both hands score 0.000 at cap 0.034; the zero was object size"),
    (r"identical pipeline \*?without\*? the wrist reaches \*\*0\.838",
     ["README.md"],
     "0.838 vs 0.478 compared two different object size caps; matched, the "
     "deltas are -0.034, -0.046, +0.014, +0.074"),
    (r"cloning reaches 0\.448\b",
     ["README.md"],
     "the arm reaches 0.536 with a held anchor"),
]


def aggregate(label: str, level: str):
    """Mean and 95% interval across seeds, or None if the file is absent."""
    path = os.path.join(REPO, "experiments", "results", label + "_eval.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    xs = [r["levels"][level]["success_rate"]
          for r in data["runs"] if level in r["levels"]]
    if not xs:
        return None
    n = len(xs)
    mean = st.mean(xs)
    if n < 2:
        return mean, mean, mean, n
    half = T95.get(n, 2.776) * st.stdev(xs) / math.sqrt(n)
    return mean, max(mean - half, 0.0), min(mean + half, 1.0), n


def main() -> int:
    failures = []

    for label, level, files, prose in CLAIMS:
        agg = aggregate(label, level)
        if agg is None:
            failures.append(
                "{}: no evaluation at level {} -- the documents quote a number "
                "nothing recomputes".format(label, level))
            continue
        mean, lo, hi, n = agg
        wanted = "{:.3f}".format(mean)
        for rel in files:
            text = io.open(os.path.join(REPO, rel), encoding="utf-8").read()
            if wanted not in text:
                failures.append(
                    "{} quotes {} ({}) but the current value over {} seeds is "
                    "{} [{:.3f}, {:.3f}] and does not appear".format(
                        rel, prose, label, n, wanted, lo, hi))

    for pattern, files, why in RETIRED:
        for rel in files:
            text = io.open(os.path.join(REPO, rel), encoding="utf-8").read()
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    failures.append(
                        "{}:{} still says \"{}\" -- retired because {}".format(
                            rel, i, line.strip()[:70], why))

    # A claim is only checkable if something recomputes it. Results that no
    # document quotes are not an error -- most runs are working material -- but
    # they are worth naming, because the registry going stale is the same
    # failure one level up.
    known = {c[0] for c in CLAIMS}
    unregistered = sorted(
        os.path.basename(p)[: -len("_eval.json")]
        for p in glob.glob(os.path.join(
            REPO, "experiments", "results", "*_eval.json"))
        if os.path.basename(p)[: -len("_eval.json")] not in known)

    # Every clip the README shows must exist. A missing GIF is a broken image
    # on the front page, and nothing else in the build would notice.
    readme = io.open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    for ref in sorted(set(re.findall(r"\((videos/[^)]+\.gif)\)", readme))):
        if not os.path.exists(os.path.join(REPO, ref)):
            failures.append(
                "README.md shows {} and the file does not exist".format(ref))

    if failures:
        print("documents disagree with the results they quote:\n")
        for f in failures:
            print("  * " + f)
        print("\n{} problem(s). Either the prose is stale or the registry in "
              "scripts/check_docs.py is.".format(len(failures)))
        return 1

    print("checked {} headline claims and {} retired phrasings: consistent"
          .format(len(CLAIMS), len(RETIRED)))
    print("{} evaluated groups are not quoted by any document"
          .format(len(unregistered)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
