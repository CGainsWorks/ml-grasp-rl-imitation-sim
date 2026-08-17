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
    ("wristhold", "wrist_bench", ["README.md", "docs/limitations.md"],
     "the wrist with a held cloning anchor"),
    ("campriv", "measured_camera", ["README.md", "docs/limitations.md"],
     "privileged distillation at camera-grade sensing"),
    ("camhist", "measured_camera", ["docs/limitations.md"],
     "a four-frame observation window at camera-grade sensing"),
    # The two controls. These are the rows whose absence let a wrong
    # conclusion ship, so they are the rows most worth pinning.
    ("nowristbc", "wrist_bench", ["README.md", "docs/limitations.md"],
     "the no-wrist control on the wrist benchmark"),
    ("camord", "measured_camera", ["README.md", "docs/limitations.md"],
     "the ordinary-demonstrator control at camera-grade sensing"),
]

# Phrasings that were true once. Each is a regex, the files it must not appear
# in, and the result that retired it.
RETIRED = [
    (r"[Ee]very policy here scores ~0\.00",
     ["README.md", "docs/limitations.md"],
     "privileged distillation reaches 0.406 at measured_camera"),
    # NOTE: an earlier version of this file retired "the wrist does not help".
    # The control reinstated it -- 0.838 without the wrist against 0.478 with
    # it -- so the entry was removed rather than kept and inverted. A registry
    # of retired claims can itself go stale, and this is what that looks like.
    (r"wrist is learnable but not discoverable",
     ["README.md", "docs/limitations.md"],
     "the no-wrist control reaches 0.838 [0.811, 0.865]; learnable is not "
     "the same as useful and that framing implied it was"),
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
