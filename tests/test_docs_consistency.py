"""The documents must still agree with the results they quote.

Twice a superseded number survived in a summary while the section it summarised
was rewritten, and both times a person reading the published page caught it
rather than the diff. `scripts/check_docs.py` encodes both failures; this runs
it in CI so neither has to be caught by eye again.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_documents_agree_with_the_results_they_quote():
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "check_docs.py")],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "documentation is out of step with experiments/results:\n\n"
        + proc.stdout + proc.stderr
    )
