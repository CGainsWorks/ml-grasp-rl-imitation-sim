"""Write run records with repository-relative POSIX paths on every platform.

Run directories and dataset paths reach the JSON records through ``glob`` and
``os.path.join``, so a run made on Windows used to record
``experiments/runs\\sac_none_s0`` and one made on Linux ``experiments/runs/...``.
The committed records are read on both. A mixed separator is also how an
absolute local path slips into a public file unnoticed. ``tests/test_committed_paths.py``
holds the committed files to this; :func:`dump_json` keeps new ones to it.
"""

from __future__ import annotations

import json
import re
from typing import Any, IO

# A relative path with at least one Windows separator and nothing else in it.
_WINDOWS_RELATIVE = re.compile(r"[A-Za-z0-9_.\-/]+(?:\\[A-Za-z0-9_.\-/]+)+")


def posix_paths(obj: Any) -> Any:
    """Copy of ``obj`` with every path-like string using ``/`` as the separator."""
    if isinstance(obj, str):
        return obj.replace("\\", "/") if _WINDOWS_RELATIVE.fullmatch(obj) else obj
    if isinstance(obj, dict):
        return {k: posix_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [posix_paths(v) for v in obj]
    return obj


def dump_json(obj: Any, fh: IO[str], **kwargs: Any) -> None:
    """``json.dump`` with :func:`posix_paths` applied first."""
    json.dump(posix_paths(obj), fh, **kwargs)
