#!/usr/bin/env bash
# The one command in the README. Runs the scripted expert on the nominal world,
# prints a success rate with a confidence interval, and traces one episode.
#
# No trained policy needed, no GPU, no display, about ten seconds.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
exec "$PYTHON" scripts/offline_demo.py --episodes 25 --randomisation none --trace "$@"
