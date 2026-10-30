#!/usr/bin/env bash
# Create a virtual environment and install everything needed to run this
# repository. CPU only: nothing here needs a GPU.
#
#   ./scripts/setup.sh              # into .venv
#   ./scripts/setup.sh /path/to/env
set -euo pipefail

VENV="${1:-.venv}"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "need Python 3.10 or newer; found $($PYTHON --version)" >&2
    exit 1
fi

echo "creating $VENV"
"$PYTHON" -m venv "$VENV"

if [ -f "$VENV/bin/activate" ]; then
    BIN="$VENV/bin"
else
    BIN="$VENV/Scripts"   # Windows
fi

"$BIN/python" -m pip install --upgrade pip
# The CPU wheel index keeps the download to a few hundred megabytes instead of
# pulling the whole CUDA runtime for networks with 128 hidden units.
"$BIN/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$BIN/python" -m pip install -r requirements.txt

echo
echo "checking the install"
"$BIN/python" -m pytest tests/ -q

cat <<MSG

done. activate it with:

    source $BIN/activate

then try:

    make demo          scripted expert, ten seconds
    make train-quick   a short SAC run, a few minutes
    make experiments   the whole grid, about three hours on eight cores
MSG
