#!/usr/bin/env bash
# ============================================================================
# DataBuilder — terminal launcher for macOS and Linux.
#
# Activates .venv/ and runs `python -m dataset_sorter`.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -d ".venv" ]; then
    echo "✗ No .venv/ here — run ./install.sh first."
    exit 1
fi

# Silent auto-update — ignore failures (offline / detached HEAD).
git pull --ff-only >/dev/null 2>&1 || true

# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m dataset_sorter
