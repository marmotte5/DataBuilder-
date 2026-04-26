#!/usr/bin/env bash
# ============================================================================
# DataBuilder — macOS double-click launcher.
#
# Activates .venv/ and runs `python -m dataset_sorter`. Designed to be
# double-clicked from Finder (.command files auto-open in Terminal.app).
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -d ".venv" ]; then
    echo "✗ No .venv/ here — double-click install.command first."
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
fi

# Optional auto-update on launch — silent if offline.
git pull --ff-only >/dev/null 2>&1 || true

# shellcheck disable=SC1091
source .venv/bin/activate

python -m dataset_sorter
RC=$?

if [ $RC -ne 0 ]; then
    echo
    echo "DataBuilder exited with error code $RC."
    read -n 1 -s -r -p "Press any key to close this window..."
fi
