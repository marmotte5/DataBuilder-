#!/usr/bin/env bash
# Launch DataBuilder (macOS / Linux)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run install.sh first."
    echo "  chmod +x install.sh && ./install.sh"
    exit 1
fi

python dataset_sorter.py
