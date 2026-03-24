#!/usr/bin/env bash
# Double-click launcher for macOS (Finder)
# .command files auto-open in Terminal.app when double-clicked

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Mise à jour..."
git pull --ff-only 2>/dev/null || echo "Pas de connexion, lancement avec la version locale"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo ""
    echo "Virtual environment not found."
    echo "Please run install.sh first:"
    echo ""
    echo "  Open Terminal, then:"
    echo "  cd $SCRIPT_DIR"
    echo "  chmod +x install.sh && ./install.sh"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

python dataset_sorter.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error. Check the output above."
    read -p "Press Enter to close..."
fi
