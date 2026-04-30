#!/usr/bin/env bash
# ============================================================================
# DataBuilder — macOS double-click installer.
#
# Finder runs ".command" files in Terminal when double-clicked. This is the
# trick that turns the cross-platform install.sh into a true one-click
# experience on macOS.
#
# Notes:
#   • When double-clicked, the working directory is the user's HOME, not
#     this file's directory — so we cd to the script's location first.
#   • We trap exit so the Terminal window stays open on errors. Otherwise
#     Finder closes it instantly and the user sees nothing.
# ============================================================================

# cd into the directory this script lives in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

clear

# Run the real installer.
bash ./install.sh
RC=$?

echo
if [ $RC -eq 0 ]; then
    # Make sure the .app launcher inside the bundle is executable.
    # git on Windows / fresh checkouts can drop the +x bit.
    if [ -f "DataBuilder.app/Contents/MacOS/DataBuilder" ]; then
        chmod +x "DataBuilder.app/Contents/MacOS/DataBuilder" 2>/dev/null
        # Force Finder to refresh the bundle's icon cache so the custom
        # icns shows up immediately (otherwise Finder may keep showing
        # the generic Unix-executable icon until the next login).
        /usr/bin/touch "DataBuilder.app" 2>/dev/null
    fi

    echo "──────────────────────────────────────────────────────────────"
    echo " Done. Double-click  DataBuilder.app  to launch the app."
    echo " (or run.command if you prefer the terminal launcher)"
    echo "──────────────────────────────────────────────────────────────"
else
    echo "──────────────────────────────────────────────────────────────"
    echo " Install reported errors (exit $RC). Scroll up to see why."
    echo "──────────────────────────────────────────────────────────────"
fi

# Keep the Terminal window open so the user can read the output.
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
