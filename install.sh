#!/usr/bin/env bash
# ============================================================================
# DataBuilder — One-shot installer for macOS and Linux.
#
# What this script does:
#   1. Locates a supported Python (3.10–3.13). Refuses 3.9 and below;
#      also refuses 3.14+ because most ML wheels don't ship for it yet.
#   2. Verifies the chosen Python can actually create a working venv
#      (Homebrew's python@3.12 has been known to ship a broken pyexpat).
#   3. Creates `.venv/`, upgrades pip, and installs DataBuilder with the
#      right platform extra (`.[mac]` on Darwin, `.[cuda]` on Linux+NVIDIA,
#      `.[all]` otherwise — the pyproject markers make `[all]` safe now).
#   4. Prints a one-liner to launch the app.
#
# Re-runnable: detects an existing venv and skips creation.
# ============================================================================

set -e

# ── Colours ──────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

say()  { printf "${CYAN}▸${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${NC} %s\n" "$*"; }
die()  { printf "${RED}✗${NC} %s\n" "$*" >&2; exit 1; }

printf "${BOLD}\n"
printf "═══════════════════════════════════════════════════════════════\n"
printf "   DataBuilder — Installer\n"
printf "═══════════════════════════════════════════════════════════════${NC}\n\n"

# ── Detect OS ────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Darwin)  PLATFORM="macos" ;;
    Linux)   PLATFORM="linux" ;;
    *)       die "Unsupported OS: $OS (use install.bat on Windows)" ;;
esac
say "Platform: $PLATFORM"

# ── Pull latest if this is a checked-out git repo ────────────────────
if [ -d .git ]; then
    git pull --ff-only 2>/dev/null \
        && ok "Repo updated to latest revision" \
        || warn "Could not git-pull (offline or detached HEAD) — using local source"
fi

# ── Find a supported Python (3.10 – 3.13) ────────────────────────────
say "Looking for Python 3.10 – 3.13 …"
PYTHON_CMD=""
PYTHON_VER=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ver=$("$cmd" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
            2>/dev/null) || continue
        case "$ver" in
            3.10|3.11|3.12|3.13)
                PYTHON_CMD="$cmd"
                PYTHON_VER="$ver"
                ok "Found $cmd → Python $ver"
                break
                ;;
        esac
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    printf "\n${RED}No supported Python (3.10 – 3.13) found.${NC}\n\n"
    if [ "$PLATFORM" = "macos" ]; then
        printf "Install one with either:\n"
        printf "  ${BOLD}brew install python@3.12${NC}\n"
        printf "or download the official installer:\n"
        printf "  ${BOLD}https://www.python.org/downloads/release/python-31210/${NC}\n"
    else
        printf "Install with:\n"
        printf "  ${BOLD}sudo apt install python3.12 python3.12-venv${NC}\n"
        printf "  (or your distro's equivalent)\n"
    fi
    printf "\nThen re-run this script.\n"
    exit 1
fi

# ── Sanity-check that the chosen Python can build a working venv ────
# Homebrew has shipped broken python@3.12 builds where pyexpat is linked
# against a libexpat the OS doesn't have. Catch that early — the error
# is opaque if pip discovers it later.
say "Verifying $PYTHON_CMD can create a venv …"
VENV_TEST="$(mktemp -d)/databuilder_venv_check"
if ! "$PYTHON_CMD" -m venv "$VENV_TEST" 2>/tmp/databuilder_venv_check.log; then
    printf "\n${RED}$PYTHON_CMD cannot create a virtual environment.${NC}\n"
    printf "Error log:\n"
    sed 's/^/  /' /tmp/databuilder_venv_check.log
    printf "\n"
    if [ "$PLATFORM" = "macos" ]; then
        printf "${YELLOW}Common cause on macOS:${NC} Homebrew shipped a Python whose\n"
        printf "pyexpat is linked against a libexpat your system doesn't have.\n\n"
        printf "Fix — install the official build from python.org instead:\n"
        printf "  ${BOLD}https://www.python.org/downloads/release/python-31210/${NC}\n"
        printf "  (download the macOS 64-bit universal2 .pkg, run it, then re-run\n"
        printf "  this script — it will pick up the new python3.12.)\n\n"
    fi
    rm -rf "$VENV_TEST" /tmp/databuilder_venv_check.log
    exit 1
fi
rm -rf "$VENV_TEST" /tmp/databuilder_venv_check.log
ok "venv creation works"

# ── Create or reuse the project venv ─────────────────────────────────
if [ -d .venv ]; then
    say ".venv/ already exists — reusing it"
else
    say "Creating .venv/ with Python $PYTHON_VER …"
    "$PYTHON_CMD" -m venv .venv
    ok ".venv/ created"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ── Upgrade pip / wheel / setuptools ─────────────────────────────────
say "Upgrading pip, wheel, setuptools …"
pip install --upgrade pip wheel setuptools -q
ok "Build tooling up to date"

# ── Pick the right install extra ─────────────────────────────────────
EXTRA="all"
if [ "$PLATFORM" = "macos" ]; then
    # On Mac, skip CUDA / Linux-only deps entirely — `[mac]` is the
    # curated set that won't try to fetch wheels that don't exist.
    EXTRA="mac"
elif [ "$PLATFORM" = "linux" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        EXTRA="cuda"
    fi
fi
say "Installing DataBuilder with extra: ${BOLD}.[${EXTRA}]${NC}"

# ── Install PyTorch with the right backend ───────────────────────────
# We let the [training] extra do the heavy lifting except on Linux+CUDA
# where we need to point pip at the matching CUDA wheel index.
if [ "$EXTRA" = "cuda" ]; then
    say "Installing CUDA-enabled PyTorch (cu128 → cu126 fallback) …"
    if ! pip install -q torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128; then
        warn "CUDA 12.8 wheel unavailable — falling back to cu126"
        if ! pip install -q torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu126; then
            warn "CUDA wheels unavailable — falling back to CPU PyTorch"
            pip install -q torch torchvision torchaudio
        fi
    fi
fi

# ── Install DataBuilder + the extra ──────────────────────────────────
if ! pip install ".[${EXTRA}]"; then
    printf "\n${RED}pip install failed.${NC}\n"
    printf "Try a smaller install to isolate the broken dep:\n"
    printf "  ${BOLD}pip install \".[training,export]\"${NC}\n\n"
    exit 1
fi

# ── Final sanity check ───────────────────────────────────────────────
say "Verifying the install …"
python - <<'PY'
import sys
def line(name, val):
    print(f"  {name:<14} {val}")
line("Python:", sys.version.split()[0])
try:
    import PyQt6.QtCore as q
    line("PyQt6:", q.PYQT_VERSION_STR)
except Exception as e:
    line("PyQt6:", f"missing ({e})")
try:
    import torch
    line("PyTorch:", torch.__version__)
    if torch.cuda.is_available():
        line("CUDA:", torch.version.cuda)
        line("GPU:", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        line("Backend:", "Apple Metal (MPS)")
    else:
        line("Backend:", "CPU only")
except Exception as e:
    line("PyTorch:", f"missing ({e})")
for mod in ("diffusers", "transformers", "peft"):
    try:
        m = __import__(mod)
        line(f"{mod}:", m.__version__)
    except Exception:
        line(f"{mod}:", "not installed (training won't work)")
PY

# ── Done — print the launch instructions ─────────────────────────────
printf "\n${BOLD}═══════════════════════════════════════════════════════════════${NC}\n"
printf "${GREEN}   Install complete.${NC}\n\n"
printf "   To launch DataBuilder:\n"
printf "     ${BOLD}source .venv/bin/activate${NC}\n"
printf "     ${BOLD}python -m dataset_sorter${NC}\n\n"
printf "${BOLD}═══════════════════════════════════════════════════════════════${NC}\n\n"
