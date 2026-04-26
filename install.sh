#!/usr/bin/env bash
# ============================================================================
# DataBuilder — true one-click installer for macOS and Linux.
#
# Strategy:
#   1. If `uv` (Astral's Rust-based Python installer) isn't on PATH, we
#      bootstrap it with the official `curl | sh` one-liner. uv lives
#      in ~/.local/bin; no admin / sudo / Homebrew required.
#   2. uv downloads a known-good Python 3.12 build (python-build-standalone)
#      so we never depend on the system Python — no broken Homebrew
#      pyexpat, no Python 3.14 wheel issues, no PATH gymnastics.
#   3. uv creates `.venv/` and installs DataBuilder with the right extra
#      ([mac] on Darwin, [cuda] on Linux+NVIDIA, [all] otherwise).
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
printf "   DataBuilder — One-click installer\n"
printf "═══════════════════════════════════════════════════════════════${NC}\n\n"

# ── Anchor to script directory (when double-clicked from Finder/Files) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# ── Bootstrap uv ─────────────────────────────────────────────────────
# uv is a single static binary that can download Python interpreters
# (via python-build-standalone) without admin privileges. It's our
# escape hatch from the system-Python jungle.
ensure_uv_in_path() {
    # uv installs to ~/.local/bin by default; some older releases used
    # ~/.cargo/bin. Add both to PATH so `command -v uv` finds it
    # immediately, no shell restart required.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

ensure_uv_in_path
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv (Astral's Python installer) …"
    if ! command -v curl >/dev/null 2>&1; then
        die "curl not found — please install curl and re-run."
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ensure_uv_in_path
    if ! command -v uv >/dev/null 2>&1; then
        die "uv install completed but binary not on PATH. Restart your terminal and re-run."
    fi
    ok "uv installed: $(command -v uv)"
else
    ok "uv already installed: $(command -v uv)"
fi

# ── Install Python 3.12 via uv ───────────────────────────────────────
# This downloads a hermetic python-build-standalone build to uv's cache.
# It does NOT modify the system Python or require admin rights.
say "Ensuring Python 3.12 is available (via uv) …"
uv python install 3.12 -q
ok "Python 3.12 ready"

# ── Create or reuse the project venv ─────────────────────────────────
if [ -d .venv ]; then
    # Sanity-check that the existing venv was built with the right Python.
    # If it's the broken Homebrew Python we've seen before, it would still
    # appear to "work" until pip explodes — wipe and rebuild instead.
    if .venv/bin/python -c 'import ssl, ensurepip' >/dev/null 2>&1; then
        say "Reusing existing .venv/"
    else
        warn "Existing .venv/ is broken — recreating with uv-managed Python"
        rm -rf .venv
        uv venv .venv --python 3.12 -q
        ok ".venv/ rebuilt"
    fi
else
    say "Creating .venv/ with Python 3.12 …"
    uv venv .venv --python 3.12 -q
    ok ".venv/ created"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ── Pick the right install extra ─────────────────────────────────────
EXTRA="all"
if [ "$PLATFORM" = "macos" ]; then
    EXTRA="mac"
elif [ "$PLATFORM" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    EXTRA="cuda"
fi
say "Installing DataBuilder with extra: ${BOLD}.[${EXTRA}]${NC}"

# ── For Linux+NVIDIA, point pip at the CUDA wheel index for PyTorch ──
if [ "$EXTRA" = "cuda" ]; then
    say "Installing CUDA-enabled PyTorch (cu128 → cu126 fallback) …"
    if ! uv pip install -q torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128; then
        warn "CUDA 12.8 wheels unavailable — trying cu126"
        if ! uv pip install -q torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu126; then
            warn "CUDA wheels unavailable — falling back to CPU PyTorch"
            uv pip install -q torch torchvision torchaudio
        fi
    fi
fi

# ── Install DataBuilder + the platform extra ─────────────────────────
if ! uv pip install ".[${EXTRA}]"; then
    printf "\n${RED}Install failed.${NC}\n"
    printf "Try a smaller install to isolate the broken dep:\n"
    printf "  ${BOLD}source .venv/bin/activate && uv pip install \".[training,export]\"${NC}\n\n"
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

# ── Done ─────────────────────────────────────────────────────────────
printf "\n${BOLD}═══════════════════════════════════════════════════════════════${NC}\n"
printf "${GREEN}   Install complete.${NC}\n\n"
printf "   To launch DataBuilder, double-click ${BOLD}run.command${NC}\n"
printf "   (or in a terminal: ${BOLD}./run.sh${NC})\n"
printf "${BOLD}═══════════════════════════════════════════════════════════════${NC}\n\n"
