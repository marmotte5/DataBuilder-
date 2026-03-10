#!/usr/bin/env bash
# ============================================================================
# DataBuilder - One-Click Installer (macOS / Linux)
#
# Installs Python environment, PyTorch (Metal MPS / CUDA / CPU), and all
# dependencies for the Dataset Sorter + Trainer application.
#
# macOS: Uses Apple Metal (MPS) GPU acceleration automatically
# Linux: Uses CUDA 12.8/12.6 if available, falls back to CPU
#
# Requires: Python 3.10+ installed
# ============================================================================

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${BOLD}=============================================================${NC}"
echo -e "${BOLD}   DataBuilder - Installer${NC}"
echo -e "${BOLD}   Dataset Sorter + SDXL / Z-Image Trainer${NC}"
echo -e "${BOLD}=============================================================${NC}"
echo ""

# ── Detect OS ────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Darwin)  PLATFORM="macos" ;;
    Linux)   PLATFORM="linux" ;;
    *)       echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1 ;;
esac
echo -e "${GREEN}[info]${NC} Platform: $PLATFORM ($OS)"

# ── Check Python ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[1/7] Checking Python...${NC}"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}ERROR: Python not found.${NC}"
    if [ "$PLATFORM" = "macos" ]; then
        echo "  Install with: brew install python@3.11"
        echo "  Or download from: https://www.python.org/downloads/"
    else
        echo "  Install with: sudo apt install python3 python3-venv python3-pip"
    fi
    exit 1
fi

PYVER=$($PYTHON_CMD --version 2>&1)
echo "       Found $PYVER"

# ── Create virtual environment ───────────────────────────────────────
echo ""
echo -e "${BOLD}[2/7] Creating virtual environment...${NC}"

if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "       Created venv/"
else
    echo "       venv/ already exists, reusing."
fi

# Activate venv
source venv/bin/activate

# ── Upgrade pip ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/7] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel -q
echo "       pip upgraded."

# ── Install PyTorch ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[4/7] Installing PyTorch...${NC}"

if [ "$PLATFORM" = "macos" ]; then
    echo "       Installing PyTorch with Metal (MPS) support..."
    pip install torch torchvision torchaudio
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to install PyTorch.${NC}"
        exit 1
    fi
else
    echo "       Trying CUDA 12.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}       CUDA 12.8 failed. Trying CUDA 12.6...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 2>/dev/null
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}       CUDA failed. Installing CPU-only PyTorch...${NC}"
            pip install torch torchvision torchaudio
        fi
    fi
fi

# ── Install core dependencies ────────────────────────────────────────
echo ""
echo -e "${BOLD}[5/7] Installing core dependencies...${NC}"
pip install "PyQt6>=6.5" "numpy>=1.24" "Pillow>=10.0"
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install core dependencies.${NC}"
    exit 1
fi

# ── Install training dependencies ────────────────────────────────────
echo ""
echo -e "${BOLD}[6/7] Installing training dependencies...${NC}"
pip install "diffusers>=0.28" "transformers>=4.38" "accelerate>=0.27" "safetensors>=0.4" "peft>=0.10"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}WARNING: Some training dependencies failed. Training may not work.${NC}"
fi

# Optional optimizers
echo ""
echo -e "${BOLD}[6b/7] Installing optional optimizers...${NC}"

for pkg in bitsandbytes prodigyopt lion-pytorch dadaptation; do
    pip install "$pkg" -q 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "       $pkg - OK"
    else
        echo "       $pkg - skipped"
    fi
done

# ── Verify installation ─────────────────────────────────────────────
echo ""
echo -e "${BOLD}[7/7] Verifying installation...${NC}"
echo ""

python3 -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA:         {torch.version.cuda}')
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1)} GB')
    print(f'  bf16:         {torch.cuda.is_bf16_supported()}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  GPU:          Apple Silicon (Metal/MPS)')
    print(f'  MPS backend:  Available')
else:
    print(f'  GPU:          None (CPU only)')
"

python3 -c "import diffusers; print(f'  Diffusers:    {diffusers.__version__}')" 2>/dev/null
python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null
python3 -c "import peft; print(f'  PEFT:         {peft.__version__}')" 2>/dev/null
python3 -c "import PyQt6.QtCore; print(f'  PyQt6:        {PyQt6.QtCore.PYQT_VERSION_STR}')" 2>/dev/null

echo ""
echo -e "${BOLD}=============================================================${NC}"
echo -e "${GREEN}   Installation complete!${NC}"
echo ""
echo "   To run DataBuilder:"
if [ "$PLATFORM" = "macos" ]; then
    echo "     1. Double-click  run.command"
    echo "     2. Or run:  ./run.sh"
    echo "     3. Or run:  source venv/bin/activate && python dataset_sorter.py"
else
    echo "     1. Run:  ./run.sh"
    echo "     2. Or run:  source venv/bin/activate && python dataset_sorter.py"
fi
echo -e "${BOLD}=============================================================${NC}"
echo ""
