#!/usr/bin/env bash
# Cloud-side setup. Installs dependencies, then `python train.py` is
# the only thing left to run.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> DataBuilder remote training: setup"
echo "    bundle root : $(pwd)"
echo "    python      : $(python --version 2>&1)"
echo "    nvidia-smi  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'no GPU detected')"

# venv (skip if already inside one)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo "==> Creating .venv"
        python -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

echo "==> Upgrading pip"
python -m pip install --upgrade pip --quiet

echo "==> Installing requirements (this may take 2-5 minutes the first time)"
pip install -r requirements.txt --quiet

# If model/ doesn't exist, the bundle expects to download from HF using
# the model_path stored in training_config.json. Warn the user if HF_TOKEN
# is missing — gated repos will 401 otherwise.
if [[ ! -d model ]]; then
    if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        echo
        echo "WARNING: no model/ directory in this bundle, AND no HF_TOKEN set."
        echo "         Gated HuggingFace repos (Flux dev, SD3.x) will fail to download."
        echo "         Set it before running train.py:"
        echo "             export HF_TOKEN=hf_xxxxx"
        echo
    fi
fi

echo "==> Setup complete. Run training with:"
echo "    python train.py"
