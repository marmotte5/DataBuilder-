```
██████╗  █████╗ ████████╗ █████╗ ██████╗ ██╗   ██╗██╗██╗     ██████╗ ███████╗██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██║   ██║██║██║     ██╔══██╗██╔════╝██╔══██╗
██║  ██║███████║   ██║   ███████║██████╔╝██║   ██║██║██║     ██║  ██║█████╗  ██████╔╝
██║  ██║██╔══██║   ██║   ██╔══██║██╔══██╗██║   ██║██║██║     ██║  ██║██╔══╝  ██╔══██╗
██████╔╝██║  ██║   ██║   ██║  ██║██████╔╝╚██████╔╝██║███████╗██████╔╝███████╗██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

<div align="center">

**An all-in-one desktop trainer and generator for text-to-image AI models.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41cd52?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-orange?logo=pytest&logoColor=white)](tests/)
[![Models](https://img.shields.io/badge/Models-16%20architectures-purple)](docs/QUICKSTART.md)

[Quick Start](#quick-start) · [Supported Models](#supported-architectures) · [Documentation](docs/QUICKSTART.md) · [Contributing](CONTRIBUTING.md)

</div>

---

DataBuilder is a desktop application covering the **full text-to-image training pipeline**: dataset preparation, tag management, LoRA and full fine-tune training, image generation, and model/LoRA library management. Built with PyQt6, targeting RTX 4090 / H100 class hardware, with backends for 16 model architectures including Flux, SD3.5, and HiDream.

> **Why DataBuilder?** Most trainers force you to juggle separate tools for dataset prep, training, and generation. DataBuilder bundles them into one native desktop UI. Speed is a primary goal — we ship with `torch.compile` integration, custom Triton kernels, optional FP8 training (Ada/Hopper), memory-mapped datasets, and the Marmotte optimizer (10-20× less optimizer memory than AdamW). Whether DataBuilder is *the* fastest at any given task depends on the model and hardware; the project is honest about being a work in progress on that front.

---

## Screenshots

> *Screenshots coming soon — contributions welcome!*

| Dataset Manager | Training Tab | Generation |
|:-:|:-:|:-:|
| *(placeholder)* | *(placeholder)* | *(placeholder)* |

---

## Features

### Dataset Management
- **Tag-based sorting** — 80 rarity buckets using numpy percentile quantiles; rare concepts get up to 20× more repeats automatically
- **Visual image browser** — Navigate images with per-image bucket control and LRU pixmap cache
- **Caption tools** — Tag shuffle/dropout simulation, token count analysis, spell check, semantic grouping
- **Tag histogram** — Frequency distribution with top/rare/distribution views
- **Duplicate detection** — Exact (MD5) and near-duplicate (perceptual hash) detection
- **Bulk operations** — Rename, merge, search & replace tags; full undo/redo (Ctrl+Z)
- **One-click pipeline** — Automated clean + train workflow for beginners

### Training
- **16 model architectures** — See [full table below](#supported-architectures)
- **14 optimizers** — Including the custom Marmotte optimizer (10-20× less optimizer state memory than AdamW)
- **7 training presets** — Character LoRA, Style LoRA, Concept LoRA, Photorealistic, Quick Test, DPO, ControlNet
- **ControlNet training** — 9 conditioning types (canny, depth, pose, segmentation, normal, MLSD, softedge, scribble, inpaint)
- **DPO training** — Direct Preference Optimization with sigmoid, hinge, and IPO loss
- **Real-time monitoring** — Live loss curves, LR schedule preview, VRAM usage bar, sample preview
- **Mid-training controls** — Save Now, Sample Now, Backup Project, Pause/Resume, Stop
- **Smart resume** — Analyzes loss curves and auto-adjusts LR/optimizer on resume
- **SpeeD (CVPR 2025)** — Asymmetric timestep sampling + change-aware loss weighting

### Speed optimizations (The Marmotte Philosophy)
Speed is a primary goal after correctness. The optimizations below are integrated and configurable from the UI; the *speedup* and *memory* columns are the published or commonly-cited ranges for the underlying technique, not validated DataBuilder benchmarks. Real-world numbers depend heavily on model, GPU, and config — please benchmark on your own hardware before relying on a specific number.

| Optimization | Reported range | Notes |
|---|---|---|
| `torch.compile` | 20-40% | Full graph capture; benefits vary by backend |
| Triton fused kernels | 5-15% | AdamW, MSE loss, flow interpolation |
| FP8 training | up to ~2× FP16 TFLOPS | Ada (RTX 40xx) / Hopper only; quality impact under evaluation |
| `channels_last` layout | 10-20% | Reduced memory bandwidth on conv-based UNets |
| Sequence packing | 10-30% | Removes padding waste for DiT models |
| Zero-bottleneck DataLoader | Bypasses GIL | mmap + pinned DMA |
| Memory-mapped datasets | Zero-copy I/O | `mmap_dataset.py` |
| Marmotte optimizer | 10-20× less optimizer memory | vs AdamW; convergence still being characterised on full FT runs |

### UI & UX
- **Dark/light theme** — Ctrl+T, persisted across sessions
- **Drag & drop** — Drop folders onto any input
- **Keyboard shortcuts** — Ctrl+S, Ctrl+O, Ctrl+E, Ctrl+R, Ctrl+T, Ctrl+D, Ctrl+Z
- **Tooltips** — Detailed explanations on every parameter
- **Toast notifications** — Non-blocking status messages
- **Virtual tag table** — O(1) rendering, handles 1M+ tags

### Generation & Library
- **Integrated generator** — Text-to-image and img2img with LoRA support
- **Batch generation** — Prompt queue, CSV/JSON/TXT import, auto-save
- **A/B comparison** — Side-by-side generation with per-side overrides
- **Model merging** — Weighted sum, SLERP, add-difference
- **LoRA library** — Favorites, notes, tags, ratings

---

## Supported Architectures

| Model | Backend | Prediction | Text Encoder(s) |
|---|---|---|---|
| SD 1.5 | `train_backend_sd15.py` | epsilon | CLIP |
| SD 2.x | `train_backend_sd2.py` | v-prediction | OpenCLIP |
| SDXL / Pony | `train_backend_sdxl.py` | epsilon | CLIP-L + CLIP-G |
| SD 3 | `train_backend_sd3.py` | flow | CLIP-L + CLIP-G + T5-XXL |
| SD 3.5 | `train_backend_sd35.py` | flow | CLIP-L + CLIP-G + T5-XXL |
| Flux | `train_backend_flux.py` | flow | CLIP-L + T5-XXL |
| Flux 2 | `train_backend_flux2.py` | flow | LLM (Mistral-3 / Qwen-3) |
| Z-Image | `train_backend_zimage.py` | flow | Qwen3 (chat template) |
| PixArt | `train_backend_pixart.py` | flow | T5-XXL |
| Kolors | `train_backend_kolors.py` | epsilon | ChatGLM-6B |
| Stable Cascade | `train_backend_cascade.py` | epsilon | CLIP-G |
| Chroma | `train_backend_chroma.py` | flow | T5-XXL |
| AuraFlow | `train_backend_auraflow.py` | flow | T5 |
| Sana | `train_backend_sana.py` | flow | Gemma-2B |
| HunyuanDiT | `train_backend_hunyuan.py` | epsilon | CLIP-L + mT5 |
| HiDream | `train_backend_hidream.py` | flow | CLIP-L + CLIP-G + T5-XXL + Llama |

Distilled variants (Lightning, LCM, Hyper-SD, Turbo, Flux Schnell, DMD/DMD2) are auto-detected by filename and routed to CFG=1.0 with low step counts.

All backends support both **diffusers directories** and **single-file `.safetensors` / `.ckpt` checkpoints**.

---

## Quick Start

### Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3080 (10 GB VRAM) | RTX 4090 / H100 |
| RAM | 16 GB | 32 GB+ |
| Python | 3.10 | 3.11+ |
| OS | Windows 10, Linux, macOS | Ubuntu 22.04 |
| Storage | 20 GB free | NVMe SSD |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/marmotte5/DataBuilder-.git
cd DataBuilder-

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install PyTorch first (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install DataBuilder

# Basic install (dataset sorting & UI only)
pip install .

# With training support
pip install ".[training]"

# Full install (all optimizers + speed extras)
pip install ".[all]"

# 5. Launch
python -m dataset_sorter
```

#### Windows one-click installer

```bat
install.bat   # Sets up venv + installs all dependencies
run.bat       # Launches the app
```

#### Linux system dependencies

On headless Linux or minimal installations:

```bash
sudo apt-get install libegl1 libgl1
```

#### macOS / Apple Silicon

> Tested on M1, M1 Pro/Max, M2, M3 series. Training runs on the MPS backend.

**Hardware requirements**

| Tier | Hardware | What you can train |
|---|---|---|
| Minimum | M1 8 GB | LoRA SD 1.5 @ 256 px, batch=1 |
| Recommended | M1 Pro/Max 16 GB+ | LoRA SDXL @ 512 px |
| Ideal | M2 Ultra 64 GB+ | Full fine-tune |

**Prerequisites**

```bash
# 1. Install Xcode Command Line Tools (required for compilers)
xcode-select --install

# 2. Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Install Python 3.10+ via Homebrew (do NOT use the macOS system Python)
brew install python@3.10
```

**Installation**

```bash
# Clone and enter the repo
git clone https://github.com/marmotte5/DataBuilder-.git
cd DataBuilder-

# Create a virtual environment with the Homebrew Python
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch — MPS support is built-in, no special index URL needed
pip install torch torchvision

# Install DataBuilder with training support
# Use [training] instead of [all] on macOS — [all] pulls in flash-attn,
# triton, and transformer-engine which are CUDA-only and will fail to build.
pip install -e ".[training]"

# Launch
.venv/bin/dataset-sorter
# or: python -m dataset_sorter
```

**Known MPS limitations**

- FP16 training is not stable on MPS — DataBuilder automatically falls back to BF16/FP32.
- `flash-attn`, `triton`, and `transformer-engine` are CUDA-only; skip the `[speed]` and `[all]` extras.
- `bitsandbytes` has limited MPS support; 8-bit/4-bit quantised optimizers will not be available.
- `num_workers > 0` in the DataLoader can cause hangs on macOS — DataBuilder sets `num_workers=0` automatically when MPS is detected.
- `torch.compile` requires `triton` (not available on MPS) and is automatically disabled.
- FP32 training can trigger MPS OOM on the first forward pass — DataBuilder automatically sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` at startup to disable the memory ceiling and let the Metal allocator use all available unified memory. You can override this by setting the env var yourself before launching.

---

## Your First LoRA in 5 Steps

For a detailed step-by-step tutorial, see **[docs/QUICKSTART.md](docs/QUICKSTART.md)**.

**TL;DR:**

1. **Scan** your image dataset (drag & drop the folder)
2. **Review tags** — delete noise, merge duplicates, adjust buckets
3. **Export project** (Ctrl+E) — creates the structured training folder
4. **Pick a preset** — "Character LoRA" or "Style LoRA" are great starting points
5. **Start Training** — monitor live loss, save samples at any step

---

## Project Folder Structure

```
my_project/
├── project.json                   # Project metadata
├── dataset/                       # Exported & organized dataset
│   ├── 1_1_common/                # 1× repeat (most common tags)
│   ├── 5_10_medium/               # 5× repeat
│   └── 20_40_rare_concept/        # 20× repeat (rarest tags)
├── models/                        # Final trained weights
├── samples/                       # Sample images generated during training
├── checkpoints/                   # Step/epoch checkpoints
├── backups/                       # Timestamped full project backups
├── logs/                          # Training logs + TensorBoard events
└── .cache/                        # Latent & text encoder caches
```

---

## Architecture Overview

```
dataset_sorter/
├── __main__.py              # Entry point
├── models.py                # TrainingConfig dataclass, ImageEntry, DatasetStats
├── constants.py             # Model types, optimizers, presets
├── trainer.py               # Core training engine
├── training_worker.py       # QThread training worker
├── generate_worker.py       # QThread generation worker
├── backend_registry.py      # Auto-discovery of model backends
├── train_backend_base.py    # Base class for backends
├── train_backend_*.py       # 17 model backends
├── optimizers.py            # Marmotte, SOAP, Muon optimizers
├── triton_kernels.py        # Triton fused kernels
├── fp8_training.py          # FP8 training (2× TFLOPS)
├── sequence_packing.py      # Zero padding waste
├── mmap_dataset.py          # Memory-mapped datasets
├── zero_bottleneck_dataloader.py  # GIL-free dataloader
└── ui/
    ├── main_window.py       # Main window orchestrator
    ├── training_tab.py      # Training UI (180+ parameters, 7 tabs)
    ├── generate_tab.py      # Generation tab
    ├── library_tab.py       # LoRA & model library
    ├── batch_generation_tab.py  # Batch generation
    ├── comparison_tab.py    # A/B comparison
    └── model_merge_tab.py   # Model merging
```

---

## Contributing

We welcome contributions! Whether you want to add a new model backend, improve an optimizer, fix a bug, or improve documentation — you're in the right place.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines on:
- Setting up your dev environment
- Adding a new model backend (it's designed to be easy!)
- Code style and conventions
- Submitting pull requests

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Free to use, modify, and distribute. Commercial use allowed. No warranty.

---

<div align="center">

Built with love by the DataBuilder contributors.
If DataBuilder saves you time, consider starring the repo ⭐

</div>
