# Dataset Sorter

A desktop application for sorting image datasets by tag rarity — designed for Stable Diffusion / LoRA training dataset preparation. Features a modern dark/light UI built with PyQt6, integrated training with 60+ parameters, and comprehensive dataset management tools.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### Core
- **Tag-based sorting** — Automatically sorts images into 80 buckets based on tag rarity using numpy percentile quantiles
- **Security-first design** — READ-ONLY on source files; all writes go to the output directory only
- **Config persistence** — Save/load overrides, deleted tags, and bucket names as JSON
- **Progress persistence** — Remembers source/output dirs, theme, and settings across sessions
- **Undo/redo** — Full undo/redo stack (Ctrl+Z / Ctrl+Shift+Z) for all tag operations

### Training
- **Integrated training** — Full training UI with 60+ parameters across 7 config tabs (Model, Optimizer, Dataset, Advanced, Sampling, ControlNet, DPO)
- **17 model types** — SD 1.5, SD 2, SDXL, Pony, Flux, Flux 2, SD3, SD 3.5, Z-Image, PixArt, Stable Cascade, Hunyuan DiT, Kolors, AuraFlow, Sana, HiDream, Chroma
- **11 optimizers** — Adafactor, Prodigy, AdamW, AdamW 8-bit, D-Adapt Adam, CAME, AdamW Schedule-Free, Lion, SGD, SOAP, Muon
- **7 training presets** — Character LoRA, Style LoRA, Concept LoRA, Photorealistic, Quick Test, DPO Preference, ControlNet
- **ControlNet training** — 9 conditioning types (canny, depth, pose, segmentation, normal, MLSD, softedge, scribble, inpaint)
- **DPO training** — Direct Preference Optimization with 3 loss types (sigmoid, hinge, IPO)
- **Adversarial fine-tuning** — Discriminator with feature matching loss
- **Custom LR schedulers** — Piecewise composite schedules (warmup+cosine+cooldown, cyclic cosine, step decay)
- **Loss curve visualization** — Real-time QPainter line chart of training loss
- **LR schedule preview** — ASCII graph preview of learning rate schedule before training
- **VRAM pre-estimation** — Estimate GPU memory usage before starting training
- **Training config save/load** — Export/import full training configuration as JSON
- **Real-time monitoring** — Progress bar, step/loss/LR display, VRAM usage bar, sample preview
- **Mid-training controls** — Save Now, Sample Now, Backup Project, Pause/Resume, Stop
- **Activation checkpointing** — Granular control (none/full/selective/every_n)
- **SpeeD (CVPR 2025)** — Asymmetric timestep sampling + change-aware loss weighting
- **Advanced memory** — MeBP, approximate VJP, async GPU prefetch, fused backward pass, fp8 base model

### Dataset Management
- **Caption preview** — Tag shuffle/dropout simulation with configurable variants
- **Token count analysis** — Per-image token counts with over-limit detection (SD/Flux/Kolors limits)
- **Tag histogram** — Frequency distribution visualization with top/rare/distribution views
- **Spell check** — Typo detection, near-duplicate tags, semantic grouping suggestions
- **Data augmentation** — Configurable augmentation pipeline with enable/disable toggles
- **Duplicate detection** — Exact (MD5) and near-duplicate (perceptual hash) image detection
- **Disk space checks** — Pre-export and pre-training disk space validation

### UI & UX
- **Dark/light theme** — Toggle with Ctrl+T, persisted across sessions
- **Drag & drop** — Drop folders onto source/output inputs or the main window
- **Keyboard shortcuts** — Ctrl+S save, Ctrl+O load, Ctrl+E export, Ctrl+R scan, Ctrl+T theme, Ctrl+D dry run, Ctrl+Z undo, Ctrl+Shift+Z redo
- **Tooltips** — Detailed tooltips on every parameter explaining purpose and recommended values
- **Image browser** — Navigate images with per-image bucket control, jump-to navigation, and LRU pixmap caching
- **Virtual tag table** — O(1) rendering with QAbstractTableModel (handles 1M+ tags)

### Infrastructure
- **Training recommendations** — State-of-the-art parameter suggestions based on model, VRAM, dataset size
- **VRAM monitoring** — Real-time GPU memory tracking during training
- **Export formats** — OneTrainer TOML and kohya_ss JSON config export
- **Parallel scanning** — Configurable worker threads for fast dataset loading
- **Optional GPU validation** — Detect corrupt images early using torchvision
- **pyproject.toml** — Proper Python packaging with optional dependency groups

## Installation

### Requirements

- Python 3.10+
- PyQt6
- NumPy

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DataBuilder-.git
cd DataBuilder-

# Install (basic — sorting & UI only)
pip install .

# Install with training support
pip install ".[training]"

# Install everything (all optimizers)
pip install ".[all]"

# Run the application
dataset-sorter
# or
python -m dataset_sorter
```

### Alternative: Run from source

```bash
pip install -r requirements.txt
python -m dataset_sorter
```

### System dependencies (Linux)

On headless Linux or minimal installations, you may need:

```bash
sudo apt-get install libegl1 libgl1
```

## Usage

### 1. Scan a dataset

1. Click **Browse** next to "Source" and select your image dataset folder (or drag & drop)
2. Click **Browse** next to "Output" and select where sorted images will be exported
3. Adjust **Workers** count (higher = faster on SSDs)
4. Optionally enable **GPU validation** if you have CUDA
5. Click **Scan** (Ctrl+R)

### 2. Review and edit tags

- **Filter** tags using the search box in the left panel
- **Select** one or more tags to see image previews
- **Override** bucket assignments for specific tags
- **Delete** tags you don't want in the exported captions
- **Rename**, **merge**, or **search & replace** tags in bulk
- **Undo/redo** any tag operation with Ctrl+Z / Ctrl+Shift+Z

### 3. Configure training

Switch to the **Training** tab:
- Select a **preset** (Character LoRA, Style LoRA, etc.) or configure manually
- Click **Estimate VRAM** to check if your settings fit your GPU
- Click **Preview LR** to see the learning rate schedule
- **Save/Load Config** to export/import settings as JSON

### 4. Export

- **Dry Run** — Preview which images go to which buckets
- **Export** — Copy images and filtered tag files to the output directory

## Project Structure

```
dataset_sorter/
├── __init__.py              # Package marker
├── __main__.py              # Entry point
├── constants.py             # Global constants (17 model types, optimizers, etc.)
├── models.py                # Data classes (TrainingConfig, ImageEntry, DatasetStats)
├── utils.py                 # Path validation, sanitization, GPU detection
├── recommender.py           # Training parameter recommendation engine
├── workers.py               # QThread workers for scan and export
├── training_worker.py       # QThread worker for model training
├── training_presets.py      # 7 presets, ControlNet/DPO config, custom schedulers
├── lr_preview.py            # LR schedule preview (pure math, no torch)
├── vram_estimator.py        # VRAM estimation for 17 model types
├── duplicate_detector.py    # Duplicate/near-duplicate image detection
├── disk_space.py            # Disk space validation
├── dataset_management.py    # Caption augmentation, token counts, spell check
└── ui/
    ├── theme.py             # Dark/light theme with runtime toggling
    ├── tag_panel.py         # Virtual tag table (O(1) rendering)
    ├── override_panel.py    # Tools, stats, config panel
    ├── preview_tab.py       # Image thumbnail preview grid
    ├── reco_tab.py          # Training recommendations tab
    ├── image_tab.py         # Image browser with pixmap caching
    ├── training_tab.py      # Full training UI with 7 config sub-tabs
    ├── dataset_tab.py       # Dataset management (6 sub-tabs)
    ├── generate_tab.py      # Image generation tab
    ├── dialogs.py           # Export summary dialog
    └── main_window.py       # Main window orchestrator
```

## License

MIT License — see [LICENSE](LICENSE) for details.
