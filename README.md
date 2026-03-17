# Dataset Sorter

A desktop application for sorting image datasets by tag rarity — designed for Stable Diffusion / LoRA training dataset preparation. Features a modern dark/light UI built with PyQt6, integrated training with 180+ parameters, automatic repeat balancing, and a unified project folder system.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### Core
- **Tag-based sorting** — Automatically sorts images into 80 buckets based on tag rarity using numpy percentile quantiles
- **Automatic repeats** — Rare buckets get more repeats (1x-20x) so the model trains on them equally; folder names follow Kohya-style `{repeats}_{bucket}_{name}` format
- **Project folder system** — Export creates a full project directory (dataset, models, samples, checkpoints, backups, logs) shared between export and training
- **Security-first design** — READ-ONLY on source files; all writes go to the output directory only
- **Config persistence** — Save/load overrides, deleted tags, and bucket names as JSON
- **Progress persistence** — Remembers source/output dirs, theme, and settings across sessions
- **Undo/redo** — Full undo/redo stack (Ctrl+Z / Ctrl+Shift+Z) for all tag operations
- **Sequential file renaming** — Exported files are renamed `{bucket}_{index}.{ext}` for clean organization

### Training
- **Integrated training** — Full training UI with 180+ parameters across 7 config tabs (Model, Optimizer, Dataset, Advanced, Sampling, ControlNet, DPO)
- **17 model types** — SD 1.5, SD 2, SDXL, Pony, Flux, Flux 2, SD3, SD 3.5, Z-Image, PixArt, Stable Cascade, Hunyuan DiT, Kolors, AuraFlow, Sana, HiDream, Chroma
- **12 optimizers** — Marmotte, Adafactor, Prodigy, AdamW, AdamW 8-bit, D-Adapt Adam, CAME, AdamW Schedule-Free, Lion, SGD, SOAP, Muon
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
- **Smart resume** — Analyzes loss curves and auto-adjusts LR/optimizer on resume
- **Extreme speed** — Triton fused kernels, FP8 training, sequence packing, memory-mapped datasets, CUDA graph training
- **Zero-bottleneck dataloader** — mmap + pinned DMA replaces PyTorch DataLoader (bypasses GIL/pickle)
- **Z-Image optimizations** — Unified stream attention, fused 3D RoPE, fat latent cache, logit-normal timesteps, velocity weighting
- **Z-Image inventions** — L2-pinned attention, speculative gradient, stream-bending bias, timestep bandit
- **Pipeline integration** — Pre-training validation, auto-config, live loss monitoring with auto-adjustment
- **Marmotte optimizer** — Ultra-low memory per-channel adaptive optimizer (~10-20x less memory than Adam)
- **Masked training** — Region-specific loss computation using binary masks for inpainting/subject-focused training
- **TensorBoard logging** — Real-time loss, LR, and VRAM monitoring with graceful fallback when not installed
- **Noise schedule rescaling** — Zero terminal SNR fix (Lin et al. 2024) for improved dark/light image quality
- **Textual inversion** — Embedding training support with configurable token count and initialization

### Dataset Management
- **Caption preview** — Tag shuffle/dropout simulation with configurable variants
- **Token count analysis** — Per-image token counts with over-limit detection (SD/Flux/Kolors limits)
- **Tag histogram** — Frequency distribution visualization with top/rare/distribution views
- **Spell check** — Typo detection, near-duplicate tags, semantic grouping suggestions
- **Tag importance analysis** — Identify noise tags vs. meaningful tags for cleaning
- **Data augmentation** — Configurable augmentation pipeline with enable/disable toggles
- **Duplicate detection** — Exact (MD5) and near-duplicate (perceptual hash) image detection
- **Disk space checks** — Pre-export and pre-training disk space validation
- **One-click pipeline** — Automated clean + train workflow for beginners

### UI & UX
- **Dark/light theme** — Toggle with Ctrl+T, persisted across sessions
- **Drag & drop** — Drop folders onto source/output inputs or the main window
- **Keyboard shortcuts** — Ctrl+S save, Ctrl+O load, Ctrl+E export, Ctrl+R scan, Ctrl+T theme, Ctrl+D dry run, Ctrl+Z undo, Ctrl+Shift+Z redo
- **Tooltips** — Detailed tooltips on every parameter explaining purpose and recommended values
- **Image browser** — Navigate images with per-image bucket control, jump-to navigation, and LRU pixmap caching
- **Virtual tag table** — O(1) rendering with QAbstractTableModel (handles 1M+ tags)
- **Toast notifications** — Non-blocking status messages for operations

### Infrastructure
- **Training recommendations** — State-of-the-art parameter suggestions based on model, VRAM, dataset size
- **VRAM monitoring** — Real-time GPU memory tracking during training
- **Training history** — Learns from past runs to improve recommendations
- **Export formats** — OneTrainer TOML and kohya_ss JSON config export
- **Parallel scanning** — Configurable worker threads for fast dataset loading
- **Parallel export** — ThreadPoolExecutor-based file copying (3-5x speedup on SSDs)
- **Optional GPU validation** — Detect corrupt images early using torchvision
- **Config validation** — Validates training config before launch
- **Backend registry** — Plugin-based auto-discovery of model backends (supports third-party extensions)
- **Performance profiling** — Built-in profiling utilities for bottleneck identification
- **Nuitka build** — Build standalone executables with `build_nuitka.py`
- **pyproject.toml** — Proper Python packaging with optional dependency groups

## Installation

### Requirements

- Python 3.10+
- PyQt6
- NumPy

### Quick Start

```bash
# Clone the repository
git clone https://github.com/marmotte5/DataBuilder-.git
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
2. Click **Browse** next to "Output" and select where the project folder will be created
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

### 4. Export your project

- **Dry Run** (Ctrl+D) — Preview which images go to which buckets, with repeat counts
- **Export Project** (Ctrl+E) — Create the full project folder structure

### 5. Train

After exporting, switch to the **Training** tab and click **Start Training**. The trainer uses the same project folder — models, samples, and checkpoints all go to the right place automatically.

## Project Folder Structure

When you export, the output directory becomes a unified project folder:

```
my_project/                        # Your output folder
├── project.json                   # Project metadata
├── dataset/                       # Exported images organized by bucket
│   ├── 1_1_common/                # 1x repeat — most common tags
│   │   ├── 1_0001.png
│   │   ├── 1_0001.txt
│   │   ├── 1_0002.jpg
│   │   └── 1_0002.txt
│   ├── 5_10_medium/               # 5x repeat — medium rarity
│   │   ├── 10_0001.png
│   │   └── 10_0001.txt
│   └── 20_40_rare_concept/        # 20x repeat — rarest tags
│       ├── 40_0001.png
│       └── 40_0001.txt
├── models/                        # Trained model outputs (final weights)
├── samples/                       # Sample images generated during training
├── checkpoints/                   # Step/epoch checkpoint saves
│   ├── step_000500/
│   ├── epoch_0001/
│   └── final/
├── backups/                       # Full timestamped project backups
├── logs/                          # Training logs, TensorBoard events, and export error logs
└── .cache/                        # Latent / text encoder caches
```

### Folder naming: `{repeats}_{bucket}_{name}`

- **repeats** — How many times the training loop sees each image in this folder per epoch. Automatically calculated: common images = 1x, rare images = up to 20x.
- **bucket** — The bucket number (1-80) based on tag rarity percentile.
- **name** — The bucket name you set in the UI (defaults to "bucket").

### How repeats work

The repeat system ensures the model sees rare concepts as often as common ones:

| Bucket | Rarity     | Repeats | Effect                              |
|--------|------------|---------|-------------------------------------|
| 1      | Very common| 1x      | Seen once per epoch                 |
| 20     | Common     | 5x      | Seen 5 times per epoch              |
| 40     | Medium     | 10x     | Seen 10 times per epoch             |
| 60     | Rare       | 15x     | Seen 15 times per epoch             |
| 80     | Very rare  | 20x     | Seen 20 times per epoch             |

This prevents the model from over-learning common tags while under-learning rare ones.

## Source Structure

```
dataset_sorter/
├── __init__.py                  # Package marker
├── __main__.py                  # Entry point
├── constants.py                 # Global constants (17 model types, 12 optimizers, etc.)
├── models.py                    # Data classes (TrainingConfig, ImageEntry, DatasetStats)
├── utils.py                     # Path validation, sanitization, GPU detection
├── workers.py                   # QThread workers for scan/export, project structure, repeats
├── trainer.py                   # Core training engine with checkpoint/sample/backup
├── training_worker.py           # QThread worker for model training
├── training_presets.py          # 7 presets, ControlNet/DPO config, custom schedulers
├── training_history.py          # Cross-run training history for recommendations
├── smart_resume.py              # Loss curve analysis and auto-adjustment on resume
├── recommender.py               # Training parameter recommendation engine
├── recommender_profiles.py      # Per-model recommendation profiles
├── lr_preview.py                # LR schedule preview (pure math, no torch)
├── vram_estimator.py            # VRAM estimation for 17 model types
├── bucket_sampler.py            # Aspect ratio bucketing for multi-resolution training
├── train_dataset.py             # Training dataset with caching and augmentation
├── backend_registry.py          # Plugin registry for training backends (auto-discovery)
├── train_backend_base.py        # Base class for model-specific backends
├── train_backend_*.py           # 17 model backends (sd15, sdxl, flux, sd3, zimage, etc.)
├── optimizer_factory.py         # Optimizer creation (12 types)
├── optimizers.py                # Custom optimizer implementations (Marmotte, SOAP, Muon)
├── ema.py                       # Exponential Moving Average for weights
├── speed_optimizations.py       # torch.compile, MeBP, CUDA optimizations
├── triton_kernels.py            # Triton fused kernels (AdamW, MSE loss, flow interpolation)
├── fp8_training.py              # FP8 training wrapper (2x TFLOPS on Ada/Hopper GPUs)
├── sequence_packing.py          # Sequence packing for zero padding waste (DiT models)
├── mmap_dataset.py              # Memory-mapped dataset for zero-copy I/O
├── zero_bottleneck_dataloader.py # mmap + pinned DMA dataloader (bypasses GIL/pickle)
├── zimage_optimizations.py      # Z-Image (S3-DiT) exclusive speed optimizations
├── zimage_inventions.py         # Z-Image advanced inventions (L2 attention, speculative grad)
├── pipeline_integrator.py       # Pre-training pipeline, live monitoring, auto-config
├── async_io.py                  # Async data loading and GPU prefetching
├── io_speed.py                  # Fast image scanning, dimension reading, deduplication
├── dataset_management.py        # Caption augmentation, token counts, spell check
├── tag_importance.py            # Tag importance scoring for dataset cleaning
├── tag_specificity.py           # Tag specificity analysis
├── concept_probing.py           # Probe base model knowledge before training
├── token_weighting.py           # Per-token caption loss weighting
├── attention_map_debugger.py    # Attention visualization during training
├── curriculum_learning.py       # Loss-based adaptive image sampling
├── dpo_trainer.py               # DPO preference fine-tuning
├── masked_loss.py               # Masked training loss (region-specific)
├── noise_rescale.py             # Noise schedule rescaling (zero terminal SNR)
├── tensorboard_logger.py        # TensorBoard logging with graceful fallback
├── duplicate_detector.py        # Duplicate/near-duplicate image detection
├── disk_space.py                # Disk space validation
├── auto_pipeline.py             # One-click clean + train pipeline
├── config_validator.py          # Training config validation
├── metadata_cache.py            # Persistent metadata caching
├── profiling.py                 # Performance profiling utilities
├── generate_worker.py           # Image generation worker
└── ui/
    ├── theme.py                 # Dark/light theme with runtime toggling
    ├── toast.py                 # Toast notification system
    ├── tag_panel.py             # Virtual tag table (O(1) rendering)
    ├── override_panel.py        # Tools, stats, config panel
    ├── preview_tab.py           # Image thumbnail preview grid
    ├── image_tab.py             # Image browser with pixmap caching
    ├── dataset_tab.py           # Dataset management (6 sub-tabs)
    ├── dataset_sections.py      # Dataset sub-tab sections
    ├── reco_tab.py              # Training recommendations tab
    ├── training_tab.py          # Full training UI with 7 config sub-tabs
    ├── training_tab_builders.py # Training tab UI builders
    ├── training_config_io.py    # Training config save/load (JSON)
    ├── generate_tab.py          # Image generation tab
    ├── help_tab.py              # Getting started guide
    ├── dialogs.py               # Export preview dialog with repeat info
    ├── rlhf_dialog.py           # RLHF preference collection dialog
    ├── auto_pipeline_dialog.py  # One-click pipeline dialog
    └── main_window.py           # Main window orchestrator
```

## License

MIT License — see [LICENSE](LICENSE) for details.
