# Dataset Sorter

A desktop application for sorting image datasets by tag rarity — designed for Stable Diffusion / LoRA training dataset preparation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Tag-based sorting** — Automatically sorts images into 80 buckets based on tag rarity using numpy percentile quantiles
- **Training recommendations** — State-of-the-art parameter suggestions for SD 1.5, SDXL, Flux, SD3, and Pony Diffusion (LoRA and full finetune)
- **8 model types** — SD 1.5 LoRA/Full, SDXL LoRA/Full, Flux LoRA, SD3 LoRA, Pony LoRA/Full
- **6 optimizers** — Adafactor, Prodigy, AdamW, AdamW 8-bit, D-Adapt Adam, SGD
- **4 network types** — LoRA, DoRA, LoHa, LoKr with auto-tuned rank/alpha
- **6 VRAM profiles** — 8, 12, 16, 24, 48, 96 GB with optimized batch/memory settings
- **Tag management** — Rename, merge, search & replace, delete/restore tags
- **Manual overrides** — Force specific tags or images into custom buckets
- **Image browser** — Navigate and preview images with per-image bucket control
- **Parallel scanning** — Configurable worker threads for fast dataset loading
- **Optional GPU validation** — Detect corrupt images early using torchvision (requires CUDA)
- **Config persistence** — Save/load your overrides, deleted tags, and bucket names as JSON
- **Security-first design** — READ-ONLY on source files; all writes go to the output directory only

## Screenshot

The application features a modern dark theme with three-panel layout:
- **Left** — Tag table with filtering, counts, buckets, overrides, and deletion status
- **Center** — Tools panel with override controls, tag editor, bucket names, config, and statistics
- **Right** — Tabbed view with image preview, training recommendations, and image browser

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

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m dataset_sorter
```

### Alternative: Run directly

```bash
python dataset_sorter.py
```

### Optional: GPU validation

To enable GPU-accelerated image validation (catches corrupt files during scan):

```bash
pip install torch torchvision
```

The GPU checkbox in the scan bar will be automatically enabled when CUDA is available.

### System dependencies (Linux)

On headless Linux or minimal installations, you may need:

```bash
sudo apt-get install libegl1 libgl1
```

## Usage

### 1. Scan a dataset

1. Click **Browse** next to "Source" and select your image dataset folder
2. Click **Browse** next to "Output" and select where sorted images will be exported
3. Adjust **Workers** count (higher = faster on SSDs with large datasets)
4. Optionally enable **GPU validation** if you have CUDA
5. Click **Scan**

The scanner recursively finds all images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.gif`) and reads their associated `.txt` tag files.

### 2. Review and edit tags

- **Filter** tags using the search box in the left panel
- **Select** one or more tags to see image previews
- **Override** bucket assignments for specific tags
- **Delete** tags you don't want in the exported captions
- **Rename**, **merge**, or **search & replace** tags in bulk

### 3. Get training recommendations

Switch to the **Recommendations** tab and configure:
- **Model type** (SD 1.5, SDXL, Flux, SD3, Pony)
- **VRAM** (8–96 GB)
- **Network type** (LoRA, DoRA, LoHa, LoKr)
- **Optimizer** (Adafactor, Prodigy, AdamW, etc.)

Click **Recalculate** to get optimized parameters including learning rate, batch size, epochs, rank/alpha, scheduler, and advanced settings (noise offset, min SNR gamma, caption dropout).

### 4. Export

- **Dry Run** — Preview which images go to which buckets before committing
- **Export** — Copy images and filtered tag files to the output directory, organized into numbered bucket folders

Exported tag files have deleted tags removed automatically. Source files are **never** modified.

## Project Structure

```
dataset_sorter/
├── __init__.py          # Package marker
├── __main__.py          # Entry point
├── constants.py         # Global constants (image extensions, model types, etc.)
├── models.py            # Data classes (TrainingConfig, ImageEntry, DatasetStats)
├── utils.py             # Path validation, sanitization, GPU detection
├── recommender.py       # Training parameter recommendation engine
├── workers.py           # QThread workers for scan and export
└── ui/
    ├── theme.py         # Dark theme colors and stylesheets
    ├── tag_panel.py     # Left panel — tag table
    ├── override_panel.py# Center panel — tools
    ├── preview_tab.py   # Image thumbnail preview
    ├── reco_tab.py      # Training recommendations tab
    ├── image_tab.py     # Image browser tab
    ├── dialogs.py       # Export summary dialog
    └── main_window.py   # Main window orchestrator
```

## How It Works

### Tag Rarity Sorting

1. All images are scanned and their `.txt` tag files are parsed (comma-separated tags)
2. Tag frequencies are counted across the entire dataset
3. Tags are assigned to 80 buckets using numpy percentile-based quantiles — rare tags get high bucket numbers, common tags get low numbers
4. Each image is assigned to the bucket of its rarest active tag
5. On export, images are copied into numbered folders (`1_bucket/`, `2_bucket/`, etc.)

### Training Recommendations

The recommendation engine incorporates community best practices (2025–2026) from kohya_ss, OneTrainer, SimpleTuner, and civitai guides:

- **Flux**: LR 5–10x higher than SDXL (base 2e-3), both text encoders frozen, steps capped at 1500–2000
- **Prodigy**: LR=1.0 (self-adjusting), d_coef=0.8, decouple=True, 25% more steps
- **Adafactor**: scale_parameter=False, 5x LR multiplier for LoRA
- **Min SNR gamma=5**: Always enabled (~3.4x faster convergence, ICCV 2023)
- **Network rank**: Auto-scaled by model type, dataset size, diversity, and VRAM
- **VRAM profiles**: Tested batch/gradient accumulation/checkpointing combos per model+VRAM pair

## Configuration

Settings are saved as JSON and include:
- Manual tag overrides
- Bucket names
- Deleted tags
- Source/output directory paths

## License

MIT License — see [LICENSE](LICENSE) for details.
