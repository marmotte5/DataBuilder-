# DataBuilder — Quick Start Guide

Your first LoRA from scratch, step by step.

This guide takes you from zero to a trained LoRA in about 20 minutes (plus training time). We'll train a **character LoRA** on SDXL — one of the most common use cases — but the process is identical for other model types.

---

## Prerequisites

- DataBuilder installed and running (`python -m dataset_sorter`)
- A GPU with at least **8 GB VRAM** (16+ GB recommended for SDXL)
- **20-50 images** of your subject — more is better, but 20 high-quality images beats 200 low-quality ones
- An SDXL base model (e.g., `stabilityai/stable-diffusion-xl-base-1.0` or a `.safetensors` file)

If you haven't installed DataBuilder yet, see the [README](../README.md#quick-start).

---

## Step 1 — Prepare Your Images

Before you even open DataBuilder, take a few minutes to prepare your dataset. This is where most training quality is won or lost.

### Image quality checklist

- **Resolution**: At least 768×768 pixels. Higher is better.
- **Variety**: Different angles, lighting, backgrounds, expressions, outfits.
- **Consistency**: All images should clearly show your subject.
- **Cleanliness**: No watermarks, text overlays, or heavily compressed images.

### Captioning

DataBuilder works best with captioned images. Each image should have a `.txt` file with the same name:

```
my_dataset/
├── photo_001.jpg
├── photo_001.txt    ← "a woman with red hair, smiling, outdoor setting"
├── photo_002.png
├── photo_002.txt    ← "a woman with red hair, serious expression, studio lighting"
└── ...
```

**Recommended captioning tools:**
- [WD14 Tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) — Fast, good for anime-style content
- [BLIP-2 / LLaVA](https://github.com/salesforce/BLIP) — Better for realistic photos
- Manual captioning — Highest quality, most control

**Tip:** Include a unique trigger word in every caption (e.g., `"ohwx woman"`). This helps the model associate the word with your subject and makes generation much more controllable.

---

## Step 2 — Scan Your Dataset

1. **Open DataBuilder** (`python -m dataset_sorter`)

2. **Set the Source folder**: Click **Browse** next to "Source" and select your image folder (or drag & drop the folder onto the window).

3. **Set the Output folder**: Click **Browse** next to "Output" and pick an empty folder where your project will be created. DataBuilder will create the full project structure here.

4. **Configure workers**: The default (4 workers) is fine for most cases. Increase it if you have a fast NVMe SSD.

5. **Click Scan** (or press **Ctrl+R**)

DataBuilder will analyze all images and their captions, compute tag frequencies, and organize everything into rarity buckets. This takes a few seconds for a small dataset, up to a minute for large ones.

---

## Step 3 — Review and Clean Your Tags

After scanning, the main tag panel shows all tags sorted by rarity. This is where you shape what the model will learn.

### What to look for

**Tags to delete** (click the tag → press Delete):
- Typos and misspellings (`blond hair` vs `blonde hair`)
- Generic noise tags that add nothing (`image`, `photo`, `picture`)
- Tags that conflict with what you're training (`multiple people` if training a single character)
- Overly specific tags that only appear once (`wearing specific brand shirt 2019`)

**Tags to keep and protect**:
- Your trigger word — make sure it's present and consistent
- Physical attributes you want the model to learn
- Style tags if you're doing a style LoRA

### Useful tools

- **Search box** (top of tag panel) — Filter tags by text
- **Tag histogram** (Dataset tab → Histogram) — See frequency distribution
- **Spell check** (Dataset tab → Spell Check) — Find typos and near-duplicates
- **Undo/Redo** — Ctrl+Z / Ctrl+Shift+Z — experiment freely

### Bucket overrides (optional)

Each tag is assigned a "rarity bucket" from 1-80. Common tags get fewer repeats; rare tags get more. If a tag is in the wrong bucket, you can override it:

1. Select the tag in the left panel
2. In the right panel, change "Override Bucket"

Most of the time, the automatic assignment is correct. Don't over-tweak this.

---

## Step 4 — Export Your Project

Once your tags look clean:

1. **Dry Run** (Ctrl+D) — Preview which images go to which buckets and how many repeats they'll get. No files are written. Check that the distribution looks reasonable.

2. **Export Project** (Ctrl+E) — Creates the full project folder structure:

```
my_project/
├── dataset/              ← Your organized, captioned images
│   ├── 1_1_common/       ← Common tags, 1× repeat
│   ├── 5_30_medium/      ← Medium rarity, 5× repeat
│   └── 20_75_rare/       ← Rare tags, 20× repeat
├── models/               ← Trained LoRA will appear here
├── samples/              ← Sample images generated during training
├── checkpoints/          ← Step checkpoints
├── logs/                 ← Training logs
└── .cache/               ← Latent cache (speeds up training)
```

Export completes in seconds for small datasets, a few minutes for large ones.

---

## Step 5 — Configure Training

Switch to the **Training** tab.

### Load a preset

Click **Preset** and select **Character LoRA**. This fills in sensible defaults for:
- LoRA rank (32)
- Learning rate (1e-4)
- Optimizer (Marmotte — fast and memory-efficient)
- Steps (2000)
- Sampling prompt (customize this to your trigger word)

You can also use:
- **Style LoRA** — For training an art style; uses lower rank (16) and longer training
- **Quick Test** — 200 steps, useful for verifying your setup before a full run

### Set your model

In the **Model** sub-tab:
1. **Base model** — Click Browse and point to your SDXL `.safetensors` file, or enter a HuggingFace repo ID (`stabilityai/stable-diffusion-xl-base-1.0`)
2. **Model type** — Select `SDXL` (or `Pony` if you're using a Pony-based model)
3. **Output name** — What to call your LoRA (e.g., `my_character_v1`)

### Check VRAM

Click **Estimate VRAM** to see how much GPU memory your configuration will use.

| Configuration | VRAM usage |
|---|---|
| SDXL LoRA, rank 32, batch 1 | ~14 GB |
| SDXL LoRA, rank 16, batch 1 | ~12 GB |
| SDXL LoRA, rank 8, FP8 base | ~8 GB |
| Flux LoRA, rank 16, batch 1 | ~20 GB |

If you're over budget:
- Reduce LoRA rank
- Enable **FP8 base model** (if you have an RTX 4000+ series)
- Enable **Gradient checkpointing**
- Reduce batch size to 1

### Preview the LR schedule

Click **Preview LR** to see an ASCII graph of your learning rate over training. The default warmup + cosine schedule looks like:

```
LR
^
|  /\
| /  \________
|/
+-------------> steps
```

This is a good shape — fast warmup, stable plateau, gentle cooldown.

### Set your sample prompt

In the **Sampling** sub-tab, set a prompt that includes your trigger word:

```
ohwx woman, portrait, professional photography, sharp focus
```

DataBuilder will generate sample images at regular intervals so you can watch training progress visually.

---

## Step 6 — Start Training

Click **Start Training**.

DataBuilder will:
1. Load your base model
2. Encode and cache all your captions (first run only — speeds up subsequent runs)
3. Encode and cache all your image latents (first run only)
4. Begin training

### What to watch

- **Loss curve** — Should decrease over time with minor fluctuations. A flat curve means the LR is too low; an exploding curve means too high.
- **VRAM bar** — Keep an eye on this, especially early in training.
- **Sample images** — Generated at the step interval you configured. You'll see the subject emerge as training progresses.

### Mid-training controls

| Button | What it does |
|---|---|
| **Save Now** | Saves the current LoRA weights immediately |
| **Sample Now** | Generates a sample image at the current step |
| **Backup Project** | Creates a timestamped zip of the entire project |
| **Pause/Resume** | Pauses training gracefully (no step is wasted) |
| **Stop** | Stops training and saves the current LoRA |

---

## Step 7 — Evaluate Your LoRA

After training completes, your LoRA is in `my_project/models/my_character_v1.safetensors`.

### Quick test in DataBuilder

Switch to the **Generate** tab:
1. Load your base model
2. Load your LoRA (`my_project/models/my_character_v1.safetensors`)
3. Set LoRA strength to 0.8-1.0
4. Write a test prompt: `ohwx woman, portrait, smiling`
5. Generate!

### A/B comparison

Use the **Comparison** tab to generate the same prompt with and without your LoRA side-by-side. This makes it easy to judge how much the LoRA is contributing.

---

## Troubleshooting

### "CUDA out of memory"

- Reduce LoRA rank (try 16 or 8)
- Enable FP8 base model
- Set batch size to 1
- Enable gradient checkpointing

### Loss is not decreasing

- Check that captions are correct and contain the trigger word
- Try a higher learning rate (1e-4 → 2e-4)
- Make sure your dataset has enough variety (>15 images with different poses/lighting)

### Generated images don't look like my subject

- Your trigger word may not be consistent across captions — check it
- Training may need more steps (try 3000-5000 for complex subjects)
- Try increasing LoRA rank to 64

### Training is very slow

- Enable `torch.compile` in the **Advanced** sub-tab
- Enable sequence packing if your images vary widely in resolution
- Check that you're using the GPU (look at VRAM bar — it should be > 0)

### Poor image quality / artifacts

- Check your source images for quality issues
- Reduce learning rate
- Try a different noise schedule (Advanced tab → Noise rescaling)

---

## Next Steps

Now that you've completed your first training run, explore more advanced features:

- **[Style LoRA](../README.md#training)** — Train on an artist's style instead of a subject
- **[DPO training](../README.md#training)** — Fine-tune with preference data for better quality
- **[ControlNet training](../README.md#training)** — Add spatial conditioning to your model
- **[Batch generation](../README.md#generation--library)** — Generate hundreds of variations automatically
- **[Model merging](../README.md#generation--library)** — Combine LoRAs or merge base models
- **Custom optimizers** — Try SOAP or Muon for potentially better quality

Happy training!
