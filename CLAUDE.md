# DataBuilder

## What This Is

DataBuilder is a high-performance desktop application for text-to-image AI model training and generation. It covers the full pipeline: dataset preparation, training (LoRA and full finetune), image generation, and model/LoRA library management. Built with PyQt6, targeting RTX 4090 / H100 class hardware.

**Goal: Be the fastest trainer and generator for text-to-image models.**

## How to work on this codebase (LLM behavioral guidelines)

Adapted from Andrej Karpathy's CLAUDE.md guidelines
(https://github.com/forrestchang/andrej-karpathy-skills). These bias
toward caution over speed; for trivial tasks use judgment.

### 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports / variables / functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### 5. Project-specific reminders for these guidelines

- **Never run user-facing destructive ops without confirmation** —
  see "Security" + the trust_remote_code dialog precedent.
- **Verify audit-agent findings before acting on them.** Recent rounds
  showed ~60-85% false-positive rates on STE / FP8 / PEFT semantics
  the agent didn't fully understand. Read the actual code at
  the cited file:line, write a failing test that proves the bug,
  THEN fix.
- **Don't silently regress convergence.** When fixing optimizer or
  loss bugs, run the existing convergence smoke tests — see
  ``tests/test_marmotte_optimizer.py::TestMarmotteBasic`` for the
  cautionary tale of "fixing" the dormant rank-k feedback only to
  blow up gradients.
- **Match the existing style of huge files** — ``trainer.py`` and
  ``main_window.py`` are 3500-line god classes by design (UI signal
  wiring + sequential setup steps). Don't refactor them mid-feature;
  add to the right section instead.

**These guidelines are working if:** fewer unnecessary changes in
diffs, fewer rewrites due to overcomplication, and clarifying
questions come before implementation rather than after mistakes.

## Security

**NEVER commit API keys, tokens, or credentials.** This is a public open-source project (MIT). All secrets stay on the user's machine. No telemetry, no cloud calls except HuggingFace model downloads.

## Quick Reference

```
Entry point:     dataset_sorter/__main__.py
Main window:     dataset_sorter/ui/main_window.py
Core trainer:    dataset_sorter/trainer.py
Training worker: dataset_sorter/training_worker.py (QThread)
Generate worker: dataset_sorter/generate_worker.py (QThread)
Batch generate:  dataset_sorter/ui/batch_generation_tab.py
A/B comparison:  dataset_sorter/ui/comparison_tab.py
Model merge:     dataset_sorter/ui/model_merge_tab.py
Library:         dataset_sorter/ui/library_tab.py
Backend base:    dataset_sorter/train_backend_base.py
Model backends:  dataset_sorter/train_backend_*.py (17 files)
Optimizers:      dataset_sorter/optimizers.py (Marmotte, SOAP, Muon)
Triton kernels:  dataset_sorter/triton_kernels.py
Config/models:   dataset_sorter/models.py (TrainingConfig dataclass)
Constants:       dataset_sorter/constants.py
Tests:           tests/ (pytest)
```

## Architecture

### Backend Registry Pattern
Model backends are auto-discovered via `backend_registry.py`. Each backend inherits `TrainBackendBase` and overrides:
- `load_model()` - Load pipeline, extract components
- `encode_text_batch()` - Model-specific text encoding
- `training_step()` - Forward pass + loss computation
- `get_added_cond()` - Extra conditioning (time_ids for SDXL, etc.)

### 17 Supported Model Types
| Model | Backend File | Prediction | Text Encoder(s) |
|-------|-------------|------------|-----------------|
| SD 1.5 | `train_backend_sd15.py` | epsilon | CLIP |
| SD 2.x | `train_backend_sd2.py` | v-prediction | OpenCLIP |
| SDXL/Pony | `train_backend_sdxl.py` | epsilon | CLIP-L + CLIP-G |
| SD3 | `train_backend_sd3.py` | flow | CLIP-L + CLIP-G + T5-XXL |
| SD 3.5 | `train_backend_sd35.py` | flow | CLIP-L + CLIP-G + T5-XXL |
| Flux | `train_backend_flux.py` | flow | CLIP-L + T5-XXL |
| Flux 2 | `train_backend_flux2.py` | flow | LLM (Mistral-3/Qwen-3) |
| Z-Image | `train_backend_zimage.py` | flow | Qwen3 (chat template) |
| PixArt | `train_backend_pixart.py` | flow | T5-XXL |
| Kolors | `train_backend_kolors.py` | epsilon | ChatGLM-6B |
| Cascade | `train_backend_cascade.py` | epsilon | CLIP-G |
| Chroma | `train_backend_chroma.py` | flow | T5-XXL |
| AuraFlow | `train_backend_auraflow.py` | flow | T5 |
| Sana | `train_backend_sana.py` | flow | Gemma-2B |
| HunyuanDiT | `train_backend_hunyuan.py` | epsilon | CLIP-L + mT5 |
| HiDream | `train_backend_hidream.py` | flow | CLIP-L + CLIP-G + T5-XXL + Llama |

### Single-File Loading
All backends support both diffusers directories AND single-file `.safetensors`/`.ckpt` checkpoints. The loading strategy (in `train_backend_base.py`):
1. Try `Pipeline.from_single_file()`
2. Try `DiffusionPipeline.from_single_file()`
3. Load base pipeline from HuggingFace + swap fine-tuned transformer weights

Z-Image has additional handling for unprefixed checkpoint keys (no `transformer.` prefix).

### Threading Model
- **Main thread**: PyQt6 UI event loop
- **Scan/Export**: `workers.py` QThread workers
- **Training**: `training_worker.py` QThread
- **Generation**: `generate_worker.py` QThread
- Thread safety via `threading.Lock` on shared pipeline state

### Speed Optimizations (The Marmotte Philosophy)
Speed is the #1 priority after correctness. Key systems:

| System | File | Speedup |
|--------|------|---------|
| torch.compile | `speed_optimizations.py` | 20-40% |
| Triton fused kernels | `triton_kernels.py` | 5-15% |
| FP8 training | `fp8_training.py` | 2x TFLOPS |
| channels_last | `train_backend_base.py` | 10-20% |
| Sequence packing | `sequence_packing.py` | 10-30% |
| Zero-bottleneck DataLoader | `zero_bottleneck_dataloader.py` | Bypasses GIL |
| mmap datasets | `mmap_dataset.py` | Zero-copy I/O |
| Marmotte optimizer | `optimizers.py` | 10-20x less memory than Adam |

### Marmotte Optimizer
Custom ultra-low-memory optimizer in `optimizers.py`. Uses per-channel adaptive learning rates instead of per-parameter, achieving ~10-20x memory savings vs AdamW while maintaining training quality. This is our invention.

## Coding Conventions

- **Python 3.10+** with type hints (`list[str]`, `dict[str, Any]`, `X | None`)
- **Logging**: `log = logging.getLogger(__name__)` at module top, never `print()`
- **Docstrings**: Module-level docstring explaining architecture, method docstrings for non-obvious logic
- **Imports**: stdlib first, third-party second, local third. Lazy imports for heavy deps (torch, diffusers, transformers) inside methods to keep import time fast
- **Error handling**: Catch specific exceptions, log with context, re-raise or emit Qt signal
- **No secrets**: Never hardcode paths, tokens, API keys. Everything configurable via UI or JSON
- **Thread safety**: QThread for background work, `threading.Lock` for shared state, signals for UI updates

## How to Run

```bash
pip install ".[all]"       # Full install
python -m dataset_sorter   # Launch GUI
pytest tests/              # Run tests
```

## How to Add a New Model Backend

1. Create `dataset_sorter/train_backend_{name}.py`
2. Inherit `TrainBackendBase`
3. Set `model_name`, `default_resolution`, `prediction_type`
4. Implement `load_model()` using `_load_single_file_or_pretrained()` with a fallback HF repo
5. Implement `encode_text_batch()` for the model's text encoder(s)
6. Override `training_step()` if non-standard (flow matching models call `flow_training_step()`)
7. Register in `backend_registry.py` and `constants.py`
8. Add to `PIPELINE_MAP` in `generate_worker.py`

## Common Pitfalls

- **Single-file checkpoints**: Some models store weights without `transformer.` prefix. Always handle unprefixed keys (see Z-Image backend).
- **Flow vs epsilon**: Flow matching models use `flow_training_step()`, epsilon/v-pred use the base `training_step()`. Don't mix them.
- **VAE freezing**: VAE is always frozen and on-device. Never train it.
- **Text encoder caching**: TE outputs are cached before training starts. If you change TE preprocessing, the cache path in `train_dataset.py` must match.
- **MPS compatibility**: `torch.Generator(device="mps")` doesn't work in many PyTorch versions. Use CPU generator.
- **VRAM**: RTX 4090 = 24 GB. Every byte matters. Use `empty_cache()` after offloading.

## Project Roadmap

- Dataset sorting and tag management (done)
- Integrated training with 17 model types (done)
- Image generation with LoRA support (done)
- Marmotte optimizer (done)
- Extreme speed optimizations (done)
- LoRA and model library management (done) — favorites, notes, tags, ratings
- Batch generation workflows (done) — prompt queue, CSV/JSON/TXT import, auto-save
- Model merging tools (done) — weighted sum, SLERP, add-difference
- A/B comparison interface (done) — side-by-side generation with per-side overrides
