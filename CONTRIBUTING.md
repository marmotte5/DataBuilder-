# Contributing to DataBuilder

First off — thank you for considering a contribution. DataBuilder is an open-source project built by people who care about making AI model training fast and accessible. Every improvement matters.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Adding a New Model Backend](#adding-a-new-model-backend)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

---

## Code of Conduct

Be kind and constructive. We're all here to build something good. Criticism of code is welcome; criticism of people is not.

---

## How to Contribute

There are many ways to contribute beyond writing code:

- **Report bugs** — Open an issue with reproduction steps and your system info
- **Test new model backends** — Try training with new architectures and report issues
- **Improve documentation** — Fix typos, clarify confusing explanations, add examples
- **Add model backends** — Support for new architectures is always welcome
- **Optimize performance** — If you find a speedup, we want it
- **Share presets** — Good training presets for specific use cases
- **Answer questions** — Help other users in issues and discussions

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/DataBuilder-.git
cd DataBuilder-

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 3. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install DataBuilder in editable mode with dev dependencies
pip install -e ".[all,dev]"

# 5. Verify everything works
pytest tests/
python -m dataset_sorter
```

### Recommended Tools

- **Python 3.11+** — Newer versions have better error messages and performance
- **An IDE** — VSCode or PyCharm with the Python extension
- **A GPU** — At least 8 GB VRAM for testing training backends

---

## Project Architecture

Understanding the project structure will help you contribute effectively.

### Key Entry Points

| File | Purpose |
|---|---|
| `dataset_sorter/__main__.py` | Application entry point |
| `dataset_sorter/ui/main_window.py` | Main window, tab orchestration |
| `dataset_sorter/trainer.py` | Core training engine |
| `dataset_sorter/training_worker.py` | Background training QThread |
| `dataset_sorter/generate_worker.py` | Background generation QThread |
| `dataset_sorter/backend_registry.py` | Auto-discovery of model backends |
| `dataset_sorter/train_backend_base.py` | Base class all backends inherit |
| `dataset_sorter/models.py` | `TrainingConfig` dataclass and data models |
| `dataset_sorter/constants.py` | Model types, optimizers, presets lists |

### Threading Model

- **Main thread**: PyQt6 UI event loop — never block it
- **Training**: `training_worker.py` QThread
- **Generation**: `generate_worker.py` QThread
- **Scan/Export**: `workers.py` QThread workers
- Thread communication uses Qt signals/slots and `threading.Lock` for shared state

### Backend Plugin System

Model backends are auto-discovered via `backend_registry.py`. You don't need to modify any central registry file to add a new backend — just create the file with the right class name and it gets picked up automatically.

---

## Adding a New Model Backend

This is one of the most impactful contributions you can make. Here's the full process:

### 1. Create the backend file

```bash
touch dataset_sorter/train_backend_mymodel.py
```

### 2. Implement the backend

```python
"""Backend for MyModel — flow-matching DiT with T5-XXL text encoder."""

from __future__ import annotations
import logging
from typing import Any

import torch

from .train_backend_base import TrainBackendBase

log = logging.getLogger(__name__)


class MyModelBackend(TrainBackendBase):
    model_name = "MyModel"
    default_resolution = 1024
    prediction_type = "flow"  # or "epsilon" or "v_prediction"

    def load_model(self, model_path: str, config: Any) -> None:
        """Load pipeline from path (single-file or diffusers dir)."""
        self._load_single_file_or_pretrained(
            model_path,
            pipeline_class=...,  # e.g. StableDiffusion3Pipeline
            hf_fallback="org/mymodel-base",
        )
        # Extract components: self.transformer, self.vae, self.text_encoders, etc.
        ...

    def encode_text_batch(self, captions: list[str]) -> torch.Tensor:
        """Encode a batch of captions with the model's text encoder(s)."""
        ...

    def training_step(self, batch: dict, step: int) -> torch.Tensor:
        """Forward pass + loss. Call flow_training_step() for flow models."""
        return self.flow_training_step(batch, step)
```

### 3. Register the backend

Add your model to `dataset_sorter/constants.py`:

```python
MODEL_TYPES = [
    ...
    "MyModel",
]
```

Add it to `dataset_sorter/generate_worker.py` in `PIPELINE_MAP`:

```python
PIPELINE_MAP = {
    ...
    "MyModel": MyModelPipeline,
}
```

### 4. Write tests

```bash
# Create a test file
touch tests/test_mymodel_backend.py
```

Test at minimum: model loading (mocked), text encoding shape, a training step forward pass.

### 5. Open a pull request

Include in your PR description:
- Which model you're adding and a link to its HuggingFace page
- Whether you tested it with actual training (even 10 steps is useful)
- Known limitations or TODOs

### Common Pitfalls

- **Single-file checkpoint keys**: Some models store weights without the `transformer.` prefix. Always handle unprefixed keys (see `train_backend_zimage.py` for an example).
- **Flow vs epsilon**: Flow matching models should call `self.flow_training_step()`. Never mix prediction types.
- **VAE is always frozen**: Never add VAE parameters to the optimizer.
- **Text encoder caching**: If you cache TE outputs, make sure the cache path in `train_dataset.py` is consistent.
- **VRAM**: Every byte counts on 24 GB cards. Use `torch.cuda.empty_cache()` after offloading large components.
- **MPS compatibility**: `torch.Generator(device="mps")` breaks on many PyTorch versions. Use a CPU generator.

---

## Code Style

We follow a few simple conventions — no linter is enforced yet, but please match the style of existing code.

### Python Version

Python 3.10+ with modern type hints:

```python
# Good
def process(items: list[str], config: dict[str, Any] | None = None) -> bool: ...

# Avoid (old-style)
from typing import List, Dict, Optional
def process(items: List[str], config: Optional[Dict[str, Any]] = None) -> bool: ...
```

### Logging

Always use the module logger, never `print()`:

```python
import logging
log = logging.getLogger(__name__)

# Good
log.info("Loading model from %s", model_path)
log.warning("VRAM is low: %d MB remaining", vram_mb)

# Bad
print(f"Loading model from {model_path}")
```

### Imports

Standard library → third-party → local, with lazy imports for heavy deps:

```python
# stdlib
import os
import logging
from pathlib import Path

# third-party
import numpy as np

# local
from .models import TrainingConfig

# Inside methods only — keeps import time fast
def load_model(self, path: str) -> None:
    import torch
    from diffusers import FluxPipeline
    ...
```

### Docstrings

Module-level docstring explaining architecture decisions. Method docstrings for non-obvious logic only — don't document what is already obvious from the code.

```python
"""Backend for Flux — flow-matching DiT with dual text encoders (CLIP-L + T5-XXL).

Single-file loading falls back to downloading from Black Forest Labs HF repo.
Text encoder outputs are cached before training to save VRAM during the forward pass.
"""
```

### No Secrets

Never hardcode paths, tokens, API keys, or credentials. Everything must be configurable via the UI or config files. This is a public repository.

---

## Testing

We use `pytest`. Tests live in `tests/`.

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_marmotte_optimizer.py -v

# Run tests matching a keyword
pytest tests/ -k "optimizer" -v
```

Most tests mock heavy dependencies (torch, diffusers) so they run without a GPU. If your contribution requires GPU testing, note that in your PR.

When adding a feature:
- Add tests for the new functionality
- Make sure existing tests still pass
- Prefer unit tests over integration tests where possible

---

## Submitting a Pull Request

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-new-backend
   ```

2. **Make your changes** — small, focused PRs are reviewed faster

3. **Write tests** for new functionality

4. **Verify tests pass**:
   ```bash
   pytest tests/
   ```

5. **Commit with a clear message**:
   ```
   feat: add MyModel training backend

   Supports single-file and diffusers directory loading.
   Uses flow matching with T5-XXL text encoder.
   Tested on 100-image dataset, RTX 3090.
   ```

6. **Push and open a PR** against `main`

7. **Describe your changes** — what does this do, why, any trade-offs?

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] New functionality has tests
- [ ] No hardcoded paths, tokens, or credentials
- [ ] Code follows the style of the surrounding code
- [ ] PR description explains the change and motivation

---

## Reporting Bugs

Open an issue and include:

- **DataBuilder version** (or commit hash)
- **OS and Python version**
- **GPU model and VRAM**
- **Steps to reproduce** — exact sequence of actions
- **Expected vs actual behavior**
- **Error message / traceback** (full text, not a screenshot)
- **Minimal reproducing config** if training-related

The more detail you provide, the faster we can help.

---

## Feature Requests

Open an issue with the `enhancement` label. Describe:

- What you want to do that you currently can't
- Why it would be useful (for you and others)
- Any ideas on how to implement it (optional)

Speed improvements, new model support, and dataset management tools are especially welcome.

---

## Questions?

Open a GitHub issue. We're friendly.
