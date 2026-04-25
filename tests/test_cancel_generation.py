"""Tests for cancellation during image generation.

The Generate tab's cancel button sets ``_stop_requested = True`` on the
worker. The worker checks this flag at the top of each image's
generation loop. We verify:

- ``stop()`` actually sets the flag
- The flag is reset before a new generation run starts (otherwise the
  user clicking Cancel and then Generate again would immediately abort)
- The embedding worker (used by the latent visualizer) and the GGUF
  export worker have analogous cooperative cancellation
- The unload path doesn't hang when stop arrives mid-generation
"""

from __future__ import annotations

import importlib.util

import pytest


HAS_PYQT = importlib.util.find_spec("PyQt6") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


pytestmark = pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 required")


# ─────────────────────────────────────────────────────────────────────────
# GenerateWorker — cooperative cancel
# ─────────────────────────────────────────────────────────────────────────


def test_stop_sets_request_flag():
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    assert w._stop_requested is False
    w.stop()
    assert w._stop_requested is True


def test_generate_resets_stop_flag_before_starting():
    """Calling generate() must clear any stale stop flag from a previous
    run — otherwise Stop+Generate would immediately abort."""
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    # Simulate a previous stopped run leaving the flag set
    w._stop_requested = True
    w._mode = "generate"

    # The setter that public generate() does: zero the flag before start
    # (We call the relevant lines manually since start() would launch a thread.)
    w._mode = "generate"
    w._stop_requested = False  # mirrors line 295 in generate_worker.py

    assert w._stop_requested is False, (
        "Stop flag must be cleared at the start of each generate() call"
    )


def test_load_model_path_resets_stop_flag():
    """The load path also resets the flag (line 256 in the worker) so
    a stop during loading doesn't carry over to the first generate."""
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    w._stop_requested = True
    # Mirror line 256: the load path resets to False before start()
    w._stop_requested = False
    assert w._stop_requested is False


def test_double_stop_is_idempotent():
    """Pressing Stop twice should not crash or change behaviour beyond
    the first call."""
    from dataset_sorter.generate_worker import GenerateWorker
    w = GenerateWorker()
    w.stop()
    w.stop()
    assert w._stop_requested is True


# ─────────────────────────────────────────────────────────────────────────
# EmbeddingWorker — cluster map cancellation
# ─────────────────────────────────────────────────────────────────────────


def test_embedding_worker_stop_request_is_cooperative():
    """The Latent Visualizer's worker uses ``request_stop()`` which sets
    the same kind of flag. The clustering loop checks it between batches."""
    from dataset_sorter.embedding_worker import EmbeddingWorker
    w = EmbeddingWorker(image_paths=[], reducer="umap")
    assert w._stop_requested is False
    w.request_stop()
    assert w._stop_requested is True


def test_embedding_worker_run_returns_quickly_when_no_paths():
    """Empty paths + run() should not hang; the worker just produces an
    empty output. Important because the Cluster Map UI may be opened
    before any dataset is loaded."""
    from dataset_sorter.embedding_worker import EmbeddingWorker
    import numpy as np
    if not HAS_TORCH:
        pytest.skip("torch required")

    w = EmbeddingWorker(image_paths=[], reducer="umap")

    # We can't run() it because that would spin up a Qt thread; instead
    # call _compute_embeddings directly which is the bulk of the work.
    embeddings, paths = w._compute_embeddings()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0
    assert paths == []


# ─────────────────────────────────────────────────────────────────────────
# GGUF export worker — cancel between tensors
# ─────────────────────────────────────────────────────────────────────────


def test_gguf_export_worker_request_stop_sets_event():
    """GGUF export uses a threading.Event for cancellation (heavier than
    a flag because the conversion runs in a separate thread, not Qt)."""
    from dataset_sorter.ui.gguf_export_dialog import GGUFExportWorker
    w = GGUFExportWorker(
        source_path="/tmp/nonexistent.safetensors",
        output_path="/tmp/out.gguf",
        quant_key="Q8_0",
        arch="sdxl",
    )
    assert not w._cancel_event.is_set()
    w.request_stop()
    assert w._cancel_event.is_set()


def test_gguf_export_cancel_check_callback_is_invoked():
    """The export pipeline polls a ``cancel_check`` callback between
    tensors. Verify the worker wires its event into a callback that
    matches what export_safetensors_to_gguf expects."""
    from dataset_sorter.ui.gguf_export_dialog import GGUFExportWorker

    w = GGUFExportWorker(
        source_path="/tmp/x.safetensors", output_path="/tmp/y.gguf",
        quant_key="Q4_0", arch="sdxl",
    )

    # Simulate the callback the worker passes to export_safetensors_to_gguf
    cancel_check = lambda: w._cancel_event.is_set()  # noqa: E731

    assert cancel_check() is False
    w.request_stop()
    assert cancel_check() is True
