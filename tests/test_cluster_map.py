"""Smoke tests for the Latent Space Visualizer.

Validates the embedding worker's reducer logic on synthetic embeddings
(skips the expensive CLIP image-encoding step), and the cluster_map_tab
module-level constants. Falls back gracefully when optional deps are
missing so CI without UMAP/sklearn still passes.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


HAS_UMAP = importlib.util.find_spec("umap") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_PYQT = importlib.util.find_spec("PyQt6") is not None


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 not installed")
def test_embedding_worker_imports_clean():
    """The worker module must import without error when its deps are present."""
    from dataset_sorter.embedding_worker import EmbeddingWorker
    w = EmbeddingWorker(image_paths=[], reducer="umap")
    assert w.reducer == "umap"
    assert w.batch_size == 32
    assert w._stop_requested is False


@pytest.mark.skipif(not (HAS_PYQT and HAS_UMAP), reason="PyQt6 or UMAP missing")
def test_reduce_umap_on_synthetic_embeddings():
    """UMAP reducer should produce a (N, 2) array of finite floats."""
    from dataset_sorter.embedding_worker import EmbeddingWorker

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((50, 64)).astype(np.float32)
    # L2-normalize so the cosine metric behaves sanely
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)

    w = EmbeddingWorker(image_paths=[], reducer="umap")
    coords = w._reduce(embeddings)
    assert coords.shape == (50, 2)
    assert np.all(np.isfinite(coords))


@pytest.mark.skipif(not (HAS_PYQT and HAS_SKLEARN), reason="PyQt6 or sklearn missing")
def test_reduce_tsne_on_synthetic_embeddings():
    """t-SNE fallback path must also produce (N, 2) finite output."""
    from dataset_sorter.embedding_worker import EmbeddingWorker

    rng = np.random.default_rng(123)
    embeddings = rng.standard_normal((30, 64)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)

    w = EmbeddingWorker(image_paths=[], reducer="tsne")
    coords = w._reduce(embeddings)
    assert coords.shape == (30, 2)
    assert np.all(np.isfinite(coords))


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 not installed")
def test_reduce_handles_tiny_dataset():
    """For < 4 points the reducer must fall back to a grid placement."""
    from dataset_sorter.embedding_worker import EmbeddingWorker

    embeddings = np.random.randn(2, 16).astype(np.float32)
    w = EmbeddingWorker(image_paths=[], reducer="umap")
    coords = w._reduce(embeddings)
    assert coords.shape == (2, 2)


@pytest.mark.skipif(not HAS_PYQT, reason="PyQt6 not installed")
def test_cluster_map_tab_imports_clean():
    """The cluster_map_tab module must import without error."""
    from dataset_sorter.ui.cluster_map_tab import ClusterMapTab  # noqa: F401
    # Cannot instantiate without a QApplication; importing is enough to
    # catch syntax/import errors without spinning up the full Qt app.
