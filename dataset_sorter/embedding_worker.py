"""Embedding + dimensionality-reduction worker for the Latent Space Visualizer.

Computes CLIP image embeddings for the current dataset and projects them onto
a 2D plane using UMAP (or t-SNE as a fallback). Designed to be run in a
QThread so the UI stays responsive even on 5000+ image datasets.

Why CLIP image embeddings instead of VAE latents?
    - CLIP is trained for semantic similarity → clusters group by content
      ("portrait of woman", "outdoor landscape") which matches the diagnostic
      use case ("am I missing a concept?").
    - VAE latents are 16384-dim and dominated by texture/colour, producing
      uninformative blobs for human inspection.
    - CLIP-ViT-B/32 is small (~150 MB), runs in ~2 ms/image on RTX 4090.

The worker emits per-batch progress so the UI can show a progress bar while
embeddings are being computed; coords arrive once the reducer finishes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


# CLIP model id — small ViT-B/32 (~150 MB) is the sweet spot for desktop:
# fast on CPU, tiny on GPU, and the embedding quality is plenty for clustering.
_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class EmbeddingWorker(QThread):
    """Compute CLIP embeddings + 2D projection for a list of image paths.

    Signals:
        progress(int, int, str): (done, total, message) per-batch updates.
        coords_ready(np.ndarray, list[str]): 2D coords (N, 2) + ordered paths.
        error(str): emitted on any failure; the worker terminates afterwards.
    """

    progress = pyqtSignal(int, int, str)
    coords_ready = pyqtSignal(object, object)  # np.ndarray (N,2), list[str]
    error = pyqtSignal(str)

    def __init__(
        self,
        image_paths: list[str],
        reducer: str = "umap",
        batch_size: int = 32,
        parent=None,
    ):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.reducer = reducer.lower()
        self.batch_size = batch_size
        self._stop_requested = False

    def request_stop(self):
        """Cooperative stop — checked between batches."""
        self._stop_requested = True

    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        try:
            embeddings, valid_paths = self._compute_embeddings()
            if self._stop_requested or len(valid_paths) == 0:
                return
            coords = self._reduce(embeddings)
            if self._stop_requested:
                return
            self.coords_ready.emit(coords, valid_paths)
        except Exception as e:  # noqa: BLE001
            log.exception("EmbeddingWorker failed: %s", e)
            self.error.emit(f"{type(e).__name__}: {e}")

    # ─────────────────────────────────────────────────────────────────────

    def _compute_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Load CLIP and embed every image in batches."""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        total = len(self.image_paths)
        if total == 0:
            return np.zeros((0, 512), dtype=np.float32), []

        self.progress.emit(0, total, "Loading CLIP-ViT-B/32...")

        # Pick the best available device but stay tolerant if accel is missing.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32  # MPS prefers fp32 for CLIP stability
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        model = CLIPModel.from_pretrained(_CLIP_MODEL_ID, torch_dtype=dtype).to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)

        embeddings: list[np.ndarray] = []
        valid_paths: list[str] = []
        done = 0

        for batch_start in range(0, total, self.batch_size):
            if self._stop_requested:
                break
            batch_paths = self.image_paths[batch_start:batch_start + self.batch_size]
            images: list[Image.Image] = []
            kept_paths: list[str] = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    kept_paths.append(p)
                except Exception as e:  # noqa: BLE001
                    log.warning("Skipping %s: %s", p, e)

            if not images:
                done += len(batch_paths)
                self.progress.emit(done, total, f"Skipped batch ({batch_start})")
                continue

            inputs = processor(images=images, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            # L2-normalize so distance ≈ 1 - cosine similarity (UMAP-friendly).
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            embeddings.append(feats.float().cpu().numpy())
            valid_paths.extend(kept_paths)

            done += len(batch_paths)
            self.progress.emit(done, total, f"Embedding {done}/{total} images")

        # Free CLIP weights as soon as we're done — UMAP fit is CPU-only.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

        if not embeddings:
            return np.zeros((0, 512), dtype=np.float32), []

        return np.concatenate(embeddings, axis=0).astype(np.float32), valid_paths

    # ─────────────────────────────────────────────────────────────────────

    def _reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """Project (N, D) embeddings down to (N, 2) for plotting.

        UMAP is the default — preserves both local and global structure better
        than t-SNE for this use case. Falls back to t-SNE on UMAP failure or
        when the user explicitly requests it.
        """
        n = embeddings.shape[0]
        if n < 4:
            # Too few points for any reducer — place them on a tiny grid.
            self.progress.emit(n, n, "Tiny dataset — placing points on a grid")
            grid = np.array([[i % 2, i // 2] for i in range(n)], dtype=np.float32)
            return grid

        self.progress.emit(n, n, f"Reducing with {self.reducer.upper()}...")

        if self.reducer == "tsne":
            return self._reduce_tsne(embeddings)
        # default: umap
        try:
            return self._reduce_umap(embeddings)
        except Exception as e:  # noqa: BLE001
            log.warning("UMAP failed (%s) — falling back to t-SNE", e)
            return self._reduce_tsne(embeddings)

    def _reduce_umap(self, embeddings: np.ndarray) -> np.ndarray:
        import umap
        n = embeddings.shape[0]
        # n_neighbors controls local vs global structure; cap to N-1.
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            n_jobs=1,  # avoid threadpool deadlocks inside QThread
        )
        coords = reducer.fit_transform(embeddings)
        return coords.astype(np.float32)

    def _reduce_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        import inspect
        n = embeddings.shape[0]
        # perplexity must be < N (sklearn requirement).
        perplexity = max(5.0, min(30.0, (n - 1) / 3))
        kwargs: dict = dict(
            n_components=2,
            perplexity=perplexity,
            metric="cosine",
            init="pca",
            random_state=42,
        )
        # sklearn 1.5+ renamed n_iter → max_iter; pick whichever the
        # installed version exposes so the call works on both old and new.
        sig = inspect.signature(TSNE.__init__).parameters
        if "max_iter" in sig:
            kwargs["max_iter"] = 1000
        elif "n_iter" in sig:
            kwargs["n_iter"] = 1000
        reducer = TSNE(**kwargs)
        coords = reducer.fit_transform(embeddings)
        return coords.astype(np.float32)
