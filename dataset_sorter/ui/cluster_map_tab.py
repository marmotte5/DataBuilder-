"""Latent Space Visualizer — interactive scatter plot of dataset embeddings.

Projects every image in the loaded dataset onto a 2D map using CLIP embeddings
+ UMAP / t-SNE. Lets the user spot:
- **Empty regions** (concept gaps — "I have no profile shots")
- **Dense clusters** (over-representation — "70% of my images are smiling")
- **Outliers** (mislabelled or off-topic shots that drift away from clusters)

Architecture:
    ClusterMapTab(QWidget)
        ├── toolbar  — Compute / Reducer dropdown / progress bar / stats
        ├── ClusterMapView (QGraphicsView)
        │       └── ClusterScene (QGraphicsScene with N PointItems)
        └── thumbnail preview (QLabel, 240x240, follows hover)

The point items are drawn manually with QGraphicsEllipseItem + a hover hook
that emits a signal upstream so we can show a thumbnail of the corresponding
image. Click selects, double-click navigates to that image in the Images tab.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QCursor, QPainter, QPen, QPixmap, QImage, QWheelEvent,
)
from PyQt6.QtWidgets import (
    QComboBox, QGraphicsEllipseItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
    QSizePolicy,
)

from dataset_sorter.embedding_worker import EmbeddingWorker
from dataset_sorter.ui.theme import (
    COLORS, accent_button_style, nav_button_style,
    section_header_style, muted_label_style,
)

log = logging.getLogger(__name__)


# Visual constants — tuned for ≤2000 points to stay readable.
_POINT_RADIUS = 4.0
_POINT_HOVER_RADIUS = 7.0
_VIEW_PADDING = 24.0
_THUMBNAIL_SIZE = 240


# ─────────────────────────────────────────────────────────────────────────────
# Scatter point — one per image
# ─────────────────────────────────────────────────────────────────────────────


class _PointItem(QGraphicsEllipseItem):
    """A single point on the scatter plot.

    Stores the image path and emits hover events through the parent view
    (Qt graphics items can't hold signals — the view handles fan-out).
    """

    def __init__(self, x: float, y: float, image_path: str, color: QColor):
        super().__init__(-_POINT_RADIUS, -_POINT_RADIUS,
                         2 * _POINT_RADIUS, 2 * _POINT_RADIUS)
        self.setPos(x, y)
        self.image_path = image_path
        self._base_color = color
        self._base_brush = QBrush(color)
        self._base_pen = QPen(color.darker(140), 0.5)
        self.setBrush(self._base_brush)
        self.setPen(self._base_pen)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setZValue(0)

    def hoverEnterEvent(self, event):
        # Grow + brighten on hover for clear feedback.
        self.setRect(-_POINT_HOVER_RADIUS, -_POINT_HOVER_RADIUS,
                     2 * _POINT_HOVER_RADIUS, 2 * _POINT_HOVER_RADIUS)
        glow = QColor(self._base_color)
        glow.setAlpha(255)
        self.setBrush(QBrush(glow))
        self.setPen(QPen(QColor(COLORS["accent"]), 2.0))
        self.setZValue(10)
        scene = self.scene()
        if scene is not None:
            scene.point_hovered.emit(self.image_path)  # type: ignore[attr-defined]
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setRect(-_POINT_RADIUS, -_POINT_RADIUS,
                     2 * _POINT_RADIUS, 2 * _POINT_RADIUS)
        self.setBrush(self._base_brush)
        self.setPen(self._base_pen)
        self.setZValue(0)
        scene = self.scene()
        if scene is not None:
            scene.point_unhovered.emit()  # type: ignore[attr-defined]
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        scene = self.scene()
        if scene is not None:
            scene.point_double_clicked.emit(self.image_path)  # type: ignore[attr-defined]
        super().mouseDoubleClickEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Scene — owns the points, fans out hover/click signals
# ─────────────────────────────────────────────────────────────────────────────


class ClusterScene(QGraphicsScene):
    """Scene that hosts the scatter points and forwards hover/click events."""

    point_hovered = pyqtSignal(str)
    point_unhovered = pyqtSignal()
    point_double_clicked = pyqtSignal(str)


# ─────────────────────────────────────────────────────────────────────────────
# View — pannable, zoomable
# ─────────────────────────────────────────────────────────────────────────────


class ClusterMapView(QGraphicsView):
    """QGraphicsView with mouse-wheel zoom + middle-button pan + auto-fit."""

    def __init__(self, scene: ClusterScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(
            f"QGraphicsView {{ background: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 12px; }}"
        )

    def wheelEvent(self, event: QWheelEvent):
        # Zoom centered on mouse pointer.
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit_to_scene(self):
        rect = self.scene().itemsBoundingRect()
        if rect.isEmpty():
            return
        rect.adjust(-_VIEW_PADDING, -_VIEW_PADDING, _VIEW_PADDING, _VIEW_PADDING)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)


# ─────────────────────────────────────────────────────────────────────────────
# Tab — top-level widget
# ─────────────────────────────────────────────────────────────────────────────


class ClusterMapTab(QWidget):
    """Main tab widget for the Latent Space Visualizer.

    Public API:
        set_image_paths(paths): seed the dataset to visualize
        compute(): start the embedding+reduction worker
        navigate_to_image(path): emitted on double-click of a point
    """

    navigate_to_image = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_paths: list[str] = []
        self._coords: Optional[np.ndarray] = None
        self._mapped_paths: list[str] = []
        self._worker: Optional[EmbeddingWorker] = None
        self._thumb_cache: dict[str, QPixmap] = {}
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Header / explainer
        title = QLabel("Latent Space Visualizer")
        title.setStyleSheet(section_header_style())
        root.addWidget(title)

        subtitle = QLabel(
            "Project your dataset onto a 2D map of CLIP image embeddings to "
            "spot concept gaps, dense clusters and outliers. Hover a point "
            "for a preview, double-click to jump to that image."
        )
        subtitle.setStyleSheet(muted_label_style())
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._reducer_combo = QComboBox()
        self._reducer_combo.addItem("UMAP (recommended)", "umap")
        self._reducer_combo.addItem("t-SNE", "tsne")
        self._reducer_combo.setFixedWidth(180)
        self._reducer_combo.setToolTip(
            "UMAP is faster and preserves global structure better; "
            "t-SNE focuses on local neighborhoods."
        )
        toolbar.addWidget(QLabel("Reducer:"))
        toolbar.addWidget(self._reducer_combo)

        self._btn_compute = QPushButton("Compute Map")
        self._btn_compute.setStyleSheet(accent_button_style())
        self._btn_compute.setToolTip(
            "Embed every image with CLIP and project to 2D. "
            "Progress shown below; the model loads on first run (~150 MB)."
        )
        self._btn_compute.clicked.connect(self.compute)
        toolbar.addWidget(self._btn_compute)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setStyleSheet(nav_button_style())
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        toolbar.addWidget(self._btn_cancel)

        toolbar.addStretch()

        self._stats_label = QLabel("No data — load a dataset and click Compute Map.")
        self._stats_label.setStyleSheet(muted_label_style())
        toolbar.addWidget(self._stats_label)

        root.addLayout(toolbar)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # Main area — scatter (left) + thumbnail preview (right)
        body = QHBoxLayout()
        body.setSpacing(12)

        self._scene = ClusterScene()
        self._scene.point_hovered.connect(self._on_point_hovered)
        self._scene.point_unhovered.connect(self._on_point_unhovered)
        self._scene.point_double_clicked.connect(self.navigate_to_image.emit)

        self._view = ClusterMapView(self._scene)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body.addWidget(self._view, stretch=3)

        # Right-side preview pane
        side = QVBoxLayout()
        side.setSpacing(6)
        side_title = QLabel("Preview")
        side_title.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; "
            f"font-weight: 700; background: transparent; "
            f"text-transform: uppercase; letter-spacing: 1px;"
        )
        side.addWidget(side_title)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setStyleSheet(
            f"background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 12px; color: {COLORS['text_muted']};"
        )
        self._thumb_label.setText("Hover a point\nto preview")
        side.addWidget(self._thumb_label)

        self._thumb_path_label = QLabel("")
        self._thumb_path_label.setStyleSheet(muted_label_style())
        self._thumb_path_label.setWordWrap(True)
        self._thumb_path_label.setMaximumWidth(_THUMBNAIL_SIZE)
        side.addWidget(self._thumb_path_label)
        side.addStretch()

        body.addLayout(side, stretch=0)
        root.addLayout(body, stretch=1)

    # ── Public API ──────────────────────────────────────────────────────

    def set_image_paths(self, paths: list[str]):
        """Update the list of images to visualize. Does not auto-compute."""
        self._image_paths = list(paths)
        n = len(paths)
        if n == 0:
            self._stats_label.setText("No images in current dataset.")
        else:
            self._stats_label.setText(
                f"{n} image(s) ready — click Compute Map to project them."
            )

    def compute(self):
        """Start the embedding+reduction worker on the current image set."""
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._image_paths:
            self._stats_label.setText("No images to compute — load a dataset first.")
            return

        self._scene.clear()
        self._coords = None
        self._mapped_paths = []
        self._thumb_cache.clear()

        reducer = self._reducer_combo.currentData() or "umap"
        self._worker = EmbeddingWorker(self._image_paths, reducer=reducer)
        self._worker.progress.connect(self._on_progress)
        self._worker.coords_ready.connect(self._on_coords_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)

        self._progress.setRange(0, len(self._image_paths))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._btn_compute.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._stats_label.setText("Loading CLIP model...")

        self._worker.start()

    # ── Worker callbacks ────────────────────────────────────────────────

    def _on_progress(self, done: int, total: int, msg: str):
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(done)
        self._stats_label.setText(msg)

    def _on_coords_ready(self, coords: np.ndarray, paths: list[str]):
        self._coords = coords
        self._mapped_paths = paths
        self._render_points(coords, paths)
        self._stats_label.setText(
            f"Mapped {len(paths)} images. Hover to preview, "
            f"double-click to open."
        )

    def _on_error(self, msg: str):
        self._stats_label.setText(f"Error: {msg}")
        log.error("ClusterMapTab worker error: %s", msg)

    def _on_worker_finished(self):
        self._progress.setVisible(False)
        self._btn_compute.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._stats_label.setText("Cancelling — finishing current batch...")

    # ── Rendering ───────────────────────────────────────────────────────

    def _render_points(self, coords: np.ndarray, paths: list[str]):
        """Place ellipse items in the scene, scaled to fit a sensible canvas."""
        self._scene.clear()
        if coords.size == 0:
            return

        # Normalize coords to [0, 1000] x [0, 1000] so we can pick a fixed
        # point radius without worrying about reducer output scale.
        mins = coords.min(axis=0)
        spans = coords.max(axis=0) - mins
        spans[spans == 0] = 1.0
        norm = (coords - mins) / spans
        norm *= 1000.0

        accent = QColor(COLORS["accent"])
        accent.setAlphaF(0.85)

        for (x, y), path in zip(norm, paths):
            point = _PointItem(float(x), float(y), path, accent)
            self._scene.addItem(point)

        self._view.fit_to_scene()
        # Schedule another fit after the layout settles to avoid jitter.
        QTimer.singleShot(50, self._view.fit_to_scene)

    # ── Hover preview ───────────────────────────────────────────────────

    def _on_point_hovered(self, image_path: str):
        pix = self._thumb_cache.get(image_path)
        if pix is None:
            pix = self._load_thumbnail(image_path)
            if pix is not None:
                self._thumb_cache[image_path] = pix
        if pix is not None:
            self._thumb_label.setPixmap(pix)
        else:
            self._thumb_label.setText("(preview unavailable)")
        self._thumb_path_label.setText(Path(image_path).name)

    def _on_point_unhovered(self):
        # Keep the last-hovered preview visible — feels less flickery than
        # blanking it; only blank if cache is empty.
        if not self._thumb_cache:
            self._thumb_label.setText("Hover a point\nto preview")
            self._thumb_path_label.setText("")

    def _load_thumbnail(self, path: str) -> Optional[QPixmap]:
        """Load a PIL image, downscale to thumbnail size, convert to QPixmap."""
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(
                (_THUMBNAIL_SIZE - 8, _THUMBNAIL_SIZE - 8),
                Image.Resampling.LANCZOS,
            )
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, 3 * img.width,
                          QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load thumbnail %s: %s", path, e)
            return None

    def refresh_theme(self):
        """Re-apply inline styles after a theme change."""
        self._stats_label.setStyleSheet(muted_label_style())
        self._thumb_label.setStyleSheet(
            f"background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 12px; color: {COLORS['text_muted']};"
        )
        self._thumb_path_label.setStyleSheet(muted_label_style())
        self._btn_compute.setStyleSheet(accent_button_style())
        self._btn_cancel.setStyleSheet(nav_button_style())
        self._view.setStyleSheet(
            f"QGraphicsView {{ background: {COLORS['bg_alt']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 12px; }}"
        )
