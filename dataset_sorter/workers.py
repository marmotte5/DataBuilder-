"""QThread workers for long-running operations (scan, export).

Supports parallel scanning with configurable worker count,
optional GPU-accelerated image validation, and cancellation.
Optimized for 1M+ image datasets.
"""

import logging
import os
import shutil
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.constants import IMAGE_EXTENSIONS, DEFAULT_NUM_WORKERS
from dataset_sorter.models import ImageEntry
from dataset_sorter.utils import is_path_inside, sanitize_folder_name

log = logging.getLogger(__name__)

# Chunk size for submitting futures (avoids 1M-entry dict)
_SCAN_CHUNK = 5000
# Progress emit interval
_PROGRESS_INTERVAL = 200


def _parse_single_image(args: tuple) -> ImageEntry | tuple[ImageEntry, str]:
    """Parse a single image+txt pair. Runs in a worker thread.

    Returns ImageEntry on success, or (ImageEntry, error_msg) if GPU
    validation fails (so the caller can log it).
    """
    i, img_path, use_gpu = args

    txt_path = img_path.with_suffix(".txt")
    tags: list[str] = []
    txt_found = None

    if txt_path.exists():
        txt_found = txt_path
        try:
            content = txt_path.read_text(encoding="utf-8", errors="replace")
            tags = [t.strip() for t in content.split(",") if t.strip()]
        except OSError:
            tags = []

    error_msg = None
    if use_gpu:
        try:
            import torch
            from torchvision.io import read_image, ImageReadMode
            read_image(str(img_path), mode=ImageReadMode.RGB)
        except Exception as e:
            error_msg = f"{img_path}: GPU validation failed: {e}"

    unique_id = f"{i:06d}_{uuid.uuid4().hex[:8]}"
    entry = ImageEntry(
        image_path=img_path,
        txt_path=txt_found,
        tags=tags,
        assigned_bucket=1,
        unique_id=unique_id,
    )
    if error_msg is not None:
        return entry, error_msg
    return entry


class ScanWorker(QThread):
    """Scans source directory for images and txt files.

    Uses ThreadPoolExecutor with chunked submission for memory efficiency.
    Supports cancellation via cancel(). Follows symlinks but detects loops.
    """

    progress = pyqtSignal(int, int)       # current, total
    finished_scan = pyqtSignal(list)       # list[ImageEntry]
    status = pyqtSignal(str)
    scan_errors = pyqtSignal(int)          # number of errors encountered

    def __init__(
        self,
        source_dir: str,
        num_workers: int = DEFAULT_NUM_WORKERS,
        use_gpu: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.source_dir = Path(source_dir)
        self.num_workers = max(1, num_workers)
        self.use_gpu = use_gpu
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _collect_result(self, result, idx, entries, error_log):
        """Unpack _parse_single_image result into entries and error_log."""
        if isinstance(result, tuple):
            entry, err = result
            entries[idx] = entry
            error_log.append(err)
        else:
            entries[idx] = result

    def run(self):
        self.status.emit("Discovering images...")
        image_files: list[Path] = []
        seen_dirs: set[str] = set()

        for root, dirs, files in os.walk(self.source_dir, followlinks=True):
            if self._cancelled:
                self.finished_scan.emit([])
                return
            # Detect symlink loops by tracking resolved directory identities
            real = os.path.realpath(root)
            if real in seen_dirs:
                log.warning(f"Symlink loop detected, skipping: {root}")
                dirs.clear()
                continue
            seen_dirs.add(real)

            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(Path(root) / f)

        image_files.sort()
        total = len(image_files)
        self.status.emit(f"{total} images found, reading tags ({self.num_workers} workers)...")

        if total == 0:
            self.finished_scan.emit([])
            return

        entries: list[ImageEntry] = [None] * total  # type: ignore[list-item]
        error_log: list[str] = []
        completed = 0

        if self.num_workers <= 1:
            for i, img_path in enumerate(image_files):
                if self._cancelled:
                    self.finished_scan.emit([])
                    return
                try:
                    result = _parse_single_image((i, img_path, self.use_gpu))
                    self._collect_result(result, i, entries, error_log)
                except Exception as e:
                    entries[i] = ImageEntry(
                        image_path=img_path,
                        unique_id=f"{i:06d}_{uuid.uuid4().hex[:8]}",
                    )
                    error_log.append(f"{img_path}: {e}")
                completed += 1
                if completed % _PROGRESS_INTERVAL == 0 or completed == total:
                    self.progress.emit(completed, total)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk_start in range(0, total, _SCAN_CHUNK):
                    if self._cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.finished_scan.emit([])
                        return

                    chunk_end = min(chunk_start + _SCAN_CHUNK, total)
                    futures = {}
                    for i in range(chunk_start, chunk_end):
                        fut = executor.submit(
                            _parse_single_image,
                            (i, image_files[i], self.use_gpu),
                        )
                        futures[fut] = i

                    for future in as_completed(futures):
                        if self._cancelled:
                            executor.shutdown(wait=False, cancel_futures=True)
                            self.finished_scan.emit([])
                            return

                        idx = futures[future]
                        try:
                            result = future.result()
                            self._collect_result(result, idx, entries, error_log)
                        except Exception as e:
                            entries[idx] = ImageEntry(
                                image_path=image_files[idx],
                                unique_id=f"{idx:06d}_{uuid.uuid4().hex[:8]}",
                            )
                            error_log.append(f"{image_files[idx]}: {e}")
                        completed += 1
                        if completed % _PROGRESS_INTERVAL == 0 or completed == total:
                            self.progress.emit(completed, total)

        if error_log:
            log.warning(f"Scan completed with {len(error_log)} error(s)")
            self.scan_errors.emit(len(error_log))

        self.finished_scan.emit(entries)


def _unique_dest(folder_path: Path, name: str) -> Path:
    """Return a unique destination path, appending _001 etc. on collision."""
    dest = folder_path / name
    if not dest.exists():
        return dest
    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 1
    while True:
        candidate = folder_path / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class ExportWorker(QThread):
    """Copies images and filtered txt files to output directory.

    Uses lightweight snapshots instead of deepcopy for memory efficiency.
    Supports cancellation via cancel().
    """

    progress = pyqtSignal(int, int)
    finished_export = pyqtSignal(int, int)  # copied, errors
    status = pyqtSignal(str)

    def __init__(
        self,
        entries: list[ImageEntry],
        output_dir: str,
        source_dir: str,
        bucket_names: dict[int, str],
        deleted_tags: set[str],
        parent=None,
    ):
        super().__init__(parent)
        # Lightweight snapshot: only copy the fields we need as tuples
        # Avoids deepcopy which is extremely slow at 1M entries
        self._snapshots: list[tuple] = [
            (e.image_path, e.txt_path, list(e.tags), e.assigned_bucket)
            for e in entries
        ]
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)
        self.bucket_names = dict(bucket_names)
        self.deleted_tags = set(deleted_tags)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if is_path_inside(self.output_dir, self.source_dir):
            self.finished_export.emit(0, -1)
            return
        if is_path_inside(self.source_dir, self.output_dir):
            self.finished_export.emit(0, -2)
            return

        # Check write permission on output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(str(self.output_dir), os.W_OK):
            log.error(f"No write permission on {self.output_dir}")
            self.finished_export.emit(0, -3)
            return

        self.status.emit("Preparing export...")

        copied = 0
        errors = 0
        error_log: list[str] = []

        # Group by bucket
        bucket_entries: dict[int, list[tuple]] = defaultdict(list)
        for snap in self._snapshots:
            bucket_entries[snap[3]].append(snap)

        total = len(self._snapshots)

        for bucket_num, b_entries in sorted(bucket_entries.items()):
            if self._cancelled:
                break
            if not b_entries:
                continue

            bname = sanitize_folder_name(self.bucket_names.get(bucket_num, "bucket"))
            folder_name = f"{bucket_num}_{bname}"
            folder_path = self.output_dir / folder_name

            if not is_path_inside(folder_path, self.output_dir):
                errors += len(b_entries)
                self.progress.emit(copied + errors, total)
                continue

            folder_path.mkdir(parents=True, exist_ok=True)
            self.status.emit(f"Exporting bucket {bucket_num} ({len(b_entries)} images)...")

            for image_path, txt_path, tags, _ in b_entries:
                if self._cancelled:
                    break
                try:
                    dest_img = _unique_dest(folder_path, image_path.name)
                    if not is_path_inside(dest_img, self.output_dir):
                        errors += 1
                    else:
                        shutil.copy2(str(image_path), str(dest_img))

                        if txt_path is not None:
                            txt_name = dest_img.stem + ".txt"
                            dest_txt = folder_path / txt_name
                            if is_path_inside(dest_txt, self.output_dir):
                                filtered = [
                                    t for t in tags
                                    if t not in self.deleted_tags
                                ]
                                dest_txt.write_text(
                                    ", ".join(filtered), encoding="utf-8",
                                )

                        copied += 1
                except Exception as exc:
                    errors += 1
                    error_log.append(f"{image_path}: {exc}")

                if (copied + errors) % _PROGRESS_INTERVAL == 0 or (copied + errors) == total:
                    self.progress.emit(copied + errors, total)

        if error_log:
            try:
                (self.output_dir / "export_errors.log").write_text(
                    "\n".join(error_log), encoding="utf-8",
                )
            except OSError:
                pass

        self.finished_export.emit(copied, errors)
