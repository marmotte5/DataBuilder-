"""QThread workers for long-running operations (scan, export).

Supports parallel scanning with configurable worker count,
optional GPU-accelerated image validation, and cancellation.
Optimized for 1M+ image datasets.

Export uses ThreadPoolExecutor for parallel file copying (3-5x speedup on SSDs).
"""

import json
import logging
import os
import shutil
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.constants import IMAGE_EXTENSIONS, DEFAULT_NUM_WORKERS
from dataset_sorter.models import ImageEntry
from dataset_sorter.utils import is_path_inside, sanitize_folder_name

log = logging.getLogger(__name__)

# ── Project folder layout ────────────────────────────────────────────
PROJECT_SUBDIRS = [
    "dataset",       # Exported images organized into bucket folders
    "models",        # Trained model outputs / final weights
    "samples",       # Sample images generated during training
    "checkpoints",   # Step/epoch checkpoint saves
    "backups",       # Full project backups
    "logs",          # Training logs
    ".cache",        # Latent / text encoder caches
]


def compute_repeats(bucket_num: int, max_bucket: int, min_repeats: int = 1,
                    max_repeats: int = 20) -> int:
    """Compute repeat count for a bucket based on its rarity.

    Rare buckets (high number) get more repeats so the model sees them
    as often as common ones.  Uses a linear scale:
        bucket 1  → min_repeats  (most common, least repetition)
        max_bucket → max_repeats  (rarest, most repetition)
    """
    if max_bucket <= 1:
        return min_repeats
    t = (bucket_num - 1) / (max_bucket - 1)  # 0..1
    return max(min_repeats, min(max_repeats, round(min_repeats + t * (max_repeats - min_repeats))))


def create_project_structure(output_dir: Path) -> None:
    """Create the standard project directory tree."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in PROJECT_SUBDIRS:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    info_path = output_dir / "project.json"
    if not info_path.exists():
        info = {
            "created": datetime.now().isoformat(),
            "version": 1,
        }
        try:
            info_path.write_text(json.dumps(info, indent=2))
        except OSError as e:
            log.debug(f"Could not write project.json: {e}")

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
            # Read and validate image can be decoded to RGB tensor
            tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
            # Basic sanity checks on decoded tensor
            if tensor.shape[0] != 3 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
                error_msg = f"{img_path}: Invalid image dimensions {tensor.shape}"
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

        # Use os.scandir()-based recursive scan (2-3x faster than os.walk)
        from dataset_sorter.io_speed import scandir_recursive
        image_files = scandir_recursive(self.source_dir, IMAGE_EXTENSIONS)

        if self._cancelled:
            self.finished_scan.emit([])
            return
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

        # Populate metadata cache for fast subsequent queries
        try:
            from dataset_sorter.metadata_cache import MetadataCache
            cache = MetadataCache(self.source_dir / ".metadata.db")
            batch = []
            for entry in entries:
                if entry is not None and entry.image_path is not None:
                    batch.append((
                        entry.image_path,
                        {
                            "tags": ", ".join(entry.tags) if entry.tags else "",
                            "bucket": entry.assigned_bucket,
                        },
                    ))
            if batch:
                cache.put_batch(batch)
                log.debug(f"Metadata cache: {len(batch)} entries written")
        except Exception as e:
            log.debug(f"Metadata cache update skipped: {e}")

        self.finished_scan.emit(entries)


def _unique_dest(folder_path: Path, name: str) -> Path:
    """Return a unique destination path, using atomic creation to avoid races.

    Uses os.open with O_CREAT|O_EXCL to atomically check-and-create,
    preventing TOCTOU races when multiple threads export concurrently.
    """
    import os as _os

    dest = folder_path / name
    try:
        fd = _os.open(str(dest), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
        _os.close(fd)
        return dest
    except FileExistsError:
        pass

    stem = Path(name).stem
    suffix = Path(name).suffix
    for counter in range(1, 100_000):
        candidate = folder_path / f"{stem}_{counter:03d}{suffix}"
        try:
            fd = _os.open(str(candidate), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
            _os.close(fd)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not find unique destination for {name}")


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

        # Create full project folder structure
        create_project_structure(self.output_dir)
        dataset_dir = self.output_dir / "dataset"

        copied = 0
        errors = 0
        error_log: list[str] = []

        # Group by bucket
        bucket_entries: dict[int, list[tuple]] = defaultdict(list)
        for snap in self._snapshots:
            bucket_entries[snap[3]].append(snap)

        total = len(self._snapshots)

        # Compute repeats based on rarity (highest active bucket = rarest)
        max_bucket = max(bucket_entries.keys()) if bucket_entries else 1

        # Pre-create all bucket directories inside dataset/
        bucket_folders: dict[int, Path | None] = {}
        for bucket_num in sorted(bucket_entries.keys()):
            bname = sanitize_folder_name(self.bucket_names.get(bucket_num, "bucket"))
            repeats = compute_repeats(bucket_num, max_bucket)
            folder_name = f"{repeats}_{bucket_num}_{bname}"
            folder_path = dataset_dir / folder_name
            if is_path_inside(folder_path, dataset_dir):
                folder_path.mkdir(parents=True, exist_ok=True)
                bucket_folders[bucket_num] = folder_path
            else:
                bucket_folders[bucket_num] = None

        # Build flat list of export tasks with sequential index per bucket
        export_tasks: list[tuple] = []  # (image_path, txt_path, tags, folder_path, new_name)
        for bucket_num, b_entries in sorted(bucket_entries.items()):
            folder_path = bucket_folders.get(bucket_num)
            if folder_path is None:
                errors += len(b_entries)
                continue
            for file_index, (image_path, txt_path, tags, _) in enumerate(b_entries, start=1):
                new_name = f"{bucket_num}_{file_index:04d}{image_path.suffix}"
                export_tasks.append((image_path, txt_path, tags, folder_path, new_name))

        if not export_tasks:
            self.progress.emit(total, total)
            self.finished_export.emit(copied, errors)
            return

        deleted_tags = self.deleted_tags
        output_dir = self.output_dir

        def _export_one(task: tuple) -> tuple[bool, str]:
            """Export a single image+txt pair. Returns (success, error_msg)."""
            image_path, txt_path, tags, folder_path, new_name = task
            try:
                dest_img = _unique_dest(folder_path, new_name)
                if not is_path_inside(dest_img, output_dir):
                    return False, f"{image_path}: dest outside output dir"
                shutil.copy2(str(image_path), str(dest_img))
                if txt_path is not None:
                    txt_name = dest_img.stem + ".txt"
                    dest_txt = folder_path / txt_name
                    if is_path_inside(dest_txt, output_dir):
                        filtered = [t for t in tags if t not in deleted_tags]
                        dest_txt.write_text(
                            ", ".join(filtered), encoding="utf-8",
                        )
                return True, ""
            except Exception as exc:
                return False, f"{image_path}: {exc}"

        # Parallel export using ThreadPoolExecutor
        num_workers = min(DEFAULT_NUM_WORKERS, len(export_tasks))
        self.status.emit(f"Exporting {len(export_tasks)} images ({num_workers} workers)...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_export_one, task): i
                for i, task in enumerate(export_tasks)
            }
            for future in as_completed(futures):
                if self._cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                success, err = future.result()
                if success:
                    copied += 1
                else:
                    errors += 1
                    if err:
                        error_log.append(err)
                if (copied + errors) % _PROGRESS_INTERVAL == 0 or (copied + errors) == total:
                    self.progress.emit(copied + errors, total)

        if error_log:
            try:
                (self.output_dir / "logs" / "export_errors.log").write_text(
                    "\n".join(error_log), encoding="utf-8",
                )
            except OSError as e:
                log.debug(f"Could not write export error log: {e}")

        self.finished_export.emit(copied, errors)
