"""QThread workers for long-running operations (scan, export).

Supports parallel scanning with configurable worker count and
optional GPU-accelerated image validation.
"""

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


def _parse_single_image(args: tuple) -> ImageEntry:
    """Parse a single image+txt pair. Runs in a worker thread."""
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

    # GPU validation: verify image can be decoded (optional, catches corrupt files)
    if use_gpu:
        try:
            import torch
            from torchvision.io import read_image, ImageReadMode
            read_image(str(img_path), mode=ImageReadMode.RGB)
        except Exception:
            pass  # Still include — just can't GPU-validate

    unique_id = f"{i:06d}_{uuid.uuid4().hex[:8]}"
    return ImageEntry(
        image_path=img_path,
        txt_path=txt_found,
        tags=tags,
        assigned_bucket=1,
        unique_id=unique_id,
    )


class ScanWorker(QThread):
    """Scans source directory for images and txt files.

    Uses ThreadPoolExecutor for parallel tag file reading.
    """

    progress = pyqtSignal(int, int)       # current, total
    finished_scan = pyqtSignal(list)       # list[ImageEntry]
    status = pyqtSignal(str)

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

    def run(self):
        self.status.emit("Discovering images...")
        image_files: list[Path] = []

        for root, _dirs, files in os.walk(self.source_dir):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(Path(root) / f)

        image_files.sort()
        total = len(image_files)
        self.status.emit(f"{total} images found, reading tags ({self.num_workers} workers)...")

        if total == 0:
            self.finished_scan.emit([])
            return

        # Parallel parsing with ThreadPoolExecutor
        entries: list[ImageEntry] = [None] * total  # type: ignore[list-item]
        args_list = [(i, p, self.use_gpu) for i, p in enumerate(image_files)]
        completed = 0

        if self.num_workers <= 1:
            # Single-threaded fallback
            for args in args_list:
                entry = _parse_single_image(args)
                entries[args[0]] = entry
                completed += 1
                if completed % 50 == 0 or completed == total:
                    self.progress.emit(completed, total)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(_parse_single_image, args): args[0]
                    for args in args_list
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        entries[idx] = future.result()
                    except Exception:
                        # Fallback entry on error
                        entries[idx] = ImageEntry(
                            image_path=image_files[idx],
                            unique_id=f"{idx:06d}_{uuid.uuid4().hex[:8]}",
                        )
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        self.progress.emit(completed, total)

        self.finished_scan.emit(entries)


class ExportWorker(QThread):
    """Copies images and filtered txt files to output directory."""

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
        num_workers: int = DEFAULT_NUM_WORKERS,
        parent=None,
    ):
        super().__init__(parent)
        self.entries = entries
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)
        self.bucket_names = bucket_names
        self.deleted_tags = deleted_tags
        self.num_workers = max(1, num_workers)

    def run(self):
        # Security checks
        if is_path_inside(self.output_dir, self.source_dir):
            self.finished_export.emit(0, -1)
            return
        if is_path_inside(self.source_dir, self.output_dir):
            self.finished_export.emit(0, -2)
            return

        self.status.emit("Preparing export...")

        copied = 0
        errors = 0
        error_log: list[str] = []

        # Group by bucket
        bucket_entries: dict[int, list[ImageEntry]] = defaultdict(list)
        for entry in self.entries:
            bucket_entries[entry.assigned_bucket].append(entry)

        total = len(self.entries)

        for bucket_num, b_entries in sorted(bucket_entries.items()):
            if not b_entries:
                continue

            bname = sanitize_folder_name(self.bucket_names.get(bucket_num, "bucket"))
            folder_name = f"{bucket_num}_{bname}"
            folder_path = self.output_dir / folder_name

            if not is_path_inside(folder_path, self.output_dir):
                errors += len(b_entries)
                continue

            folder_path.mkdir(parents=True, exist_ok=True)
            self.status.emit(f"Exporting bucket {bucket_num} ({len(b_entries)} images)...")

            for entry in b_entries:
                try:
                    dest_img = folder_path / entry.image_path.name
                    if not is_path_inside(dest_img, self.output_dir):
                        errors += 1
                        continue
                    shutil.copy2(str(entry.image_path), str(dest_img))

                    if entry.txt_path is not None:
                        dest_txt = folder_path / entry.txt_path.name
                        if is_path_inside(dest_txt, self.output_dir):
                            if self.deleted_tags:
                                filtered = [
                                    t for t in entry.tags
                                    if t not in self.deleted_tags
                                ]
                                dest_txt.write_text(
                                    ", ".join(filtered), encoding="utf-8",
                                )
                            else:
                                shutil.copy2(str(entry.txt_path), str(dest_txt))

                    copied += 1
                except Exception as exc:
                    errors += 1
                    error_log.append(f"{entry.image_path}: {exc}")

                self.progress.emit(copied + errors, total)

        if error_log:
            try:
                (self.output_dir / "export_errors.log").write_text(
                    "\n".join(error_log), encoding="utf-8",
                )
            except OSError:
                pass

        self.finished_export.emit(copied, errors)
