"""Workers QThread pour les opérations longues (scan, export)."""

import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from dataset_sorter.constants import IMAGE_EXTENSIONS
from dataset_sorter.models import ImageEntry
from dataset_sorter.utils import is_path_inside, sanitize_folder_name


class ScanWorker(QThread):
    """Scanne le répertoire source à la recherche d'images et de fichiers txt."""

    progress = pyqtSignal(int, int)       # current, total
    finished_scan = pyqtSignal(list)       # list[ImageEntry]
    status = pyqtSignal(str)               # message de statut

    def __init__(self, source_dir: str, parent=None):
        super().__init__(parent)
        self.source_dir = Path(source_dir)

    def run(self):
        self.status.emit("Recherche des images...")
        image_files: list[Path] = []

        for root, _dirs, files in os.walk(self.source_dir):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(Path(root) / f)

        total = len(image_files)
        self.status.emit(f"{total} images trouvées, lecture des tags...")

        entries: list[ImageEntry] = []
        for i, img_path in enumerate(sorted(image_files)):
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

            unique_id = f"{i:06d}_{uuid.uuid4().hex[:8]}"
            entries.append(ImageEntry(
                image_path=img_path,
                txt_path=txt_found,
                tags=tags,
                assigned_bucket=1,
                unique_id=unique_id,
            ))

            if (i + 1) % 50 == 0 or i + 1 == total:
                self.progress.emit(i + 1, total)

        self.finished_scan.emit(entries)


class ExportWorker(QThread):
    """Copie les images et fichiers txt filtrés vers le dossier de sortie."""

    progress = pyqtSignal(int, int)               # current, total
    finished_export = pyqtSignal(int, int)         # copied, errors
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
        self.entries = entries
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)
        self.bucket_names = bucket_names
        self.deleted_tags = deleted_tags

    def run(self):
        # Vérifications de sécurité
        if is_path_inside(self.output_dir, self.source_dir):
            self.finished_export.emit(0, -1)
            return
        if is_path_inside(self.source_dir, self.output_dir):
            self.finished_export.emit(0, -2)
            return

        self.status.emit("Préparation de l'export...")

        copied = 0
        errors = 0
        error_log: list[str] = []

        # Regrouper par bucket
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
            self.status.emit(f"Export bucket {bucket_num} ({len(b_entries)} images)...")

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
