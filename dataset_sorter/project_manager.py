"""Project management for DataBuilder.

A *project* is a self-contained directory that groups dataset, training
configuration, output checkpoints, generated samples, and run history for a
single LoRA / finetune experiment.

Default layout::

    ~/DataBuilder-Projects/
    └── my_lora/
        ├── project.json     # rich project metadata (extends workers.py minimal version)
        ├── dataset/         # images + captions (compatible with existing export worker)
        ├── models/          # final trained weights
        ├── samples/         # samples generated during training
        ├── checkpoints/     # step/epoch checkpoints
        ├── backups/         # full project backups
        ├── logs/            # TensorBoard logs
        ├── .cache/          # latent / text-encoder caches
        ├── presets/         # saved TrainingConfig JSON files
        └── history/         # per-run JSON records

The project directory is intentionally compatible with the existing trainer:
``project.path`` can be passed directly as ``output_dir`` to ``TrainingWorker``.

Usage::

    pm = ProjectManager()
    proj = pm.create_project("my_lora", base_model="black-forest-labs/FLUX.1-dev", architecture="flux")
    pm.list_projects()         # -> [Project(...), ...]
    proj = pm.load_project("my_lora")
    proj.save()                # writes project.json
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dataset_sorter.app_settings import AppSettings
from dataset_sorter.workers import PROJECT_SUBDIRS, create_project_structure

log = logging.getLogger(__name__)

_PROJECT_JSON = "project.json"
_SCHEMA_VERSION = 2          # bump when breaking changes land
_PRESET_EXT = ".json"
_HISTORY_EXT = ".json"

# Additional subdirs beyond what workers.py already creates
_EXTRA_SUBDIRS = ["presets", "history"]


# ─────────────────────────────────────────────────────────────────────────────
# Project dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Project:
    """Represents a single DataBuilder project on disk.

    All mutable fields are updated in-memory; call :meth:`save` to persist.
    """

    name: str
    path: Path

    # Training setup
    base_model: str = ""          # HuggingFace model ID or local path
    architecture: str = ""        # e.g. "sd15", "sdxl", "flux"

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_trained: Optional[datetime] = None

    # Internal
    _schema_version: int = field(default=_SCHEMA_VERSION, repr=False)

    # ── Derived paths ──────────────────────────────────────────────────

    @property
    def dataset_path(self) -> Path:
        """Path to the dataset subdirectory."""
        return self.path / "dataset"

    @property
    def output_path(self) -> Path:
        """Checkpoints output path (compatible with TrainingWorker output_dir)."""
        return self.path

    @property
    def checkpoints_path(self) -> Path:
        return self.path / "checkpoints"

    @property
    def samples_path(self) -> Path:
        return self.path / "samples"

    @property
    def logs_path(self) -> Path:
        return self.path / "logs"

    @property
    def models_path(self) -> Path:
        return self.path / "models"

    @property
    def presets_path(self) -> Path:
        return self.path / "presets"

    @property
    def history_path(self) -> Path:
        return self.path / "history"

    # ── Checkpoint helpers ────────────────────────────────────────────

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Return the most recently modified checkpoint directory, or None."""
        ckpt_dir = self.checkpoints_path
        if not ckpt_dir.exists():
            return None
        candidates = [p for p in ckpt_dir.iterdir() if p.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # ── Sample helpers ────────────────────────────────────────────────

    def get_samples(self) -> list[Path]:
        """Return sorted list of generated sample images."""
        samples_dir = self.samples_path
        if not samples_dir.exists():
            return []
        return sorted(
            p for p in samples_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

    # ── Preset helpers ────────────────────────────────────────────────

    def save_preset(self, name: str, config: dict[str, Any]) -> Path:
        """Save a TrainingConfig dict as a named preset JSON."""
        self.presets_path.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\-.]", "_", name)
        preset_path = self.presets_path / f"{safe_name}{_PRESET_EXT}"
        preset_path.write_text(
            json.dumps({"name": name, "config": config}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.debug("Saved preset '%s' to %s", name, preset_path)
        return preset_path

    def list_presets(self) -> list[str]:
        """Return names of saved presets."""
        if not self.presets_path.exists():
            return []
        return [p.stem for p in sorted(self.presets_path.glob(f"*{_PRESET_EXT}"))]

    def load_preset(self, name: str) -> dict[str, Any]:
        """Load a saved preset by name. Returns the config dict."""
        safe_name = re.sub(r"[^\w\-.]", "_", name)
        preset_path = self.presets_path / f"{safe_name}{_PRESET_EXT}"
        raw = json.loads(preset_path.read_text(encoding="utf-8"))
        return raw.get("config", raw)

    # ── History helpers ───────────────────────────────────────────────

    def record_training_run(self, run_info: dict[str, Any]) -> Path:
        """Append a training-run record to the history directory."""
        self.history_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        record_path = self.history_path / f"run_{ts}{_HISTORY_EXT}"
        run_info.setdefault("timestamp", datetime.now().isoformat())
        record_path.write_text(
            json.dumps(run_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.last_trained = datetime.now()
        self.save()
        return record_path

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return all training-run records, sorted oldest first."""
        if not self.history_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for p in sorted(self.history_path.glob(f"*{_HISTORY_EXT}")):
            try:
                records.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                log.debug("Skipping bad history file %s: %s", p, exc)
        return records

    # ── Persistence ───────────────────────────────────────────────────

    def save(self) -> None:
        """Write project.json to disk."""
        data = {
            "version": _SCHEMA_VERSION,
            "name": self.name,
            "base_model": self.base_model,
            "architecture": self.architecture,
            "created": self.created_at.isoformat(),
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
        }
        path = self.path / _PROJECT_JSON
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.error("Could not save project.json for '%s': %s", self.name, exc)

    @classmethod
    def _from_dict(cls, path: Path, data: dict[str, Any]) -> "Project":
        proj = cls(name=data.get("name", path.name), path=path)
        proj.base_model = data.get("base_model", "")
        proj.architecture = data.get("architecture", "")
        proj._schema_version = data.get("version", 1)
        for attr, key in (("created_at", "created"), ("last_trained", "last_trained")):
            raw = data.get(key)
            if raw:
                try:
                    setattr(proj, attr, datetime.fromisoformat(raw))
                except (ValueError, TypeError):
                    pass
        return proj


# ─────────────────────────────────────────────────────────────────────────────
# ProjectManager
# ─────────────────────────────────────────────────────────────────────────────

class ProjectManager:
    """Manages the collection of DataBuilder projects.

    Projects live under :attr:`projects_root` (defaults to
    ``~/DataBuilder-Projects``).  The root can be changed permanently via
    :meth:`set_projects_root` (persists to :class:`AppSettings`) or
    temporarily by passing a ``projects_root`` argument to the constructor.
    """

    def __init__(self, projects_root: Optional[Path] = None) -> None:
        self._settings: Optional[AppSettings] = None
        if projects_root is not None:
            self._root = Path(projects_root)
        else:
            self._settings = AppSettings.load()
            self._root = self._settings.projects_root

    # ── Root management ───────────────────────────────────────────────

    def get_projects_root(self) -> Path:
        return self._root

    def set_projects_root(self, path: Path) -> None:
        """Change the projects root and persist to AppSettings."""
        self._root = Path(path)
        settings = self._settings or AppSettings.load()
        settings.projects_root = self._root
        settings.save()
        self._settings = settings

    # ── Project CRUD ──────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        base_model: str = "",
        architecture: str = "",
        dataset_path: Optional[Path] = None,
    ) -> Project:
        """Create a new project directory and return a :class:`Project`.

        Raises ``FileExistsError`` if a project with *name* already exists.
        """
        safe_name = _safe_dir_name(name)
        project_path = self._root / safe_name

        if project_path.exists() and (project_path / _PROJECT_JSON).exists():
            raise FileExistsError(f"Project '{name}' already exists at {project_path}")

        # Create base structure (compatible with existing workers.py)
        create_project_structure(project_path)
        # Additional dirs not in workers.py
        for sub in _EXTRA_SUBDIRS:
            (project_path / sub).mkdir(exist_ok=True)

        proj = Project(
            name=name,
            path=project_path,
            base_model=base_model,
            architecture=architecture,
        )
        proj.save()

        # Update recent projects
        settings = self._settings or AppSettings.load()
        settings.add_recent_project(name)
        settings.save()
        self._settings = settings

        log.info("Created project '%s' at %s", name, project_path)
        return proj

    def load_project(self, name: str) -> Project:
        """Load an existing project by name.

        Raises ``FileNotFoundError`` if the project does not exist.
        """
        safe_name = _safe_dir_name(name)
        project_path = self._root / safe_name
        json_path = project_path / _PROJECT_JSON

        if not json_path.exists():
            raise FileNotFoundError(f"No project.json found at {json_path}")

        try:
            data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(
                f"Could not read project.json at {json_path}: {exc}"
            ) from exc

        proj = Project._from_dict(project_path, data)

        # Update recent projects
        settings = self._settings or AppSettings.load()
        settings.add_recent_project(name)
        settings.save()
        self._settings = settings

        return proj

    def list_projects(self) -> list[Project]:
        """Return all valid projects under :attr:`projects_root`, sorted by name."""
        if not self._root.exists():
            return []

        projects: list[Project] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            json_path = entry / _PROJECT_JSON
            if not json_path.exists():
                continue
            try:
                data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
                # Only consider v2+ projects (with "name" field)
                if data.get("version", 1) >= _SCHEMA_VERSION or "base_model" in data:
                    projects.append(Project._from_dict(entry, data))
            except (OSError, json.JSONDecodeError) as exc:
                log.debug("Skipping %s: %s", entry, exc)

        return projects

    def delete_project(self, name: str, *, confirm: bool = True) -> None:
        """Delete a project directory.

        Pass ``confirm=False`` to skip the guard (useful in tests).
        Raises ``FileNotFoundError`` if the project doesn't exist.
        """
        safe_name = _safe_dir_name(name)
        project_path = self._root / safe_name

        if not (project_path / _PROJECT_JSON).exists():
            raise FileNotFoundError(f"Project '{name}' not found at {project_path}")

        if confirm:
            # Caller should prompt the user before reaching here
            raise RuntimeError(
                "Pass confirm=False explicitly after user has confirmed deletion."
            )

        import shutil
        shutil.rmtree(project_path)

        settings = self._settings or AppSettings.load()
        settings.remove_recent_project(name)
        settings.save()
        self._settings = settings

        log.info("Deleted project '%s'", name)

    def project_exists(self, name: str) -> bool:
        safe_name = _safe_dir_name(name)
        return (self._root / safe_name / _PROJECT_JSON).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_dir_name(name: str) -> str:
    """Convert a project name to a safe directory name."""
    safe = re.sub(r"[^\w\-. ]", "_", name).strip().replace(" ", "_")
    return safe or "unnamed_project"
