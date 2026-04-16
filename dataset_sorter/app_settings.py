"""
Module: app_settings.py
========================
Global application settings — persistent user preferences.

Role in DataBuilder:
    - Single entry point for persistent application configuration
    - Saves preferences (directories, recent projects, UI) across sessions
    - Used by ProjectManager, TrainingWorker and all UI tabs to read
      file paths and default settings

Classes/Fonctions principales:
    - AppSettings: Dataclass de configuration avec load()/save() automatique
      vers ~/.config/databuilder/settings.json (ou DATABUILDER_CONFIG_DIR)

Dependencies: stdlib only (json, pathlib, dataclasses, os)

Usage::

    settings = AppSettings.load()
    settings.projects_root = Path("~/MyProjects").expanduser()
    settings.save()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CONFIG_DIR_ENV = "DATABUILDER_CONFIG_DIR"
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "databuilder"
_SETTINGS_FILE = "settings.json"
_MAX_RECENT = 10


@dataclass
class AppSettings:
    """Persistent global settings for DataBuilder.

    Stored as JSON in ``~/.config/databuilder/settings.json``.
    Environment variable ``DATABUILDER_CONFIG_DIR`` overrides the directory.
    """

    # ── Directories ────────────────────────────────────────────────────
    projects_root: Path = field(
        default_factory=lambda: Path.home() / "DataBuilder-Projects"
    )
    models_cache: Path = field(
        default_factory=lambda: Path.home() / "DataBuilder-Models"
    )
    default_output_dir: Path = field(
        default_factory=lambda: Path.home() / "DataBuilder-Projects"
    )
    huggingface_cache: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        )
    )

    # ── Model scan directories ─────────────────────────────────────────
    # Scanned at startup for .safetensors / .ckpt files to populate model
    # path autocompleters in the training and generation tabs.
    model_scan_dirs: list[str] = field(default_factory=lambda: [
        str(Path.home() / "Models"),
        str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))),
    ])
    # LoRA output dirs to scan (added automatically when a training completes)
    lora_scan_dirs: list[str] = field(default_factory=list)

    # ── Recent projects ────────────────────────────────────────────────
    recent_projects: list[str] = field(default_factory=list)

    # ── UI preferences ─────────────────────────────────────────────────
    ui_preferences: dict[str, Any] = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────
    # Class-level helpers
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def get_config_dir(cls) -> Path:
        """Return the config directory, en respectant la variable d'environnement DATABUILDER_CONFIG_DIR."""
        env = os.environ.get(_CONFIG_DIR_ENV)
        return Path(env) if env else _DEFAULT_CONFIG_DIR

    @classmethod
    def get_settings_path(cls) -> Path:
        return cls.get_config_dir() / _SETTINGS_FILE

    # ─────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "AppSettings":
        """Load settings from disk; return defaults if file missing/invalid."""
        path = cls.get_settings_path()
        if not path.exists():
            log.debug("No settings file at %s — using defaults", path)
            return cls()

        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read settings (%s) — using defaults", exc)
            return cls()

        return cls._from_dict(raw)

    def save(self) -> None:
        """Persist settings to disk."""
        path = self.get_settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            log.error("Could not save settings: %s", exc)

    # ─────────────────────────────────────────────────────────────────
    # Recent projects helpers
    # ─────────────────────────────────────────────────────────────────

    def add_recent_project(self, name: str) -> None:
        """Push *name* to the front of recent_projects (max 10 entries)."""
        if name in self.recent_projects:
            self.recent_projects.remove(name)
        self.recent_projects.insert(0, name)
        self.recent_projects = self.recent_projects[:_MAX_RECENT]

    def remove_recent_project(self, name: str) -> None:
        if name in self.recent_projects:
            self.recent_projects.remove(name)

    # ─────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ─────────────────────────────────────────────────────────────────

    def _to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Path objects are not natively JSON-serializable — convert them to str
        for key in ("projects_root", "models_cache", "default_output_dir", "huggingface_cache"):
            if key in d and d[key] is not None:
                d[key] = str(d[key])
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "AppSettings":
        settings = cls()
        for key in ("projects_root", "models_cache", "default_output_dir", "huggingface_cache"):
            if key in d and d[key]:
                try:
                    setattr(settings, key, Path(d[key]))
                except (TypeError, ValueError) as exc:
                    log.debug("Could not parse path for %s: %s", key, exc)
        if "recent_projects" in d and isinstance(d["recent_projects"], list):
            settings.recent_projects = [str(p) for p in d["recent_projects"]][:_MAX_RECENT]
        if "ui_preferences" in d and isinstance(d["ui_preferences"], dict):
            settings.ui_preferences = d["ui_preferences"]
        if "model_scan_dirs" in d and isinstance(d["model_scan_dirs"], list):
            settings.model_scan_dirs = [str(p) for p in d["model_scan_dirs"]]
        if "lora_scan_dirs" in d and isinstance(d["lora_scan_dirs"], list):
            settings.lora_scan_dirs = [str(p) for p in d["lora_scan_dirs"]]
        return settings
