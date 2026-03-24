"""Tests for AppSettings and ProjectManager."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# AppSettings
# ─────────────────────────────────────────────────────────────────────────────

class TestAppSettings:
    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        s = AppSettings.load()
        assert s.projects_root == Path.home() / "DataBuilder-Projects"
        assert isinstance(s.recent_projects, list)
        assert isinstance(s.ui_preferences, dict)

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        s = AppSettings()
        s.projects_root = tmp_path / "projects"
        s.recent_projects = ["alpha", "beta"]
        s.ui_preferences = {"theme": "dark"}
        s.save()

        loaded = AppSettings.load()
        assert loaded.projects_root == tmp_path / "projects"
        assert loaded.recent_projects == ["alpha", "beta"]
        assert loaded.ui_preferences == {"theme": "dark"}

    def test_add_recent_project_deduplicates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        s = AppSettings()
        s.add_recent_project("foo")
        s.add_recent_project("bar")
        s.add_recent_project("foo")  # should move to front
        assert s.recent_projects[0] == "foo"
        assert s.recent_projects.count("foo") == 1

    def test_add_recent_project_max_10(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        s = AppSettings()
        for i in range(15):
            s.add_recent_project(f"project_{i}")
        assert len(s.recent_projects) == 10

    def test_remove_recent_project(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        s = AppSettings()
        s.recent_projects = ["a", "b", "c"]
        s.remove_recent_project("b")
        assert "b" not in s.recent_projects
        assert "a" in s.recent_projects

    def test_load_invalid_json_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("not valid json")
        s = AppSettings.load()
        assert isinstance(s.recent_projects, list)

    def test_get_config_dir_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path))
        from dataset_sorter.app_settings import AppSettings
        assert AppSettings.get_config_dir() == tmp_path

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "nested" / "config"
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(config_dir))
        from dataset_sorter.app_settings import AppSettings
        s = AppSettings()
        s.save()
        assert (config_dir / "settings.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# ProjectManager
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectManager:
    def _make_pm(self, tmp_path):
        """Return a ProjectManager rooted at tmp_path (no AppSettings I/O)."""
        from dataset_sorter.project_manager import ProjectManager
        return ProjectManager(projects_root=tmp_path)

    def test_create_project_creates_structure(self, tmp_path):
        pm = self._make_pm(tmp_path)
        proj = pm.create_project("test_lora", base_model="runwayml/sd-v1-5", architecture="sd15")
        assert (proj.path / "project.json").exists()
        for subdir in ["dataset", "models", "samples", "checkpoints", "logs", ".cache",
                       "presets", "history"]:
            assert (proj.path / subdir).is_dir(), f"Missing subdir: {subdir}"

    def test_create_project_metadata(self, tmp_path):
        pm = self._make_pm(tmp_path)
        proj = pm.create_project("my_flux", base_model="black-forest-labs/FLUX.1-dev", architecture="flux")
        assert proj.name == "my_flux"
        assert proj.base_model == "black-forest-labs/FLUX.1-dev"
        assert proj.architecture == "flux"
        assert isinstance(proj.created_at, datetime)

    def test_create_project_duplicate_raises(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.create_project("duplicate")
        with pytest.raises(FileExistsError):
            pm.create_project("duplicate")

    def test_load_project(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.create_project("loadme", base_model="sd-v1-5", architecture="sd15")
        loaded = pm.load_project("loadme")
        assert loaded.name == "loadme"
        assert loaded.base_model == "sd-v1-5"
        assert loaded.architecture == "sd15"

    def test_load_project_not_found(self, tmp_path):
        pm = self._make_pm(tmp_path)
        with pytest.raises(FileNotFoundError):
            pm.load_project("nonexistent")

    def test_list_projects(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.create_project("alpha")
        pm.create_project("beta")
        pm.create_project("gamma")
        projects = pm.list_projects()
        names = [p.name for p in projects]
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_list_projects_empty(self, tmp_path):
        pm = self._make_pm(tmp_path)
        assert pm.list_projects() == []

    def test_delete_project(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.create_project("to_delete")
        assert pm.project_exists("to_delete")
        pm.delete_project("to_delete", confirm=False)
        assert not pm.project_exists("to_delete")

    def test_delete_project_requires_confirm(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.create_project("guarded")
        with pytest.raises(RuntimeError):
            pm.delete_project("guarded")  # confirm=True by default

    def test_delete_project_not_found(self, tmp_path):
        pm = self._make_pm(tmp_path)
        with pytest.raises(FileNotFoundError):
            pm.delete_project("ghost", confirm=False)

    def test_project_exists(self, tmp_path):
        pm = self._make_pm(tmp_path)
        assert not pm.project_exists("nope")
        pm.create_project("yep")
        assert pm.project_exists("yep")

    def test_name_with_spaces(self, tmp_path):
        pm = self._make_pm(tmp_path)
        proj = pm.create_project("My Cool Project")
        assert proj.path.name == "My_Cool_Project"
        loaded = pm.load_project("My Cool Project")
        assert loaded.name == "My Cool Project"

    def test_set_projects_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABUILDER_CONFIG_DIR", str(tmp_path / "config"))
        from dataset_sorter.project_manager import ProjectManager
        pm = ProjectManager()
        new_root = tmp_path / "new_root"
        pm.set_projects_root(new_root)
        assert pm.get_projects_root() == new_root


# ─────────────────────────────────────────────────────────────────────────────
# Project helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectHelpers:
    def _make_project(self, tmp_path, name="test"):
        from dataset_sorter.project_manager import ProjectManager
        pm = ProjectManager(projects_root=tmp_path)
        return pm.create_project(name)

    def test_dataset_path(self, tmp_path):
        proj = self._make_project(tmp_path)
        assert proj.dataset_path == proj.path / "dataset"

    def test_get_latest_checkpoint_empty(self, tmp_path):
        proj = self._make_project(tmp_path)
        assert proj.get_latest_checkpoint() is None

    def test_get_latest_checkpoint(self, tmp_path):
        proj = self._make_project(tmp_path)
        ckpt1 = proj.checkpoints_path / "step_000100"
        ckpt2 = proj.checkpoints_path / "step_000200"
        ckpt1.mkdir(parents=True)
        ckpt2.mkdir(parents=True)
        # Touch ckpt2 to make it newer
        import time
        time.sleep(0.01)
        ckpt2.touch()
        latest = proj.get_latest_checkpoint()
        assert latest == ckpt2

    def test_get_samples_empty(self, tmp_path):
        proj = self._make_project(tmp_path)
        assert proj.get_samples() == []

    def test_get_samples(self, tmp_path):
        proj = self._make_project(tmp_path)
        img = proj.samples_path / "sample_000.png"
        img.write_bytes(b"fake png")
        samples = proj.get_samples()
        assert img in samples

    def test_save_and_load_preset(self, tmp_path):
        proj = self._make_project(tmp_path)
        config = {"learning_rate": 1e-4, "lora_rank": 32}
        proj.save_preset("my_preset", config)
        assert "my_preset" in proj.list_presets()
        loaded = proj.load_preset("my_preset")
        assert loaded["learning_rate"] == 1e-4
        assert loaded["lora_rank"] == 32

    def test_list_presets_empty(self, tmp_path):
        proj = self._make_project(tmp_path)
        assert proj.list_presets() == []

    def test_training_history(self, tmp_path):
        proj = self._make_project(tmp_path)
        assert proj.get_training_history() == []
        proj.record_training_run({"steps": 1000, "final_loss": 0.02})
        proj.record_training_run({"steps": 2000, "final_loss": 0.015})
        history = proj.get_training_history()
        assert len(history) == 2
        assert history[0]["steps"] == 1000

    def test_record_training_updates_last_trained(self, tmp_path):
        proj = self._make_project(tmp_path)
        assert proj.last_trained is None
        proj.record_training_run({"steps": 500})
        assert proj.last_trained is not None

    def test_project_save_round_trip(self, tmp_path):
        proj = self._make_project(tmp_path, "roundtrip")
        proj.base_model = "some/model"
        proj.architecture = "sdxl"
        proj.save()
        data = json.loads((proj.path / "project.json").read_text())
        assert data["base_model"] == "some/model"
        assert data["architecture"] == "sdxl"
        assert data["name"] == "roundtrip"
        assert data["version"] == 2
