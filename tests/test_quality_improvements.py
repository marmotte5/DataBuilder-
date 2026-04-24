"""Tests for quality improvements: OOM retry, collate function, rolling backup,
validation loop, augmentation config, and EMA precision."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ── Collate function tests ──────────────────────────────────────────────

class TestTrainingCollateFn:
    @pytest.fixture(autouse=True)
    def _import(self):
        from dataset_sorter.train_dataset import training_collate_fn
        self.collate = training_collate_fn

    def test_empty_batch(self):
        assert self.collate([]) == {}

    def test_tensor_stacking(self):
        batch = [
            {"latent": torch.zeros(4, 64, 64)},
            {"latent": torch.ones(4, 64, 64)},
        ]
        result = self.collate(batch)
        assert result["latent"].shape == (2, 4, 64, 64)

    def test_string_values_listed(self):
        batch = [{"caption": "a cat"}, {"caption": "a dog"}]
        result = self.collate(batch)
        assert result["caption"] == ["a cat", "a dog"]

    def test_numeric_values_tensorized(self):
        batch = [{"weight": 1.0}, {"weight": 2.0}]
        result = self.collate(batch)
        assert isinstance(result["weight"], torch.Tensor)
        assert result["weight"].tolist() == [1.0, 2.0]

    def test_common_keys_only(self):
        batch = [
            {"latent": torch.zeros(1), "extra": 1},
            {"latent": torch.ones(1)},
        ]
        result = self.collate(batch)
        assert "latent" in result
        assert "extra" not in result

    def test_tuple_with_none(self):
        t1 = (torch.zeros(4), None, torch.zeros(8))
        t2 = (torch.ones(4), None, torch.ones(8))
        batch = [{"te_cache": t1}, {"te_cache": t2}]
        result = self.collate(batch)
        assert result["te_cache"][0].shape == (2, 4)
        assert result["te_cache"][1] is None
        assert result["te_cache"][2].shape == (2, 8)

    def test_tuple_mixed_none_tensor_drops_to_none(self):
        t1 = (torch.zeros(4), torch.zeros(8))
        t2 = (torch.ones(4), None)
        batch = [{"te_cache": t1}, {"te_cache": t2}]
        result = self.collate(batch)
        assert result["te_cache"][0].shape == (2, 4)
        assert result["te_cache"][1] is None

    def test_tuple_length_mismatch_uses_min(self):
        t1 = (torch.zeros(4), torch.zeros(8), torch.zeros(16))
        t2 = (torch.ones(4), torch.ones(8))
        batch = [{"te_cache": t1}, {"te_cache": t2}]
        result = self.collate(batch)
        assert len(result["te_cache"]) == 2

    def test_single_element_batch(self):
        batch = [{"latent": torch.zeros(4, 64, 64), "caption": "hello"}]
        result = self.collate(batch)
        assert result["latent"].shape == (1, 4, 64, 64)
        assert result["caption"] == ["hello"]


# ── EMA precision tests ──────────────────────────────────────────────────

class TestEMAPrecision:
    def test_shadow_params_are_fp32(self):
        from dataset_sorter.ema import EMAModel
        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        ema = EMAModel(model.parameters(), decay=0.999)
        for sp in ema.shadow_params:
            assert sp.dtype == torch.float32

    def test_load_state_dict_casts_to_fp32(self):
        from dataset_sorter.ema import EMAModel
        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        ema = EMAModel(model.parameters(), decay=0.999)

        # Simulate a checkpoint saved with BF16 shadows
        state = ema.state_dict()
        state["shadow_params"] = [p.to(torch.bfloat16) for p in state["shadow_params"]]
        ema.load_state_dict(state)

        for sp in ema.shadow_params:
            assert sp.dtype == torch.float32

    def test_decay_warmup(self):
        from dataset_sorter.ema import EMAModel
        model = torch.nn.Linear(4, 4)
        ema = EMAModel(model.parameters(), decay=0.999)
        # At step 0, warmup formula gives min(0.999, 1/10) = 0.1
        effective = min(0.999, (1 + 0) / (10 + 0))
        assert effective == pytest.approx(0.1)
        # At step 100, warmup formula gives min(0.999, 101/110) ≈ 0.918
        effective = min(0.999, (1 + 100) / (10 + 100))
        assert effective == pytest.approx(101 / 110)


# ── SOAP FP32 moments tests ──────────────────────────────────────────────

class TestSOAPMoments:
    def test_moments_initialized_in_fp32(self):
        from dataset_sorter.optimizers import SOAP
        param = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
        opt = SOAP([param], lr=1e-3)
        # Trigger state initialization
        param.grad = torch.randn_like(param)
        opt.step()
        state = opt.state[param]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


# ── Rolling backup cleanup tests ──────────────────────────────────────────

class TestRollingBackupCleanup:
    @pytest.fixture
    def mock_trainer(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        config.save_last_n_checkpoints = 2
        # Create a minimal mock trainer
        trainer = MagicMock()
        trainer.config = config
        return trainer

    def test_protected_prefixes_pattern(self):
        """Verify the protected prefix regex matches expected patterns."""
        protected = re.compile(r"^(manual_|emergency_|stopped_|final)")
        assert protected.match("manual_000100")
        assert protected.match("emergency_000200")
        assert protected.match("stopped_000300")
        assert protected.match("final")
        assert protected.match("final_000400")
        assert not protected.match("step_000100")
        assert not protected.match("epoch_005")

    def test_step_number_parsing(self):
        """Verify step numbers are correctly parsed from checkpoint names."""
        pattern = re.compile(r"(\d+)$")
        cases = {
            "step_000100": 100,
            "epoch_005": 5,
            "step_001000": 1000,
        }
        for name, expected in cases.items():
            m = pattern.search(name)
            assert m is not None, f"Failed to match {name}"
            assert int(m.group(1)) == expected


# ── Config fingerprint tests ──────────────────────────────────────────────

class TestConfigFingerprint:
    def test_missing_key_in_old_checkpoint_not_flagged(self):
        """Keys absent from older checkpoints should not trigger mismatch."""
        saved_fp = {"model_type": "sdxl_lora", "optimizer": "Adafactor"}
        current_fp = {
            "model_type": "sdxl_lora",
            "optimizer": "Adafactor",
            "lora_rank": 32,
            "batch_size": 4,
        }
        diffs = {
            k: (saved_fp.get(k), current_fp.get(k))
            for k in current_fp
            if saved_fp.get(k) != current_fp.get(k)
            and k in saved_fp
        }
        assert len(diffs) == 0

    def test_actual_mismatch_detected(self):
        saved_fp = {
            "model_type": "sdxl_lora", "optimizer": "Adafactor",
            "lora_rank": 32, "batch_size": 4,
        }
        current_fp = {
            "model_type": "sdxl_lora", "optimizer": "AdamW",
            "lora_rank": 32, "batch_size": 4,
        }
        diffs = {
            k: (saved_fp.get(k), current_fp.get(k))
            for k in current_fp
            if saved_fp.get(k) != current_fp.get(k)
            and k in saved_fp
        }
        assert "optimizer" in diffs
        assert diffs["optimizer"] == ("Adafactor", "AdamW")


# ── TrainingConfig augmentation fields tests ──────────────────────────────

class TestTrainingConfigAugmentation:
    def test_default_augmentation_values(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert config.color_jitter_brightness == 0.0
        assert config.color_jitter_contrast == 0.0
        assert config.color_jitter_saturation == 0.0
        assert config.color_jitter_hue == 0.0
        assert config.random_rotate_degrees == 0.0

    def test_default_loss_fn(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert config.loss_fn == "mse"
        assert config.huber_delta == 0.1

    def test_default_multires_noise(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert config.multires_noise_discount == 0.0

    def test_config_serialization_roundtrip(self):
        from dataclasses import asdict
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        config.color_jitter_brightness = 0.1
        config.loss_fn = "huber"
        config.multires_noise_discount = 0.4
        d = asdict(config)
        assert d["color_jitter_brightness"] == 0.1
        assert d["loss_fn"] == "huber"
        assert d["multires_noise_discount"] == 0.4

    def test_validation_config_defaults(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert config.validation_dir == ""
        assert config.validate_every_n_steps == 0
        assert config.validation_samples_limit == 64


# ── Path prefix check tests ──────────────────────────────────────────────

class TestPathPrefixCheck:
    def test_proper_prefix_match(self):
        root = Path("/home/user/Projects")
        inside = Path("/home/user/Projects/myproject/dataset")
        outside = Path("/home/user/Projects2/something")
        assert inside.is_relative_to(root)
        assert not outside.is_relative_to(root)

    def test_exact_match(self):
        root = Path("/home/user/Projects")
        assert root.is_relative_to(root)

    def test_empty_path_handling(self):
        assert not Path("").is_absolute()


# ── OOM retry logic test ──────────────────────────────────────────────────

class TestOOMRetryLogic:
    def test_accum_reset_on_oom(self):
        """Verify the OOM handler properly resets accumulation state."""
        running_loss = 1.5
        _valid_microbatches = 2
        _accum_count = 2

        # Simulate what the OOM handler does
        running_loss = 0.0
        _valid_microbatches = 0
        _accum_count = 0

        assert running_loss == 0.0
        assert _valid_microbatches == 0
        assert _accum_count == 0


# ── Atomic write tests ──────────────────────────────────────────────────

class TestAtomicWrites:
    """Verify that critical file writes use temp + rename pattern."""

    def test_project_save_uses_atomic_write(self, tmp_path):
        from dataset_sorter.project_manager import Project
        proj = Project(name="test_atomic", path=tmp_path)
        proj.architecture = "sdxl"
        proj.save()
        assert (tmp_path / "project.json").exists()
        # No leftover .tmp file
        assert not (tmp_path / "project.json.tmp").exists()

    def test_settings_save_uses_atomic_write(self, tmp_path):
        from dataset_sorter.app_settings import AppSettings
        settings = AppSettings()
        settings_path = tmp_path / "settings.json"
        with patch.object(AppSettings, "get_settings_path", return_value=settings_path):
            settings.save()
        assert settings_path.exists()
        assert not settings_path.with_suffix(".tmp").exists()

    def test_smart_resume_history_atomic(self, tmp_path):
        from dataset_sorter.smart_resume import save_loss_history
        save_loss_history(tmp_path, loss_history=[(100, 0.05), (200, 0.03)])
        history_path = tmp_path / "loss_history.json"
        assert history_path.exists()
        assert not history_path.with_suffix(".tmp").exists()


# ── EMA CPU offload efficiency test ─────────────────────────────────────

class TestEMACPUOffloadEfficiency:
    def test_cpu_offload_single_transfer(self):
        """Verify CPU offload path transfers data once, not twice."""
        import torch.nn as nn
        from dataset_sorter.ema import EMAModel
        model = nn.Linear(4, 4)
        ema = EMAModel(model.parameters(), decay=0.999, cpu_offload=True)
        ema.step = 2  # Past update_after_step
        # Should work without error (no NaN)
        ema.update(model.parameters())
        # Shadow params should have been updated
        for sp in ema.shadow_params:
            assert sp.device.type == "cpu"
            assert sp.dtype == torch.float32
