"""Tests for audit-round fixes: encoder_attention_mask, Windows compat, batch UI."""

import inspect
import sys
import pytest


class TestPixArtEncoderAttentionMask:
    """PixArt backend must return attention_mask and pass it to the transformer."""

    def test_encode_text_batch_returns_mask(self):
        """encode_text_batch must return (hidden, mask) tuple."""
        source = _get_source("dataset_sorter.train_backend_pixart", "PixArtBackend",
                             "encode_text_batch")
        assert "attention_mask" in source
        assert "return (encoder_hidden, attention_mask)" in source

    def test_training_step_extracts_mask(self):
        """training_step must extract mask from te_out and pass it."""
        source = _get_source("dataset_sorter.train_backend_pixart", "PixArtBackend",
                             "training_step")
        assert "encoder_attention_mask" in source
        assert "enc_mask" in source

    def test_flow_training_step_accepts_mask(self):
        """flow_training_step must accept encoder_attention_mask kwarg."""
        from dataset_sorter.train_backend_base import TrainBackendBase
        sig = inspect.signature(TrainBackendBase.flow_training_step)
        assert "encoder_attention_mask" in sig.parameters


class TestSanaEncoderAttentionMask:
    """Sana backend must return attention_mask and pass it to the transformer."""

    def test_encode_text_batch_returns_mask(self):
        source = _get_source("dataset_sorter.train_backend_sana", "SanaBackend",
                             "encode_text_batch")
        assert "attention_mask" in source
        assert "return (encoder_hidden, attention_mask)" in source

    def test_training_step_extracts_mask(self):
        source = _get_source("dataset_sorter.train_backend_sana", "SanaBackend",
                             "training_step")
        assert "encoder_attention_mask" in source


class TestWindowsTmpfsCompat:
    """get_tmpfs_cache_dir must not crash on Windows."""

    def test_returns_none_on_windows(self):
        """On Windows (or when simulated), return None without os.statvfs crash."""
        from unittest.mock import patch
        from dataset_sorter.io_speed import get_tmpfs_cache_dir
        # Patch sys.platform at the module level since io_speed imports sys locally
        with patch("sys.platform", "win32"):
            result = get_tmpfs_cache_dir()
            assert result is None

    def test_source_has_win32_guard(self):
        """Source must contain early return for win32."""
        import dataset_sorter.io_speed as mod
        source = inspect.getsource(mod.get_tmpfs_cache_dir)
        assert "win32" in source


class TestBatchGenButtonDisabling:
    """Queue control buttons must be stored as instance vars for disabling."""

    def test_buttons_are_instance_vars(self):
        source = inspect.getsource(
            __import__("dataset_sorter.ui.batch_generation_tab",
                       fromlist=["BatchGenerationTab"]).BatchGenerationTab._build_ui
        )
        for attr in ("self._btn_add", "self._btn_remove", "self._btn_clear",
                      "self._btn_import_csv", "self._btn_import_json",
                      "self._btn_import_txt", "self._btn_export",
                      "self._btn_duplicate"):
            assert attr in source, f"{attr} not stored as instance variable"

    def test_set_queue_buttons_enabled_exists(self):
        from dataset_sorter.ui.batch_generation_tab import BatchGenerationTab
        assert hasattr(BatchGenerationTab, "_set_queue_buttons_enabled")


class TestNaNLossEpochAccounting:
    """NaN-only optimizer steps must not inflate epoch loss average."""

    def test_had_valid_guard_in_source(self):
        from dataset_sorter.trainer import Trainer
        source = inspect.getsource(Trainer.train)
        assert "_had_valid" in source
        assert "_had_valid = _valid_microbatches > 0" in source


class TestPixArtSanaCacheOverride:
    """PixArt and Sana must use encode_fn for TE caching to preserve masks."""

    def test_trainer_uses_encode_fn_for_pixart_sana(self):
        from dataset_sorter.trainer import Trainer
        source = inspect.getsource(Trainer.setup)
        assert "'pixart'" in source
        assert "'sana'" in source


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_source(module_path: str, class_name: str, method_name: str) -> str:
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)
    method = getattr(cls, method_name)
    return inspect.getsource(method)
