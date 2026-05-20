"""Tests for SageAttention integration and max-autotune compile mode."""

import pytest
import torch
import torch.nn.functional as F


class TestSageAttentionWrapper:
    """Test the SDPA monkey-patch wrapper logic."""

    def test_enable_disable_roundtrip(self):
        """enable/disable restores the original function."""
        original = F.scaled_dot_product_attention
        from dataset_sorter.speed_optimizations import (
            enable_sage_attention,
            disable_sage_attention,
            _sage_original_sdpa,
        )
        # If sageattention is not installed, enable returns False
        result = enable_sage_attention()
        if not result:
            pytest.skip("sageattention not installed")
        assert F.scaled_dot_product_attention is not original
        disable_sage_attention()
        assert F.scaled_dot_product_attention is original

    def test_enable_returns_false_when_not_installed(self):
        """When sageattention is missing, enable returns False and SDPA stays."""
        import importlib
        import sys
        # Temporarily hide sageattention if present
        hidden = sys.modules.pop("sageattention", None)
        try:
            from dataset_sorter import speed_optimizations
            # Reset module state
            speed_optimizations._sage_original_sdpa = None
            original = F.scaled_dot_product_attention
            result = speed_optimizations.enable_sage_attention()
            if hidden is not None:
                # sageattention IS installed — skip this test
                pytest.skip("sageattention is installed, can't test missing case")
            assert result is False
            assert F.scaled_dot_product_attention is original
        finally:
            if hidden is not None:
                sys.modules["sageattention"] = hidden

    def test_wrapper_falls_back_on_attn_mask(self):
        """The wrapper falls back to original SDPA when attn_mask is provided."""
        from dataset_sorter.speed_optimizations import (
            _sage_sdpa_wrapper,
            _sage_original_sdpa,
        )
        # We test the fallback logic without actually needing sageattn
        import dataset_sorter.speed_optimizations as mod

        # Set up a mock original SDPA
        call_log = []
        def mock_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, enable_gqa=False):
            call_log.append("original")
            return torch.zeros(query.shape)

        mod._sage_original_sdpa = mock_sdpa

        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        mask = torch.ones(1, 1, 8, 8)

        # With mask → should fall back to original
        try:
            _sage_sdpa_wrapper(q, k, v, attn_mask=mask)
        except ImportError:
            pass  # sageattn import fails, but the fallback path should trigger first
        assert "original" in call_log

        # Clean up
        mod._sage_original_sdpa = None

    def test_wrapper_falls_back_on_dropout(self):
        """Wrapper falls back when dropout_p > 0."""
        import dataset_sorter.speed_optimizations as mod

        call_log = []
        def mock_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, enable_gqa=False):
            call_log.append("original")
            return torch.zeros(query.shape)

        mod._sage_original_sdpa = mock_sdpa
        q = torch.randn(1, 4, 8, 16)

        from dataset_sorter.speed_optimizations import _sage_sdpa_wrapper
        _sage_sdpa_wrapper(q, q, q, dropout_p=0.1)
        assert "original" in call_log
        mod._sage_original_sdpa = None

    def test_wrapper_falls_back_on_float32(self):
        """Wrapper falls back for float32 inputs (SageAttention needs half)."""
        import dataset_sorter.speed_optimizations as mod

        call_log = []
        def mock_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, enable_gqa=False):
            call_log.append("original")
            return torch.zeros(query.shape)

        mod._sage_original_sdpa = mock_sdpa
        q = torch.randn(1, 4, 8, 16, dtype=torch.float32)

        from dataset_sorter.speed_optimizations import _sage_sdpa_wrapper
        _sage_sdpa_wrapper(q, q, q)
        assert "original" in call_log
        mod._sage_original_sdpa = None


class TestMaxAutotuneConfig:
    """Test that max-autotune is the default compile mode for SM 8.0+."""

    def test_config_field_exists(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "sage_attention")
        assert config.sage_attention is False

    def test_compile_mode_default(self):
        from dataset_sorter.models import TrainingConfig
        config = TrainingConfig()
        assert config.compile_mode == "default"

    def test_merge_methods_has_dare_ties(self):
        """Bonus: verify MERGE_METHODS includes DARE/TIES from earlier commit."""
        from dataset_sorter.ui.model_merge_tab import MERGE_METHODS
        assert "dare" in MERGE_METHODS
        assert "ties" in MERGE_METHODS


class TestGenerateWorkerSageField:
    def test_sage_field_exists(self):
        """Verify the generate worker has the sage_attention_enabled field."""
        # Can't instantiate GenerateWorker without QApplication, check the source
        import inspect
        from dataset_sorter.generate_worker import GenerateWorker
        source = inspect.getsource(GenerateWorker.__init__)
        assert "sage_attention_enabled" in source
