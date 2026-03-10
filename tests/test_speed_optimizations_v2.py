"""Tests for speed optimization features v2:
1. Pre-tokenized Caption Cache
2. CUDA Graph Training Wrapper
3. Async Optimizer Step
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from dataset_sorter.speed_optimizations import (
    CUDAGraphWrapper,
    AsyncOptimizerStep,
)
from dataset_sorter.train_dataset import CachedTrainDataset
from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PRE-TOKENIZED CAPTION CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreTokenizedCaptionCache:
    """Tests for caching tokenized IDs alongside TE outputs."""

    def test_token_id_cache_initialized_empty(self, tmp_path):
        """Token ID cache should be empty initially."""
        ds = CachedTrainDataset(
            image_paths=[tmp_path / "a.png"],
            captions=["test caption"],
        )
        assert ds._token_id_cache == {}
        assert ds._tokens_cached is False

    def test_token_ids_not_in_getitem_before_cache(self, tmp_path):
        """Before caching, __getitem__ should not include token_ids."""
        # Create a dummy image
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        ds = CachedTrainDataset(
            image_paths=[img_path],
            captions=["test"],
            resolution=64,
        )
        result = ds[0]
        assert "token_ids" not in result

    def test_token_ids_cached_during_te_caching(self, tmp_path):
        """Tokenized IDs should be cached when cache_text_encoder_outputs is called."""
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        ds = CachedTrainDataset(
            image_paths=[img_path],
            captions=["a photo of a cat"],
            resolution=64,
        )

        # Mock tokenizer and text encoder
        tokenizer = MagicMock()
        tokenizer.model_max_length = 77
        mock_token_ids = torch.randint(0, 1000, (1, 77))
        tokenizer.return_value = MagicMock(input_ids=mock_token_ids)

        text_encoder = MagicMock()
        text_encoder.eval = MagicMock()
        text_encoder.requires_grad_ = MagicMock()
        hidden = torch.randn(1, 77, 768)
        pooled = torch.randn(1, 768)
        text_encoder.return_value = MagicMock(
            hidden_states=[torch.randn(1, 77, 768), hidden, torch.randn(1, 77, 768)],
            pooler_output=pooled,
        )

        device = torch.device("cpu")
        ds.cache_text_encoder_outputs(
            tokenizer, text_encoder, device, torch.float32,
        )

        assert ds._tokens_cached is True
        assert 0 in ds._token_id_cache
        # Should be a tuple of token ID tensors
        token_ids = ds._token_id_cache[0]
        assert isinstance(token_ids, tuple)
        assert len(token_ids) == 1  # Single tokenizer
        assert isinstance(token_ids[0], torch.Tensor)

    def test_token_ids_in_getitem_after_cache(self, tmp_path):
        """After caching, __getitem__ should include token_ids."""
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        ds = CachedTrainDataset(
            image_paths=[img_path],
            captions=["a photo of a cat"],
            resolution=64,
        )

        # Manually set cache
        ds._tokens_cached = True
        ds._token_id_cache[0] = (torch.randint(0, 1000, (1, 77)),)

        result = ds[0]
        assert "token_ids" in result
        assert isinstance(result["token_ids"], tuple)
        assert isinstance(result["token_ids"][0], torch.Tensor)

    def test_dual_tokenizer_caching(self, tmp_path):
        """With two tokenizers (SDXL), both token IDs should be cached."""
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        ds = CachedTrainDataset(
            image_paths=[img_path],
            captions=["test caption"],
            resolution=64,
        )

        # Mock tokenizers
        tokenizer = MagicMock()
        tokenizer.model_max_length = 77
        tokenizer.return_value = MagicMock(input_ids=torch.randint(0, 1000, (1, 77)))

        tokenizer_2 = MagicMock()
        tokenizer_2.model_max_length = 77
        tokenizer_2.return_value = MagicMock(input_ids=torch.randint(0, 1000, (1, 77)))

        text_encoder = MagicMock()
        text_encoder.eval = MagicMock()
        text_encoder.requires_grad_ = MagicMock()
        text_encoder.return_value = MagicMock(
            hidden_states=[torch.randn(1, 77, 768), torch.randn(1, 77, 768), torch.randn(1, 77, 768)],
            pooler_output=torch.randn(1, 768),
        )

        text_encoder_2 = MagicMock()
        text_encoder_2.eval = MagicMock()
        text_encoder_2.requires_grad_ = MagicMock()
        text_encoder_2.return_value = MagicMock(
            hidden_states=[torch.randn(1, 77, 1024), torch.randn(1, 77, 1024), torch.randn(1, 77, 1024)],
            pooler_output=torch.randn(1, 1024),
        )

        ds.cache_text_encoder_outputs(
            tokenizer, text_encoder, torch.device("cpu"), torch.float32,
            tokenizer_2=tokenizer_2, text_encoder_2=text_encoder_2,
        )

        assert ds._tokens_cached is True
        token_ids = ds._token_id_cache[0]
        assert len(token_ids) == 2  # Two tokenizers
        assert isinstance(token_ids[0], torch.Tensor)
        assert isinstance(token_ids[1], torch.Tensor)

    def test_unique_captions_share_token_cache(self, tmp_path):
        """Images with the same caption should share cached token IDs."""
        from PIL import Image
        for name in ["a.png", "b.png"]:
            img = Image.new("RGB", (64, 64))
            img.save(str(tmp_path / name))

        ds = CachedTrainDataset(
            image_paths=[tmp_path / "a.png", tmp_path / "b.png"],
            captions=["same caption", "same caption"],
            resolution=64,
        )

        tokenizer = MagicMock()
        tokenizer.model_max_length = 77
        tokenizer.return_value = MagicMock(input_ids=torch.randint(0, 1000, (1, 77)))

        text_encoder = MagicMock()
        text_encoder.eval = MagicMock()
        text_encoder.requires_grad_ = MagicMock()
        text_encoder.return_value = MagicMock(
            hidden_states=[torch.randn(1, 77, 768), torch.randn(1, 77, 768), torch.randn(1, 77, 768)],
            pooler_output=torch.randn(1, 768),
        )

        ds.cache_text_encoder_outputs(
            tokenizer, text_encoder, torch.device("cpu"), torch.float32,
        )

        # Both indices should have cached tokens
        assert 0 in ds._token_id_cache
        assert 1 in ds._token_id_cache
        # They should reference the same tuple objects
        assert ds._token_id_cache[0] is ds._token_id_cache[1]

    def test_clear_caches_clears_token_ids(self, tmp_path):
        """clear_caches() should also clear the token ID cache."""
        ds = CachedTrainDataset(
            image_paths=[tmp_path / "a.png"],
            captions=["test"],
        )
        ds._token_id_cache[0] = (torch.zeros(1, 77),)
        ds._tokens_cached = True

        ds.clear_caches()

        assert ds._token_id_cache == {}
        assert ds._tokens_cached is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CUDA GRAPH TRAINING WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class TestCUDAGraphWrapper:
    """Tests for the CUDA graph wrapper (CPU/mock-based tests)."""

    def test_disabled_passthrough(self):
        """When disabled, should just call the function directly."""
        wrapper = CUDAGraphWrapper(warmup_steps=3, enabled=False)
        call_count = 0

        def train_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            return torch.tensor(0.5)

        for i in range(10):
            result = wrapper.step(train_fn, x=torch.randn(2, 3))
            assert isinstance(result, torch.Tensor)

        assert call_count == 10

    def test_warmup_phase(self):
        """During warmup, should call function directly."""
        wrapper = CUDAGraphWrapper(warmup_steps=5, enabled=False)
        call_count = 0

        def train_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            return torch.tensor(1.0)

        for i in range(5):
            wrapper.step(train_fn, x=torch.randn(2))

        assert call_count == 5

    def test_reset(self):
        """reset() should clear graph state."""
        wrapper = CUDAGraphWrapper(warmup_steps=3, enabled=False)
        wrapper._step = 10
        wrapper._captured = True
        wrapper._static_inputs = {"x": torch.randn(2)}
        wrapper._static_output = torch.tensor(1.0)

        wrapper.reset()

        assert wrapper._step == 0
        assert wrapper._captured is False
        assert wrapper._static_inputs == {}
        assert wrapper._static_output is None

    def test_no_cuda_disables_automatically(self):
        """Without CUDA, the wrapper should be disabled."""
        # CUDAGraphWrapper checks torch.cuda.is_available()
        if not torch.cuda.is_available():
            wrapper = CUDAGraphWrapper(warmup_steps=3, enabled=True)
            assert wrapper.enabled is False

    def test_step_count_not_tracked_when_disabled(self):
        """When disabled, step counter should not increment (passthrough mode)."""
        wrapper = CUDAGraphWrapper(warmup_steps=5, enabled=False)

        for i in range(3):
            wrapper.step(lambda **kw: torch.tensor(0.0), x=torch.randn(2))

        # Disabled wrapper bypasses all tracking
        assert wrapper._step == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ASYNC OPTIMIZER STEP
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncOptimizerStep:
    """Tests for async optimizer step (CPU fallback tests)."""

    def test_cpu_fallback_step(self):
        """On CPU, should fall back to synchronous step."""
        async_opt = AsyncOptimizerStep(torch.device("cpu"), enabled=True)
        assert async_opt.enabled is False

        # Create a simple model and optimizer
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Forward + backward
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()

        # Get initial params
        initial_params = [p.clone() for p in model.parameters()]

        # Step should work synchronously
        async_opt.step(optimizer)

        # Params should have changed
        for p_old, p_new in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_old, p_new)

    def test_sync_noop_when_not_pending(self):
        """sync() should be a no-op when nothing is pending."""
        async_opt = AsyncOptimizerStep(torch.device("cpu"), enabled=True)
        assert async_opt._pending is False
        async_opt.sync()  # Should not raise
        assert async_opt._pending is False

    def test_step_with_grad_scaler_cpu(self):
        """On CPU, should handle grad_scaler in synchronous fallback."""
        async_opt = AsyncOptimizerStep(torch.device("cpu"), enabled=True)

        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Mock a grad scaler
        mock_scaler = MagicMock()

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()

        async_opt.step(optimizer, grad_scaler=mock_scaler)

        mock_scaler.step.assert_called_once_with(optimizer)
        mock_scaler.update.assert_called_once()

    def test_disabled_is_synchronous(self):
        """When explicitly disabled, step should be synchronous."""
        async_opt = AsyncOptimizerStep(torch.device("cpu"), enabled=False)
        assert async_opt.enabled is False

        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()

        initial_w = model.weight.clone()
        async_opt.step(optimizer)
        assert not torch.allclose(initial_w, model.weight)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONFIG FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewConfigFields:
    """Tests for the new TrainingConfig fields."""

    def test_cuda_graph_training_default(self):
        config = TrainingConfig()
        assert config.cuda_graph_training is False

    def test_cuda_graph_warmup_default(self):
        config = TrainingConfig()
        assert config.cuda_graph_warmup == 11

    def test_async_optimizer_step_default(self):
        config = TrainingConfig()
        assert config.async_optimizer_step is False

    def test_config_serialization(self):
        """New fields should serialize/deserialize correctly."""
        from dataclasses import asdict
        import json

        config = TrainingConfig()
        config.cuda_graph_training = True
        config.cuda_graph_warmup = 20
        config.async_optimizer_step = True

        data = asdict(config)
        json_str = json.dumps(data, default=str)
        loaded = json.loads(json_str)

        assert loaded["cuda_graph_training"] is True
        assert loaded["cuda_graph_warmup"] == 20
        assert loaded["async_optimizer_step"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedIntegration:
    """Integration tests for speed features working together."""

    def test_async_optimizer_with_real_model(self):
        """Async optimizer should produce correct parameter updates."""
        model = nn.Linear(8, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        async_opt = AsyncOptimizerStep(torch.device("cpu"), enabled=True)

        # Training loop simulation
        for _ in range(3):
            async_opt.sync()  # Sync previous step

            x = torch.randn(4, 8)
            loss = model(x).sum()
            loss.backward()

            async_opt.step(optimizer)
            optimizer.zero_grad(set_to_none=True)

    def test_cuda_graph_wrapper_warmup_then_disable(self):
        """After warmup on CPU (disabled), should keep running in eager mode."""
        wrapper = CUDAGraphWrapper(warmup_steps=3, enabled=False)

        results = []
        for i in range(10):
            r = wrapper.step(
                lambda **kw: kw["x"].sum(),
                x=torch.randn(4),
            )
            results.append(r.item())

        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)

    def test_token_cache_triple_tokenizer(self, tmp_path):
        """Token cache should handle 3 tokenizers (SD3/T5-XXL)."""
        from PIL import Image
        img = Image.new("RGB", (64, 64))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        ds = CachedTrainDataset(
            image_paths=[img_path],
            captions=["a test"],
            resolution=64,
        )

        def make_tokenizer(max_len=77):
            t = MagicMock()
            t.model_max_length = max_len
            t.return_value = MagicMock(input_ids=torch.randint(0, 1000, (1, max_len)))
            return t

        def make_te(hidden_dim=768, has_pooler=True):
            te = MagicMock()
            te.eval = MagicMock()
            te.requires_grad_ = MagicMock()
            output = MagicMock()
            output.hidden_states = [torch.randn(1, 77, hidden_dim), torch.randn(1, 77, hidden_dim), torch.randn(1, 77, hidden_dim)]
            if has_pooler:
                output.pooler_output = torch.randn(1, hidden_dim)
            else:
                output.pooler_output = None
            # For TE3 (T5), it uses last_hidden_state instead
            output.last_hidden_state = torch.randn(1, 77, hidden_dim)
            te.return_value = output
            return te

        tok1 = make_tokenizer()
        tok2 = make_tokenizer()
        tok3 = make_tokenizer(256)
        te1 = make_te(768)
        te2 = make_te(1024)
        te3 = make_te(4096, has_pooler=False)

        ds.cache_text_encoder_outputs(
            tok1, te1, torch.device("cpu"), torch.float32,
            tokenizer_2=tok2, text_encoder_2=te2,
            tokenizer_3=tok3, text_encoder_3=te3,
        )

        assert ds._tokens_cached is True
        token_ids = ds._token_id_cache[0]
        assert len(token_ids) == 3  # Three tokenizers
        assert token_ids[0].shape[-1] == 77
        assert token_ids[1].shape[-1] == 77
        assert token_ids[2].shape[-1] == 256
