"""Tests for Curriculum Learning and Per-Timestep EMA Sampling."""

import numpy as np
import pytest
import torch

from dataset_sorter.curriculum_learning import CurriculumSampler, TimestepEMASampler
from dataset_sorter.models import TrainingConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CURRICULUM SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurriculumSampler:
    """Tests for loss-based adaptive image sampling."""

    def test_initialization(self):
        sampler = CurriculumSampler(num_images=100)
        assert sampler.num_images == 100
        assert sampler._active is False
        assert sampler._epoch == 0
        assert len(sampler._loss_ema) == 100

    def test_uniform_before_warmup(self):
        """Before warmup, weights should be uniform."""
        sampler = CurriculumSampler(num_images=10, warmup_epochs=2)
        weights = sampler.get_sampling_weights()
        expected = np.ones(10) / 10
        np.testing.assert_allclose(weights, expected, atol=1e-6)

    def test_update_loss(self):
        """Loss EMA should update correctly."""
        sampler = CurriculumSampler(num_images=5, momentum=0.5)
        sampler.update_loss([0], [2.0])
        assert sampler._loss_ema[0] == pytest.approx(2.0)
        assert sampler._seen_count[0] == 1

        sampler.update_loss([0], [4.0])
        # EMA: 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        assert sampler._loss_ema[0] == pytest.approx(3.0)
        assert sampler._seen_count[0] == 2

    def test_activation_after_warmup(self):
        """Curriculum should activate after warmup epochs with enough data."""
        sampler = CurriculumSampler(num_images=10, warmup_epochs=1)

        # See all images
        for i in range(10):
            sampler.update_loss([i], [float(i)])

        # First epoch: warmup
        sampler.on_epoch_start()
        assert sampler._active is False

        # Second epoch: should activate
        sampler.on_epoch_start()
        assert sampler._active is True

    def test_high_loss_gets_higher_weight(self):
        """Images with higher loss should get higher sampling weight."""
        sampler = CurriculumSampler(num_images=4, warmup_epochs=0, temperature=1.0)

        # Give different losses
        sampler.update_loss([0], [0.1])
        sampler.update_loss([1], [1.0])
        sampler.update_loss([2], [5.0])
        sampler.update_loss([3], [10.0])

        # Activate
        sampler._active = True

        weights = sampler.get_sampling_weights()
        assert weights[3] > weights[2] > weights[1] > weights[0]

    def test_temperature_zero_is_uniform(self):
        """Temperature=0 should give uniform weights."""
        sampler = CurriculumSampler(num_images=5, temperature=0.0)
        sampler._active = True
        sampler.update_loss([0, 1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0, 5.0])

        weights = sampler.get_sampling_weights()
        expected = np.ones(5) / 5
        np.testing.assert_allclose(weights, expected, atol=1e-6)

    def test_high_temperature_sharpens(self):
        """Higher temperature should amplify differences."""
        sampler_low = CurriculumSampler(num_images=3, temperature=1.0)
        sampler_high = CurriculumSampler(num_images=3, temperature=3.0)

        for s in [sampler_low, sampler_high]:
            s._active = True
            s.update_loss([0, 1, 2], [1.0, 2.0, 10.0])

        w_low = sampler_low.get_sampling_weights()
        w_high = sampler_high.get_sampling_weights()

        # High temp should give even higher weight to the hard example
        ratio_low = w_low[2] / w_low[0]
        ratio_high = w_high[2] / w_high[0]
        assert ratio_high > ratio_low

    def test_min_weight_prevents_starvation(self):
        """Min weight should prevent any image from being completely ignored."""
        sampler = CurriculumSampler(num_images=3, min_weight=0.1, temperature=1.0)
        sampler._active = True
        sampler.update_loss([0], [0.001])  # Very easy
        sampler.update_loss([1], [100.0])  # Very hard
        sampler.update_loss([2], [50.0])

        weights = sampler.get_sampling_weights()
        assert weights[0] > 0  # Easy image still has nonzero weight

    def test_sample_indices(self):
        """sample_indices should return valid indices."""
        sampler = CurriculumSampler(num_images=10)
        indices = sampler.sample_indices(5)
        assert len(indices) == 5
        assert all(0 <= i < 10 for i in indices)

    def test_sample_indices_biased(self):
        """With curriculum active, hard examples should appear more often."""
        sampler = CurriculumSampler(num_images=3, warmup_epochs=0, temperature=2.0)
        sampler._active = True
        sampler.update_loss([0], [0.01])  # Easy
        sampler.update_loss([1], [0.01])  # Easy
        sampler.update_loss([2], [100.0]) # Hard

        rng = np.random.default_rng(42)
        indices = sampler.sample_indices(1000, rng=rng)
        counts = [indices.count(i) for i in range(3)]

        # Hard example (index 2) should be sampled much more than easy ones
        assert counts[2] > counts[0]
        assert counts[2] > counts[1]

    def test_get_stats(self):
        """Stats should return a valid dictionary."""
        sampler = CurriculumSampler(num_images=5)
        sampler.update_loss([0, 1], [1.0, 2.0])
        stats = sampler.get_stats()
        assert "active" in stats
        assert "loss_mean" in stats
        assert "weight_ratio" in stats
        assert stats["images_seen"] == 2

    def test_out_of_range_indices_ignored(self):
        """Indices outside valid range should be silently ignored."""
        sampler = CurriculumSampler(num_images=5)
        sampler.update_loss([-1, 5, 100], [1.0, 2.0, 3.0])
        assert sampler._seen_count.sum() == 0  # No valid updates


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TIMESTEP EMA SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimestepEMASampler:
    """Tests for per-timestep-bucket loss EMA and adaptive sampling."""

    def test_initialization(self):
        sampler = TimestepEMASampler(num_train_timesteps=1000, num_buckets=20)
        assert sampler.num_buckets == 20
        assert sampler.bucket_size == 50
        assert sampler._step == 0
        assert len(sampler._loss_ema) == 20

    def test_timestep_to_bucket(self):
        """Timesteps should map to correct buckets."""
        sampler = TimestepEMASampler(num_train_timesteps=1000, num_buckets=10)
        t = torch.tensor([0, 50, 99, 100, 999])
        buckets = sampler._timestep_to_bucket(t)
        assert buckets[0].item() == 0   # 0-99 → bucket 0
        assert buckets[1].item() == 0   # 50 → bucket 0
        assert buckets[2].item() == 0   # 99 → bucket 0
        assert buckets[3].item() == 1   # 100 → bucket 1
        assert buckets[4].item() == 9   # 999 → bucket 9

    def test_uniform_during_warmup(self):
        """During warmup, sampling should be uniform."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=10, warmup_steps=50,
        )
        timesteps = sampler.sample_timesteps(1000)
        assert timesteps.min() >= 0
        assert timesteps.max() < 100

    def test_update_loss_ema(self):
        """Loss EMA should update per bucket."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=10, momentum=0.5,
        )
        # Update bucket 0 (timesteps 0-9)
        sampler.update(torch.tensor([5]), torch.tensor([2.0]))
        assert sampler._loss_ema[0].item() == pytest.approx(2.0)
        assert sampler._seen_count[0].item() == 1

        sampler.update(torch.tensor([5]), torch.tensor([4.0]))
        # EMA: 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        assert sampler._loss_ema[0].item() == pytest.approx(3.0)

    def test_low_loss_buckets_get_downweighted(self):
        """Buckets with low loss should have lower sampling probability."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=5,
            warmup_steps=0, skip_threshold=0.3,
        )

        # High loss in bucket 0, low loss in bucket 4
        for _ in range(10):
            sampler.update(torch.tensor([5]), torch.tensor([10.0]))   # bucket 0
            sampler.update(torch.tensor([95]), torch.tensor([0.1]))   # bucket 4
            sampler.update(torch.tensor([45]), torch.tensor([5.0]))   # bucket 2

        weights = sampler._compute_bucket_weights()
        # Bucket with low loss should have lower weight
        assert weights[4] < weights[0]

    def test_sample_timesteps_after_warmup(self):
        """After warmup, sampling should bias toward high-loss buckets."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=5,
            warmup_steps=5, skip_threshold=0.3,
        )

        # Warmup steps
        for i in range(6):
            sampler.update(torch.tensor([5]), torch.tensor([10.0]))  # high loss bucket 0
            sampler.update(torch.tensor([95]), torch.tensor([0.1])) # low loss bucket 4

        timesteps = sampler.sample_timesteps(1000)
        assert timesteps.min() >= 0
        assert timesteps.max() < 100

    def test_compute_loss_weights(self):
        """Loss weights should upweight high-loss buckets."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=5,
            warmup_steps=0,
        )

        # Set up different losses per bucket
        for _ in range(5):
            sampler.update(torch.tensor([5]), torch.tensor([10.0]))
            sampler.update(torch.tensor([95]), torch.tensor([1.0]))

        weights = sampler.compute_loss_weights(torch.tensor([5, 95]))
        # High-loss bucket should get higher weight
        assert weights[0] > weights[1]

    def test_loss_weights_uniform_during_warmup(self):
        """During warmup, loss weights should be uniform (all 1.0)."""
        sampler = TimestepEMASampler(warmup_steps=100)
        weights = sampler.compute_loss_weights(torch.tensor([50, 500]))
        assert torch.allclose(weights, torch.ones(2))

    def test_loss_weights_normalized(self):
        """Loss weights should have mean ≈ 1.0."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=5,
            warmup_steps=0,
        )
        for _ in range(10):
            sampler.update(torch.tensor([5, 25, 45, 65, 85]),
                          torch.tensor([1.0, 2.0, 5.0, 8.0, 10.0]))

        weights = sampler.compute_loss_weights(torch.tensor([5, 25, 45, 65, 85]))
        assert weights.mean().item() == pytest.approx(1.0, abs=0.1)

    def test_get_stats(self):
        """Stats should return a valid dictionary."""
        sampler = TimestepEMASampler(num_train_timesteps=100, num_buckets=10)
        sampler.update(torch.tensor([50]), torch.tensor([1.0]))
        stats = sampler.get_stats()
        assert "step" in stats
        assert "buckets_seen" in stats
        assert stats["buckets_seen"] == 1
        assert stats["active"] is False  # Still in warmup

    def test_bucket_clamping(self):
        """Timesteps at boundaries should map correctly."""
        sampler = TimestepEMASampler(num_train_timesteps=1000, num_buckets=20)
        t = torch.tensor([0, 999, 1000])  # 1000 is out of range
        buckets = sampler._timestep_to_bucket(t)
        assert buckets[0].item() == 0
        assert buckets[1].item() == 19
        assert buckets[2].item() == 19  # Clamped


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONFIG FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurriculumConfigFields:
    """Tests for new TrainingConfig fields."""

    def test_curriculum_defaults(self):
        config = TrainingConfig()
        assert config.curriculum_learning is False
        assert config.curriculum_temperature == 1.0
        assert config.curriculum_warmup_epochs == 1

    def test_timestep_ema_defaults(self):
        config = TrainingConfig()
        assert config.timestep_ema_sampling is False
        assert config.timestep_ema_skip_threshold == 0.3
        assert config.timestep_ema_num_buckets == 20

    def test_serialization(self):
        """New fields should serialize correctly."""
        from dataclasses import asdict
        import json

        config = TrainingConfig()
        config.curriculum_learning = True
        config.curriculum_temperature = 2.5
        config.timestep_ema_sampling = True
        config.timestep_ema_skip_threshold = 0.5

        data = asdict(config)
        json_str = json.dumps(data, default=str)
        loaded = json.loads(json_str)

        assert loaded["curriculum_learning"] is True
        assert loaded["curriculum_temperature"] == 2.5
        assert loaded["timestep_ema_sampling"] is True
        assert loaded["timestep_ema_skip_threshold"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurriculumIntegration:
    """Integration tests for both features working together."""

    def test_curriculum_with_timestep_ema(self):
        """Both features should work simultaneously without conflict."""
        curriculum = CurriculumSampler(num_images=10, warmup_epochs=0)
        timestep_ema = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=5, warmup_steps=0,
        )

        # Simulate a training loop
        for step in range(20):
            # Curriculum: update per-image losses
            img_idx = step % 10
            loss_val = float(img_idx + 1)  # Higher idx = harder
            curriculum.update_loss([img_idx], [loss_val])

            # Timestep EMA: update per-timestep losses
            t = torch.randint(0, 100, (1,))
            timestep_ema.update(t, torch.tensor([loss_val]))

        curriculum._active = True
        weights = curriculum.get_sampling_weights()
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)

        ts = timestep_ema.sample_timesteps(10)
        assert ts.min() >= 0
        assert ts.max() < 100

    def test_epoch_progression(self):
        """Curriculum should progress through warmup correctly."""
        sampler = CurriculumSampler(num_images=5, warmup_epochs=2)

        for i in range(5):
            sampler.update_loss([i], [float(i)])

        sampler.on_epoch_start()  # epoch 1 (warmup)
        assert not sampler._active

        sampler.on_epoch_start()  # epoch 2 (warmup)
        assert not sampler._active

        sampler.on_epoch_start()  # epoch 3 (activate)
        assert sampler._active

    def test_timestep_ema_diverse_buckets(self):
        """All buckets should be reachable via sampling."""
        sampler = TimestepEMASampler(
            num_train_timesteps=100, num_buckets=5, warmup_steps=0,
        )
        # See all buckets
        for b in range(5):
            t = torch.tensor([b * 20 + 10])
            sampler.update(t, torch.tensor([1.0]))

        # Sample many timesteps
        ts = sampler.sample_timesteps(10000)
        buckets_seen = set()
        for t in ts:
            buckets_seen.add((t.item() // 20))
        assert len(buckets_seen) == 5  # All buckets should be reachable
