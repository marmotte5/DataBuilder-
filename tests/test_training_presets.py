"""Tests for training presets, ControlNet/DPO configs, checkpointing, and custom schedulers."""

import math
import pytest

from dataset_sorter.training_presets import (
    TRAINING_PRESETS,
    get_preset_names,
    get_preset_labels,
    apply_preset,
    ControlNetConfig,
    DPOConfig,
    AdversarialConfig,
    CheckpointConfig,
    CONTROLNET_TYPES,
    DPO_LOSS_TYPES,
    CHECKPOINT_GRANULARITY,
    SchedulerSegment,
    CustomSchedulerConfig,
    build_custom_lr_lambda,
    CUSTOM_SCHEDULES,
)
from dataset_sorter.models import TrainingConfig


# ── Presets ─────────────────────────────────────────────────────────

class TestPresets:
    def test_preset_names(self):
        names = get_preset_names()
        assert len(names) == 12  # 7 standard + 5 turbo/distilled model presets
        assert "character_lora" in names
        assert "style_lora" in names
        assert "dpo_preference" in names
        assert "controlnet" in names

    def test_preset_labels(self):
        labels = get_preset_labels()
        assert labels["character_lora"] == "Character LoRA"
        assert labels["fast_test"] == "Quick Test (5 min)"

    def test_all_presets_have_required_keys(self):
        for key, preset in TRAINING_PRESETS.items():
            assert "label" in preset, f"{key} missing label"
            assert "description" in preset, f"{key} missing description"
            assert "config" in preset, f"{key} missing config"
            assert isinstance(preset["config"], dict)

    def test_apply_preset_character(self):
        config = TrainingConfig()
        config = apply_preset(config, "character_lora")
        assert config.lora_rank == 32
        assert config.learning_rate == 1e-4
        assert config.tag_shuffle is True
        assert config.keep_first_n_tags == 1
        assert config.epochs == 15

    def test_apply_preset_style(self):
        config = TrainingConfig()
        config = apply_preset(config, "style_lora")
        assert config.lora_rank == 64
        assert config.use_dora is True
        assert config.keep_first_n_tags == 0

    def test_apply_preset_fast_test(self):
        config = TrainingConfig()
        config = apply_preset(config, "fast_test")
        assert config.lora_rank == 8
        assert config.epochs == 3
        assert config.train_text_encoder is False
        assert config.save_every_n_steps == 0

    def test_apply_preset_unknown(self):
        config = TrainingConfig()
        original_lr = config.learning_rate
        config = apply_preset(config, "nonexistent_preset")
        assert config.learning_rate == original_lr  # unchanged

    def test_apply_preset_preserves_unset_fields(self):
        config = TrainingConfig()
        config.vram_gb = 48
        config = apply_preset(config, "character_lora")
        assert config.vram_gb == 48  # not overwritten by preset

    def test_preset_config_fields_exist_on_training_config(self):
        for key, preset in TRAINING_PRESETS.items():
            for field_name in preset["config"]:
                assert hasattr(TrainingConfig, field_name), (
                    f"Preset {key} references unknown field: {field_name}"
                )


# ── ControlNet Config ─────────────────────────────────────────────

class TestControlNetConfig:
    def test_defaults(self):
        cn = ControlNetConfig()
        assert cn.conditioning_type == "canny"
        assert cn.conditioning_scale == 1.0
        assert cn.train_controlnet_from_scratch is True

    def test_controlnet_types(self):
        assert len(CONTROLNET_TYPES) >= 8
        assert "canny" in CONTROLNET_TYPES
        assert "depth" in CONTROLNET_TYPES
        assert "pose" in CONTROLNET_TYPES


# ── DPO / Adversarial Config ────────────────────────────────────

class TestDPOConfig:
    def test_defaults(self):
        dpo = DPOConfig()
        assert dpo.beta == 0.1
        assert dpo.loss_type == "sigmoid"
        assert dpo.label_smoothing == 0.0

    def test_loss_types(self):
        assert len(DPO_LOSS_TYPES) == 3
        assert "sigmoid" in DPO_LOSS_TYPES
        assert "hinge" in DPO_LOSS_TYPES
        assert "ipo" in DPO_LOSS_TYPES

    def test_adversarial_defaults(self):
        adv = AdversarialConfig()
        assert adv.discriminator_lr == 1e-4
        assert adv.feature_matching is True
        assert adv.discriminator_start_step == 100


# ── Checkpoint Granularity ──────────────────────────────────────

class TestCheckpointConfig:
    def test_defaults(self):
        ckpt = CheckpointConfig()
        assert ckpt.mode == "full"
        assert ckpt.every_n == 2

    def test_granularity_options(self):
        assert len(CHECKPOINT_GRANULARITY) == 4
        assert "none" in CHECKPOINT_GRANULARITY
        assert "full" in CHECKPOINT_GRANULARITY
        assert "selective" in CHECKPOINT_GRANULARITY
        assert "every_n" in CHECKPOINT_GRANULARITY


# ── Custom LR Scheduler ────────────────────────────────────────

class TestCustomScheduler:
    def test_linear_segment(self):
        config = CustomSchedulerConfig(
            segments=[
                SchedulerSegment(start_step=0, end_step=100,
                                 start_lr_ratio=0.0, end_lr_ratio=1.0, type="linear"),
            ],
            base_lr=1e-4,
            total_steps=100,
        )
        lr_fn = build_custom_lr_lambda(config)
        assert lr_fn(0) == pytest.approx(0.0)
        assert lr_fn(50) == pytest.approx(0.5)
        assert lr_fn(100) == pytest.approx(1.0)

    def test_cosine_segment(self):
        config = CustomSchedulerConfig(
            segments=[
                SchedulerSegment(start_step=0, end_step=100,
                                 start_lr_ratio=1.0, end_lr_ratio=0.0, type="cosine"),
            ],
            base_lr=1e-4,
            total_steps=100,
        )
        lr_fn = build_custom_lr_lambda(config)
        assert lr_fn(0) == pytest.approx(1.0)
        assert lr_fn(50) == pytest.approx(0.5, abs=0.01)
        assert lr_fn(100) == pytest.approx(0.0, abs=0.01)

    def test_constant_segment(self):
        config = CustomSchedulerConfig(
            segments=[
                SchedulerSegment(start_step=0, end_step=100,
                                 start_lr_ratio=0.5, end_lr_ratio=0.5, type="constant"),
            ],
            base_lr=1e-4,
            total_steps=100,
        )
        lr_fn = build_custom_lr_lambda(config)
        assert lr_fn(0) == pytest.approx(0.5)
        assert lr_fn(50) == pytest.approx(0.5)
        assert lr_fn(100) == pytest.approx(0.5)

    def test_multi_segment(self):
        config = CustomSchedulerConfig(
            segments=[
                SchedulerSegment(start_step=0, end_step=50,
                                 start_lr_ratio=0.0, end_lr_ratio=1.0, type="linear"),
                SchedulerSegment(start_step=50, end_step=100,
                                 start_lr_ratio=1.0, end_lr_ratio=0.0, type="linear"),
            ],
            base_lr=1e-4,
            total_steps=100,
        )
        lr_fn = build_custom_lr_lambda(config)
        assert lr_fn(0) == pytest.approx(0.0)
        assert lr_fn(25) == pytest.approx(0.5)
        assert lr_fn(50) == pytest.approx(1.0)
        assert lr_fn(75) == pytest.approx(0.5)
        assert lr_fn(100) == pytest.approx(0.0)

    def test_after_all_segments(self):
        config = CustomSchedulerConfig(
            segments=[
                SchedulerSegment(start_step=0, end_step=50,
                                 start_lr_ratio=1.0, end_lr_ratio=0.3, type="linear"),
            ],
            base_lr=1e-4,
            total_steps=100,
        )
        lr_fn = build_custom_lr_lambda(config)
        # After all segments end, use last segment's end ratio
        assert lr_fn(80) == pytest.approx(0.3)

    def test_empty_segments(self):
        config = CustomSchedulerConfig(segments=[], base_lr=1e-4, total_steps=100)
        lr_fn = build_custom_lr_lambda(config)
        assert lr_fn(0) == pytest.approx(1.0)
        assert lr_fn(50) == pytest.approx(1.0)

    def test_predefined_schedules_valid(self):
        assert len(CUSTOM_SCHEDULES) >= 3
        for key, sched in CUSTOM_SCHEDULES.items():
            assert "label" in sched
            assert "description" in sched
            assert "segments" in sched
            assert len(sched["segments"]) >= 2

    def test_warmup_cosine_cooldown_schedule(self):
        sched = CUSTOM_SCHEDULES["warmup_cosine_cooldown"]
        segments = [SchedulerSegment(**s) for s in sched["segments"]]
        config = CustomSchedulerConfig(segments=segments, base_lr=1e-4, total_steps=1000)
        lr_fn = build_custom_lr_lambda(config)
        # Warmup: starts low
        assert lr_fn(0) == pytest.approx(0.01)
        # Peak at warmup end
        assert lr_fn(100) == pytest.approx(1.0)
        # Cooldown: ends low
        assert lr_fn(1000) == pytest.approx(0.01)

    def test_step_decay_schedule(self):
        sched = CUSTOM_SCHEDULES["step_decay"]
        segments = [SchedulerSegment(**s) for s in sched["segments"]]
        config = CustomSchedulerConfig(segments=segments, base_lr=1e-4, total_steps=1000)
        lr_fn = build_custom_lr_lambda(config)
        # First plateau
        assert lr_fn(100) == pytest.approx(1.0)
        # Second plateau
        assert lr_fn(500) == pytest.approx(0.3)
        # Third plateau
        assert lr_fn(800) == pytest.approx(0.1)
