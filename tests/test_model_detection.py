"""Smoke tests for the centralized model detection module.

Covers the consolidation of three previously duplicated detectors
(generate_worker, model_scanner, model_library) into one place. Tests
both the keys-based and filename-based detection plus the LoRA
fingerprinting that lives alongside.
"""

from __future__ import annotations

import pytest

from dataset_sorter.model_detection import (
    detect_arch_from_filename,
    detect_arch_from_keys,
    detect_arch_from_path,
    detect_distillation_from_filename,
    detect_lora_arch_from_keys,
)


# ─────────────────────────────────────────────────────────────────────────
# Key-based detection — full pipeline checkpoints
# ─────────────────────────────────────────────────────────────────────────


class TestDetectArchFromKeysFullPipeline:
    def test_sdxl_detected_from_dual_clip_encoders(self):
        keys = [
            "model.diffusion_model.input_blocks.0.weight",
            "conditioner.embedders.0.transformer.weight",
            "conditioner.embedders.1.transformer.weight",
        ]
        assert detect_arch_from_keys(keys) == "sdxl"

    def test_sd2_detected_from_open_clip_subkey(self):
        keys = [
            "model.diffusion_model.input_blocks.0.weight",
            "cond_stage_model.model.transformer.weight",
        ]
        assert detect_arch_from_keys(keys) == "sd2"

    def test_sd15_detected_from_hf_clip_subkey(self):
        keys = [
            "model.diffusion_model.input_blocks.0.weight",
            "cond_stage_model.transformer.weight",
        ]
        assert detect_arch_from_keys(keys) == "sd15"


# ─────────────────────────────────────────────────────────────────────────
# Key-based detection — DiT / transformer architectures
# ─────────────────────────────────────────────────────────────────────────


class TestDetectArchFromKeysDiT:
    def test_flux_detected_from_double_and_single_blocks(self):
        keys = [
            "double_blocks.0.attn.q_proj.weight",
            "single_blocks.0.linear1.weight",
        ]
        assert detect_arch_from_keys(keys) == "flux"

    def test_flux2_distinguished_by_llm_text_encoder(self):
        keys = [
            "double_blocks.0.attn.q_proj.weight",
            "single_blocks.0.linear1.weight",
            "llm.embed_tokens.weight",
        ]
        assert detect_arch_from_keys(keys) == "flux2"

    def test_sd3_detected_from_joint_blocks(self):
        keys = ["joint_blocks.0.attn.q_proj.weight"]
        assert detect_arch_from_keys(keys) == "sd3"

    def test_sd35_detected_from_joint_transformer_blocks(self):
        keys = ["joint_transformer_blocks.0.attn.q_proj.weight"]
        assert detect_arch_from_keys(keys) == "sd35"

    def test_pixart_detected_from_caption_projection_with_adaln(self):
        keys = [
            "caption_projection.0.weight",
            "adaln_single.linear.weight",
        ]
        assert detect_arch_from_keys(keys) == "pixart"

    def test_sana_detected_from_caption_projection_without_adaln(self):
        keys = ["caption_projection.0.weight"]
        assert detect_arch_from_keys(keys) == "sana"

    def test_zimage_detected_from_qwen3_text_encoder(self):
        keys = ["text_encoder.model.embed_tokens.weight"]
        assert detect_arch_from_keys(keys) == "zimage"

    def test_zimage_detected_from_unprefixed_top_level(self):
        keys = [
            "all_final_layer.weight",
            "cap_embedder.weight",
            "noise_refiner.0.weight",
        ]
        assert detect_arch_from_keys(keys) == "zimage"

    def test_hidream_detected_from_llm_keys(self):
        keys = ["llm.embed_tokens.weight", "transformer.attn.weight"]
        assert detect_arch_from_keys(keys) == "hidream"

    def test_hunyuan_detected_from_unprefixed_markers(self):
        keys = ["pooler.weight", "text_states_proj.weight", "t_block.0.weight"]
        assert detect_arch_from_keys(keys) == "hunyuan"

    def test_kolors_detected_from_chatglm_text_encoder(self):
        keys = ["text_encoder.transformer.word_embeddings.weight"]
        assert detect_arch_from_keys(keys) == "kolors"

    def test_chroma_detected_from_namespace(self):
        keys = ["chroma.transformer.0.weight"]
        assert detect_arch_from_keys(keys) == "chroma"

    def test_animatediff_detected_from_motion_modules(self):
        keys = ["motion_modules.0.attn.weight"]
        assert detect_arch_from_keys(keys) == "animatediff"

    def test_cascade_detected_from_effnet(self):
        keys = [
            "down_blocks.0.weight",
            "up_blocks.0.weight",
            "effnet.0.weight",
        ]
        assert detect_arch_from_keys(keys) == "cascade"


# ─────────────────────────────────────────────────────────────────────────
# Empty / unknown
# ─────────────────────────────────────────────────────────────────────────


def test_empty_keys_returns_empty_string():
    assert detect_arch_from_keys([]) == ""


def test_unknown_keys_returns_empty_string():
    assert detect_arch_from_keys(["weights.0", "something_random"]) == ""


# ─────────────────────────────────────────────────────────────────────────
# Filename detection
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("filename,expected", [
    ("flux2-dev-fp16.safetensors", "flux2"),
    ("flux-schnell.safetensors", "flux"),
    ("sdxl_base_1.0.safetensors", "sdxl"),
    ("sd_xl_base.safetensors", "sdxl"),
    ("sd35_medium.safetensors", "sd35"),
    ("sd3.5-large.safetensors", "sd35"),
    ("sd3-medium.safetensors", "sd3"),
    ("Pony-XL-v6.safetensors", "pony"),
    ("sd_2_1.safetensors", "sd2"),
    ("PixArt-Sigma.safetensors", "pixart"),
    ("Sana-1024.safetensors", "sana"),
    ("Kolors-pretrained.safetensors", "kolors"),
    ("StableCascade.safetensors", "cascade"),
    ("HunyuanDiT.safetensors", "hunyuan"),
    ("auraflow_v0.3.safetensors", "auraflow"),
    ("z-image-turbo.safetensors", "zimage"),
    ("Z_Image.safetensors", "zimage"),
    ("HiDream-1.safetensors", "hidream"),
    ("Chroma-v50.safetensors", "chroma"),
])
def test_filename_keywords(filename, expected):
    """Filename keyword matching covers all 17 supported architectures."""
    assert detect_arch_from_filename(filename) == expected


def test_filename_more_specific_wins():
    """flux2 must match before flux, sd35 before sd3, etc."""
    assert detect_arch_from_filename("flux2-dev.safetensors") == "flux2"
    assert detect_arch_from_filename("sd35-large.safetensors") == "sd35"


def test_filename_no_match_returns_empty():
    assert detect_arch_from_filename("random_model_v1.safetensors") == ""


# ─────────────────────────────────────────────────────────────────────────
# LoRA-specific fingerprinting
# ─────────────────────────────────────────────────────────────────────────


class TestDetectLoraArchFromKeys:
    def test_flux_lora_detected_via_double_blocks(self):
        keys = [
            "lora_unet_double_blocks_0_attn.lora_A.weight",
            "lora_unet_double_blocks_0_attn.lora_B.weight",
        ]
        assert detect_lora_arch_from_keys(keys) == "flux"

    def test_sd3_lora_detected_via_joint_blocks(self):
        keys = [
            "lora_unet_joint_blocks_0_attn.lora_A.weight",
            "lora_unet_joint_blocks_0_attn.lora_B.weight",
        ]
        assert detect_lora_arch_from_keys(keys) == "sd3"

    def test_sdxl_lora_detected_via_te2(self):
        keys = [
            "lora_te2_text_model.lora_A.weight",
            "lora_unet_input.lora_B.weight",
        ]
        assert detect_lora_arch_from_keys(keys) == "sdxl"

    def test_sdxl_lora_detected_via_add_embedding(self):
        keys = [
            "lora_unet_add_embedding.lora_A.weight",
            "lora_unet_add_embedding.lora_B.weight",
        ]
        assert detect_lora_arch_from_keys(keys) == "sdxl"

    def test_sd15_lora_default_when_no_markers(self):
        keys = ["lora_unet_input.lora_A.weight", "lora_unet_input.lora_B.weight"]
        assert detect_lora_arch_from_keys(keys) == "sd15"

    def test_non_lora_returns_empty(self):
        keys = ["transformer.weight", "model.diffusion_model.weight"]
        assert detect_lora_arch_from_keys(keys) == ""


# ─────────────────────────────────────────────────────────────────────────
# Path-level entry point (combines keys + filename fallback)
# ─────────────────────────────────────────────────────────────────────────


def test_detect_arch_from_path_falls_back_to_filename(tmp_path):
    """When keys can't be read, filename pattern matches kick in."""
    # Create a file with the right name but no safetensors content
    p = tmp_path / "flux-schnell-fp8.safetensors"
    p.write_bytes(b"\x00" * 4)  # too short for a valid header
    assert detect_arch_from_path(p) == "flux"


def test_detect_arch_from_path_default_when_unknown(tmp_path):
    """Returns the supplied default when neither keys nor filename match."""
    p = tmp_path / "random_v3.safetensors"
    p.write_bytes(b"\x00" * 4)
    assert detect_arch_from_path(p, default="sdxl") == "sdxl"
    assert detect_arch_from_path(p, default="") == ""


# ─────────────────────────────────────────────────────────────────────────
# Distillation detection — orthogonal to base architecture
# Locks in the auto-CFG=1.0 fix for distilled checkpoints (Lightning, LCM,
# Hyper-SD, SDXL Turbo, Flux Schnell, DMD/DMD2). Without these, a Flux
# Schnell loaded by file path defaults to CFG=3.5 and 28 steps, producing
# saturated outputs.
# ─────────────────────────────────────────────────────────────────────────


class TestDetectDistillation:
    @pytest.mark.parametrize("name,expected", [
        ("sdxl_lightning_8step.safetensors",        "lightning"),
        ("SDXL-Lightning-4step-Lora.safetensors",   "lightning"),
        ("juggernautXL_lightning_v8.safetensors",   "lightning"),
        ("hyperSD_4step.safetensors",               "hyper"),
        ("Hyper-SDXL-1step-Unet.safetensors",       "hyper"),
        ("flux1-schnell-fp8.safetensors",           "schnell"),
        ("flux.1.schnell.safetensors",              "schnell"),
        ("sdxl_turbo_v1.safetensors",               "turbo"),
        ("sd-turbo.safetensors",                    "turbo"),
        ("LCM_Dreamshaper_v7.safetensors",          "lcm"),
        ("anything_lcm_lora.safetensors",           "lcm"),
        ("dmd2_sdxl_4step.safetensors",             "dmd2"),
        ("model_dmd_distilled.safetensors",         "dmd"),
    ])
    def test_distillation_detected(self, name, expected):
        assert detect_distillation_from_filename(name) == expected

    @pytest.mark.parametrize("name", [
        "sdxl_base_1.0.safetensors",
        "flux1-dev.safetensors",
        "juggernautXL_v8.safetensors",
        "sd_xl_base_1.0.safetensors",
        "pony_diffusion_v6.safetensors",
        "",
    ])
    def test_no_false_positives_on_base_models(self, name):
        assert detect_distillation_from_filename(name) == ""

    def test_dmd2_takes_precedence_over_dmd(self):
        # "dmd2" matches before bare "dmd" (more specific patterns first).
        assert detect_distillation_from_filename("model_dmd2.safetensors") == "dmd2"

    def test_arch_and_distillation_are_independent(self):
        """A file can be both 'sdxl' and 'lightning' — each detector
        answers its own question."""
        name = "sdxl_lightning_8step.safetensors"
        assert detect_arch_from_filename(name) == "sdxl"
        assert detect_distillation_from_filename(name) == "lightning"

    def test_distillation_constants_table_complete(self):
        """Every detection tag must have a row in DISTILLED_DEFAULTS or the
        auto-config in generate_worker silently does nothing."""
        from dataset_sorter.constants import DISTILLED_DEFAULTS
        from dataset_sorter.model_detection import _DISTILLATION_KEYWORDS
        for _, tag in _DISTILLATION_KEYWORDS:
            assert tag in DISTILLED_DEFAULTS, (
                f"Distillation tag {tag!r} has no entry in "
                "constants.DISTILLED_DEFAULTS — auto-CFG/steps fix won't apply."
            )

    def test_distilled_defaults_use_cfg_one(self):
        """All distilled variants need CFG=1.0 — anything else is wrong."""
        from dataset_sorter.constants import DISTILLED_DEFAULTS
        for tag, defaults in DISTILLED_DEFAULTS.items():
            assert defaults["cfg_scale"] == 1.0, (
                f"DISTILLED_DEFAULTS[{tag!r}] must use CFG=1.0; distilled "
                "models ignore CFG and produce burnt outputs at CFG>1."
            )
            assert 1 <= defaults["steps"] <= 12, (
                f"DISTILLED_DEFAULTS[{tag!r}] step count {defaults['steps']} "
                "is outside the typical 1-12 range for distilled checkpoints."
            )
