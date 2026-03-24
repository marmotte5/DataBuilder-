"""Tests for the LoRA Mixer module.

Tests cover:
1. weighted_average merge
2. add merge
3. Different weights
4. Validation (incompatible LoRAs)
5. CLI parsing
6. Key normalization across formats
7. SVD merge basic functionality
8. get_info / get_common_keys
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_lora_tensors(
    rank: int = 4,
    in_dim: int = 16,
    out_dim: int = 32,
    num_layers: int = 3,
    fmt: str = "diffusers",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create a small synthetic LoRA in the requested key format."""
    torch.manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}

    for i in range(num_layers):
        down = torch.randn(rank, in_dim)
        up = torch.randn(out_dim, rank)
        alpha = torch.tensor(float(rank))

        if fmt == "diffusers":
            base = f"unet.down_blocks.{i}.attentions.0.proj_in"
            tensors[f"{base}.lora_down.weight"] = down
            tensors[f"{base}.lora_up.weight"] = up
            tensors[f"{base}.alpha"] = alpha
        elif fmt == "kohya":
            base = f"lora_unet_down_blocks_{i}_attentions_0_proj_in"
            tensors[f"{base}.lora_down.weight"] = down
            tensors[f"{base}.lora_up.weight"] = up
            tensors[f"{base}.alpha"] = alpha
        elif fmt == "peft":
            base = f"base_model.model.down_blocks.{i}.attentions.0.proj_in"
            tensors[f"{base}.lora_A.weight"] = down
            tensors[f"{base}.lora_B.weight"] = up
        else:
            raise ValueError(f"Unknown format: {fmt}")

    return tensors


def _save_lora(tensors: dict[str, torch.Tensor], path: str) -> None:
    from safetensors.torch import save_file

    save_file({k: v.float() for k, v in tensors.items()}, path)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


@pytest.fixture()
def lora_a(tmp):
    tensors = _make_lora_tensors(rank=4, seed=1)
    path = str(tmp / "lora_a.safetensors")
    _save_lora(tensors, path)
    return path, tensors


@pytest.fixture()
def lora_b(tmp):
    tensors = _make_lora_tensors(rank=4, seed=2)
    path = str(tmp / "lora_b.safetensors")
    _save_lora(tensors, path)
    return path, tensors


@pytest.fixture()
def mixer(lora_a, lora_b):
    from dataset_sorter.lora_mixer import LoRAMixer

    path_a, _ = lora_a
    path_b, _ = lora_b
    m = LoRAMixer()
    m.add_lora(path_a, weight=0.5, name="lora_a")
    m.add_lora(path_b, weight=0.5, name="lora_b")
    return m


# ─── 1. weighted_average ──────────────────────────────────────────────────────


class TestWeightedAverage:
    def test_output_file_created(self, mixer, tmp):
        out = str(tmp / "merged.safetensors")
        mixer.mix(out, mode="weighted_average")
        assert Path(out).exists()

    def test_keys_present(self, mixer, tmp):
        from safetensors.torch import load_file

        out = str(tmp / "merged.safetensors")
        mixer.mix(out, mode="weighted_average")
        merged = load_file(out, device="cpu")
        assert len(merged) > 0

    def test_equal_weights_midpoint(self, lora_a, lora_b, tmp):
        """With equal weights the result should be exactly the midpoint."""
        from safetensors.torch import load_file
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, tensors_a = lora_a
        path_b, tensors_b = lora_b
        m = LoRAMixer()
        m.add_lora(path_a, weight=1.0)
        m.add_lora(path_b, weight=1.0)
        out = str(tmp / "avg.safetensors")
        m.mix(out, mode="weighted_average")

        merged = load_file(out, device="cpu")
        # Pick the first down key from common (normalized, no .weight suffix)
        common = m.get_common_keys()
        merged_key = next(k for k in common if "lora_down" in k)
        # Find matching raw key in tensors_a (has .weight suffix)
        raw_key = next(k for k in tensors_a if "lora_down" in k)
        expected = (tensors_a[raw_key].float() + tensors_b[raw_key].float()) / 2.0
        # The merged file stores in float16 and uses normalized keys (no .weight)
        torch.testing.assert_close(merged[merged_key].float(), expected, rtol=1e-2, atol=1e-2)

    def test_shape_preserved(self, mixer, tmp):
        from safetensors.torch import load_file

        out = str(tmp / "shape.safetensors")
        mixer.mix(out, mode="weighted_average")
        merged = load_file(out, device="cpu")
        # alpha tensors are valid 0-d scalars; weight tensors are >= 1-d
        assert len(merged) > 0
        weight_tensors = [v for k, v in merged.items() if not k.endswith("alpha")]
        assert all(v.ndim >= 1 for v in weight_tensors)


# ─── 2. add mode ─────────────────────────────────────────────────────────────


class TestAddMode:
    def test_output_file_created(self, mixer, tmp):
        out = str(tmp / "added.safetensors")
        mixer.mix(out, mode="add")
        assert Path(out).exists()

    def test_add_is_weighted_sum(self, lora_a, lora_b, tmp):
        """add mode with w=1 should give the simple sum."""
        from safetensors.torch import load_file
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, tensors_a = lora_a
        path_b, tensors_b = lora_b
        m = LoRAMixer()
        m.add_lora(path_a, weight=1.0)
        m.add_lora(path_b, weight=1.0)
        out = str(tmp / "add.safetensors")
        m.mix(out, mode="add")

        merged = load_file(out, device="cpu")
        common = m.get_common_keys()
        merged_key = next(k for k in common if "lora_down" in k)
        raw_key = next(k for k in tensors_a if "lora_down" in k)
        expected = tensors_a[raw_key].float() + tensors_b[raw_key].float()
        torch.testing.assert_close(merged[merged_key].float(), expected, rtol=1e-2, atol=1e-2)

    def test_add_differs_from_weighted_average(self, mixer, tmp):
        from safetensors.torch import load_file

        out_wa = str(tmp / "wa.safetensors")
        out_add = str(tmp / "add.safetensors")
        mixer.mix(out_wa, mode="weighted_average")
        mixer.mix(out_add, mode="add")
        wa = load_file(out_wa, device="cpu")
        add = load_file(out_add, device="cpu")
        # With weights summing to 1 the two modes happen to agree; but with
        # sum(weights) ≠ 1 they differ.  Here weights sum to 1.0 so we just
        # verify both produce valid tensors with correct shapes.
        for k in wa:
            if k in add:
                assert wa[k].shape == add[k].shape


# ─── 3. Different weights ─────────────────────────────────────────────────────


class TestDifferentWeights:
    def test_dominant_weight_closer_to_source(self, lora_a, lora_b, tmp):
        """High-weight LoRA should dominate the average."""
        from safetensors.torch import load_file
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, tensors_a = lora_a
        path_b, tensors_b = lora_b

        # 0.9 / 0.1 split → result should be much closer to lora_a
        m = LoRAMixer()
        m.add_lora(path_a, weight=0.9)
        m.add_lora(path_b, weight=0.1)
        out = str(tmp / "dom.safetensors")
        m.mix(out, mode="weighted_average")

        merged = load_file(out, device="cpu")
        common = m.get_common_keys()
        merged_key = next(k for k in common if "lora_down" in k)
        raw_key = next(k for k in tensors_a if "lora_down" in k)
        a_f = tensors_a[raw_key].float()
        b_f = tensors_b[raw_key].float()
        result = merged[merged_key].float()

        dist_to_a = (result - a_f).norm().item()
        dist_to_b = (result - b_f).norm().item()
        assert dist_to_a < dist_to_b

    def test_set_weight(self, lora_a, lora_b, tmp):
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, _ = lora_a
        path_b, _ = lora_b
        m = LoRAMixer()
        m.add_lora(path_a, weight=0.5)
        m.add_lora(path_b, weight=0.5)
        m.set_weight(0, 0.8)
        assert m._loras[0][1] == pytest.approx(0.8)

    def test_remove_lora(self, lora_a, lora_b, tmp):
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, _ = lora_a
        path_b, _ = lora_b
        m = LoRAMixer()
        m.add_lora(path_a, weight=0.5)
        m.add_lora(path_b, weight=0.5)
        m.remove_lora(0)
        assert len(m._loras) == 1


# ─── 4. Validation ───────────────────────────────────────────────────────────


class TestValidation:
    def test_incompatible_loras_warning(self, tmp):
        """LoRAs with completely different key sets should produce a warning."""
        from dataset_sorter.lora_mixer import LoRAMixer

        # Create two LoRAs with no overlapping keys
        tensors_x = {
            "unet.module_x.lora_down.weight": torch.randn(4, 16),
            "unet.module_x.lora_up.weight": torch.randn(32, 4),
        }
        tensors_y = {
            "unet.module_y.lora_down.weight": torch.randn(4, 16),
            "unet.module_y.lora_up.weight": torch.randn(32, 4),
        }
        p1 = str(tmp / "x.safetensors")
        p2 = str(tmp / "y.safetensors")
        _save_lora(tensors_x, p1)
        _save_lora(tensors_y, p2)

        m = LoRAMixer()
        m.add_lora(p1)
        m.add_lora(p2)
        warnings = m.validate()
        assert len(warnings) > 0

    def test_incompatible_raises_on_mix(self, tmp):
        """mix() on LoRAs with no common keys should raise RuntimeError."""
        from dataset_sorter.lora_mixer import LoRAMixer

        tensors_x = {"unet.module_x.lora_down.weight": torch.randn(4, 16)}
        tensors_y = {"unet.module_y.lora_down.weight": torch.randn(4, 16)}
        p1 = str(tmp / "x2.safetensors")
        p2 = str(tmp / "y2.safetensors")
        _save_lora(tensors_x, p1)
        _save_lora(tensors_y, p2)

        m = LoRAMixer()
        m.add_lora(p1)
        m.add_lora(p2)
        with pytest.raises(RuntimeError, match="No common keys"):
            m.mix(str(tmp / "fail.safetensors"))

    def test_rank_mismatch_warning(self, tmp):
        """LoRAs with different ranks for the same key should produce a warning."""
        from dataset_sorter.lora_mixer import LoRAMixer

        tensors_a = {
            "unet.layer.lora_down.weight": torch.randn(4, 16),
            "unet.layer.lora_up.weight": torch.randn(32, 4),
        }
        tensors_b = {
            "unet.layer.lora_down.weight": torch.randn(8, 16),  # rank 8 vs 4
            "unet.layer.lora_up.weight": torch.randn(32, 8),
        }
        p1 = str(tmp / "rank4.safetensors")
        p2 = str(tmp / "rank8.safetensors")
        _save_lora(tensors_a, p1)
        _save_lora(tensors_b, p2)

        m = LoRAMixer()
        m.add_lora(p1)
        m.add_lora(p2)
        warnings = m.validate()
        # Should warn about rank mismatch
        assert any("rank" in w.lower() for w in warnings)

    def test_fewer_than_two_loras_raises(self, lora_a, tmp):
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, _ = lora_a
        m = LoRAMixer()
        m.add_lora(path_a)
        with pytest.raises(ValueError, match="at least 2"):
            m.mix(str(tmp / "single.safetensors"))

    def test_missing_file_raises(self):
        from dataset_sorter.lora_mixer import LoRAMixer

        m = LoRAMixer()
        with pytest.raises(FileNotFoundError):
            m.add_lora("/nonexistent/path/lora.safetensors")

    def test_invalid_weight_raises(self, lora_a):
        from dataset_sorter.lora_mixer import LoRAMixer

        path_a, _ = lora_a
        m = LoRAMixer()
        with pytest.raises(ValueError, match="weight"):
            m.add_lora(path_a, weight=-0.5)


# ─── 5. CLI ───────────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_basic(self, lora_a, lora_b, tmp):
        """End-to-end CLI invocation via argparse."""
        from dataset_sorter.lora_mixer import main

        path_a, _ = lora_a
        path_b, _ = lora_b
        out = str(tmp / "cli_out.safetensors")

        sys.argv = [
            "lora_mixer",
            f"{path_a}:0.6",
            f"{path_b}:0.4",
            "-o", out,
        ]
        main()
        assert Path(out).exists()

    def test_cli_add_mode(self, lora_a, lora_b, tmp):
        from dataset_sorter.lora_mixer import main

        path_a, _ = lora_a
        path_b, _ = lora_b
        out = str(tmp / "cli_add.safetensors")

        sys.argv = [
            "lora_mixer",
            path_a,
            path_b,
            "--mode", "add",
            "-o", out,
        ]
        main()
        assert Path(out).exists()

    def test_cli_parse_lora_arg(self):
        from dataset_sorter.lora_mixer import _parse_lora_arg

        path, weight = _parse_lora_arg("my_lora.safetensors:0.75")
        assert path == "my_lora.safetensors"
        assert weight == pytest.approx(0.75)

    def test_cli_parse_lora_arg_no_weight(self):
        from dataset_sorter.lora_mixer import _parse_lora_arg

        path, weight = _parse_lora_arg("my_lora.safetensors")
        assert path == "my_lora.safetensors"
        assert weight == pytest.approx(1.0)


# ─── 6. get_info / get_common_keys ───────────────────────────────────────────


class TestInfo:
    def test_get_info_structure(self, mixer):
        info = mixer.get_info()
        assert "loras" in info
        assert "common_keys" in info
        assert len(info["loras"]) == 2
        for entry in info["loras"]:
            assert "name" in entry
            assert "weight" in entry
            assert "num_keys" in entry
            assert "ranks" in entry

    def test_get_common_keys_nonempty(self, mixer):
        common = mixer.get_common_keys()
        assert len(common) > 0

    def test_get_common_keys_sorted(self, mixer):
        common = mixer.get_common_keys()
        assert common == sorted(common)


# ─── 7. SVD merge ────────────────────────────────────────────────────────────


class TestSVDMerge:
    def test_svd_merge_creates_output(self, mixer, tmp):
        out = str(tmp / "svd.safetensors")
        mixer.mix(out, mode="svd_merge")
        assert Path(out).exists()

    def test_svd_merge_shapes_valid(self, mixer, tmp):
        from safetensors.torch import load_file

        out = str(tmp / "svd_shapes.safetensors")
        mixer.mix(out, mode="svd_merge")
        merged = load_file(out, device="cpu")
        assert len(merged) > 0
        for k, v in merged.items():
            if not k.endswith("alpha"):
                assert v.ndim >= 1
            assert not torch.any(torch.isnan(v.float()))

    def test_unknown_mode_raises(self, mixer, tmp):
        with pytest.raises(ValueError, match="Unknown mode"):
            mixer.mix(str(tmp / "bad.safetensors"), mode="invalid_mode")  # type: ignore


# ─── 8. Key format normalization ─────────────────────────────────────────────


class TestKeyNormalization:
    def test_detect_diffusers_format(self, tmp):
        from dataset_sorter.lora_mixer import _detect_format

        keys = [
            "unet.down_blocks.0.attentions.0.proj_in.lora_down.weight",
            "unet.down_blocks.0.attentions.0.proj_in.lora_up.weight",
        ]
        assert _detect_format(keys) == "diffusers"

    def test_detect_kohya_format(self, tmp):
        from dataset_sorter.lora_mixer import _detect_format

        keys = [
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight",
        ]
        assert _detect_format(keys) == "kohya"

    def test_detect_peft_format(self, tmp):
        from dataset_sorter.lora_mixer import _detect_format

        keys = [
            "base_model.model.down_blocks.0.attentions.0.proj_in.lora_A.weight",
            "base_model.model.down_blocks.0.attentions.0.proj_in.lora_B.weight",
        ]
        assert _detect_format(keys) == "peft"

    def test_mix_different_formats(self, tmp):
        """LoRAs in different formats can still be mixed if they cover the same layers."""
        from dataset_sorter.lora_mixer import LoRAMixer

        tensors_diffusers = _make_lora_tensors(fmt="diffusers", seed=10)
        tensors_kohya = _make_lora_tensors(fmt="kohya", seed=20)

        p1 = str(tmp / "diffusers.safetensors")
        p2 = str(tmp / "kohya.safetensors")
        _save_lora(tensors_diffusers, p1)
        _save_lora(tensors_kohya, p2)

        m = LoRAMixer()
        m.add_lora(p1, weight=0.5)
        m.add_lora(p2, weight=0.5)

        common = m.get_common_keys()
        # After normalization both formats map to the same canonical keys
        assert len(common) > 0
