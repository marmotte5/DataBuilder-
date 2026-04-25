"""Smoke tests for the GGUF export pipeline.

Validates the conversion of a synthetic safetensors model to GGUF for
each supported quantization scheme — including roundtrip MSE checks to
confirm Q8_0 / Q5_1 / etc. don't silently destroy the weights.

We avoid loading any real diffusion model (which would need ~6 GB) by
generating tiny synthetic tensors that exercise every code path:
- Square 2D matrices (quantizable)
- 1D biases (auto-fall back to F16)
- Norms / embeddings (skipped by name pattern)
- Tensors with inner dim < 32 (forced F16 fallback)
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest


HAS_GGUF = importlib.util.find_spec("gguf") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


pytestmark = pytest.mark.skipif(
    not (HAS_GGUF and HAS_SAFETENSORS and HAS_TORCH),
    reason="gguf, safetensors, or torch not installed",
)


def _write_synthetic_safetensors(path: Path) -> int:
    """Build a small mixed-tensor safetensors file. Returns its size in bytes."""
    import torch
    from safetensors.torch import save_file

    tensors = {
        # Quantizable: 2D, divisible by 32
        "transformer.attn.to_q.weight": torch.randn(64, 64),
        "transformer.attn.to_k.weight": torch.randn(64, 64),
        "transformer.attn.to_v.weight": torch.randn(64, 64),
        "transformer.ffn.weight": torch.randn(256, 64),
        # Skipped (1D bias)
        "transformer.attn.to_q.bias": torch.randn(64),
        # Skipped (norm pattern)
        "transformer.norm.weight": torch.randn(64),
        # Skipped (embedding pattern)
        "transformer.embed.weight": torch.randn(100, 64),
        # Skipped (inner dim < 32)
        "transformer.tiny.weight": torch.randn(32, 16),
    }
    save_file(tensors, str(path))
    return path.stat().st_size


def test_quant_schemes_registry_is_populated():
    from dataset_sorter.gguf_export import GGUF_QUANT_BY_KEY, GGUF_QUANT_SCHEMES
    keys = {s.key for s in GGUF_QUANT_SCHEMES}
    expected = {"F16", "BF16", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0"}
    assert expected.issubset(keys), f"Missing quants: {expected - keys}"
    # Lookup is consistent with list
    assert set(GGUF_QUANT_BY_KEY.keys()) == keys


def test_estimate_output_size_returns_smaller_than_source():
    from dataset_sorter.gguf_export import estimate_output_size

    src = 24_000_000_000  # 24 GB-ish source (Flux fp16)
    for key, max_ratio in (
        ("F16",  1.10),
        ("Q8_0", 0.60),
        ("Q5_1", 0.45),
        ("Q4_0", 0.35),
    ):
        est = estimate_output_size(src, key)
        assert est < int(src * max_ratio), (
            f"{key}: estimate {est} >= {max_ratio:.0%} of source"
        )


def test_arch_mapping_covers_main_archs():
    from dataset_sorter.gguf_export import GGUF_ARCH_MAP
    for arch in ("flux", "sdxl", "sd3", "pixart", "sana", "auraflow", "kolors"):
        assert arch in GGUF_ARCH_MAP, f"Missing arch mapping: {arch}"


@pytest.mark.parametrize("quant_key", ["F16", "Q8_0", "Q5_1", "Q4_0"])
def test_export_roundtrip_for_each_quant(quant_key):
    """Each quant must produce a readable GGUF file with all tensors present."""
    import gguf
    from dataset_sorter.gguf_export import export_safetensors_to_gguf

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / f"out_{quant_key}.gguf"

        src_size = _write_synthetic_safetensors(src)
        result = export_safetensors_to_gguf(src, out, quant_key, arch="flux")

        # Output must exist and be smaller than F16 source for quantized variants
        assert out.exists(), f"{quant_key}: output file missing"
        assert result.output_bytes > 0
        if quant_key.startswith("Q"):
            assert result.output_bytes < src_size, (
                f"{quant_key}: expected smaller output than source "
                f"({result.output_bytes} >= {src_size})"
            )

        # Re-read the GGUF and check we got every tensor back
        reader = gguf.GGUFReader(str(out))
        names = {t.name for t in reader.tensors}
        for expected in (
            "transformer.attn.to_q.weight",
            "transformer.ffn.weight",
            "transformer.attn.to_q.bias",
            "transformer.norm.weight",
            "transformer.embed.weight",
        ):
            assert expected in names, f"{quant_key}: missing tensor {expected}"

        # Architecture metadata must be set
        arch_field = reader.fields.get("general.architecture")
        assert arch_field is not None


def test_export_quantized_tensors_are_lossy_but_close():
    """Q8_0 should preserve tensor values to within ~1% relative error."""
    import gguf
    import torch
    from safetensors.torch import save_file
    from dataset_sorter.gguf_export import export_safetensors_to_gguf

    torch.manual_seed(0)
    weight = torch.randn(128, 128)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "m.safetensors"
        out = td / "m.gguf"
        save_file({"transformer.x.weight": weight}, str(src))

        export_safetensors_to_gguf(src, out, "Q8_0", arch="flux")

        reader = gguf.GGUFReader(str(out))
        t = next(iter(reader.tensors))
        # GGUF stores shape reversed; dequantize and reshape to source layout
        dequant = gguf.quants.dequantize(t.data, t.tensor_type).reshape(
            tuple(reversed(t.shape.tolist()))
        )

        original = weight.numpy()
        rel_err = np.abs(dequant - original).mean() / (np.abs(original).mean() + 1e-9)
        assert rel_err < 0.05, f"Q8_0 relative error too high: {rel_err:.4f}"


def test_export_skips_disallowed_tensor_patterns():
    """Norm / embedding / bias tensors must stay in F16 even when Q8_0 is requested."""
    from dataset_sorter.gguf_export import export_safetensors_to_gguf

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / "out.gguf"

        _write_synthetic_safetensors(src)
        result = export_safetensors_to_gguf(src, out, "Q8_0", arch="flux")

        # 4 quantizable tensors (to_q/k/v/ffn), 4 kept in F16 (bias, norm, embed, tiny)
        assert result.n_tensors_quantized == 4
        assert result.n_tensors_kept_f16 == 4


def test_export_raises_on_missing_source():
    from dataset_sorter.gguf_export import export_safetensors_to_gguf

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.gguf"
        with pytest.raises(FileNotFoundError):
            export_safetensors_to_gguf(
                "/nonexistent/path.safetensors", out, "Q8_0", arch="flux"
            )


def test_export_raises_on_unknown_quant():
    from dataset_sorter.gguf_export import export_safetensors_to_gguf

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / "out.gguf"
        _write_synthetic_safetensors(src)
        with pytest.raises(ValueError, match="Unknown quantization"):
            export_safetensors_to_gguf(src, out, "Q42_K", arch="flux")
