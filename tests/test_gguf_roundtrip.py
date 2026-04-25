"""Round-trip + quality tests for GGUF export.

Existing ``test_gguf_export.py`` covers single-pass export only. This file
adds the harder cases:

1. **Round-trip via dequantize**: export to Q8_0, load it back as
   safetensors-equivalent tensors, then re-export to Q4_0. The final
   tensors must be ~Q4_0 quality (not double-degraded Q8_0→Q4_0).
2. **Quality bounds per quant scheme**: each preset must keep the
   reconstructed tensors within the documented relative-error envelope.
   Numbers come from the ``QuantScheme.description`` claims in
   ``gguf_export.py`` — a regression that drifts above these bounds
   means the exporter pulled in a buggy quantization function.
3. **Tensors with unusual shapes**: 1D vectors, tiny matrices, large
   weights. The exporter must fall back to F16 for shapes that the
   block-quant size doesn't divide cleanly, not crash.
4. **Cancellation**: ``cancel_check`` callback must abort mid-export
   without leaving a partial output file behind.
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
    reason="gguf, safetensors, or torch missing",
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_synthetic_safetensors(path: Path, shapes: list[tuple[str, tuple[int, ...]]]):
    """Create a safetensors file with named tensors of given shapes."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    tensors = {name: torch.randn(*shape) for name, shape in shapes}
    save_file(tensors, str(path))
    return tensors


def _load_gguf_tensor_as_fp32(gguf_path: Path, tensor_name: str) -> np.ndarray:
    """Load a tensor from a GGUF file and dequantize back to fp32."""
    import gguf
    reader = gguf.GGUFReader(str(gguf_path))
    for t in reader.tensors:
        if t.name == tensor_name:
            return gguf.quants.dequantize(t.data, t.tensor_type).reshape(
                tuple(reversed(t.shape.tolist()))
            ).astype(np.float32)
    raise KeyError(f"Tensor {tensor_name!r} not found in {gguf_path}")


# ─────────────────────────────────────────────────────────────────────────
# Quality bounds per quant scheme
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("quant_key,max_rel_err", [
    ("F16",  0.001),   # near-lossless
    ("Q8_0", 0.05),    # ~50% size, virtually no quality loss
    ("Q5_1", 0.10),    # ~35% size, excellent quality
    ("Q5_0", 0.12),
    ("Q4_1", 0.18),
    ("Q4_0", 0.20),    # smallest, most lossy
])
def test_quant_quality_within_documented_bounds(quant_key, max_rel_err):
    """Each preset must keep reconstruction within its documented error
    envelope. A regression that lands a buggy quantizer would push some
    quant scheme out of these bounds — and we'd catch it here."""
    import torch
    from dataset_sorter.gguf_export import export_safetensors_to_gguf

    torch.manual_seed(42)
    weight = torch.randn(256, 256)
    expected = weight.numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / f"out_{quant_key}.gguf"

        from safetensors.torch import save_file
        save_file({"transformer.x.weight": weight}, str(src))

        export_safetensors_to_gguf(src, out, quant_key, arch="flux")

        actual = _load_gguf_tensor_as_fp32(out, "transformer.x.weight")
        rel_err = np.abs(actual - expected).mean() / (
            np.abs(expected).mean() + 1e-9
        )
        assert rel_err < max_rel_err, (
            f"{quant_key}: relative error {rel_err:.4f} exceeds documented "
            f"bound {max_rel_err}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Round-trip dequant → re-quant
# ─────────────────────────────────────────────────────────────────────────


def test_q8_to_q4_double_export_keeps_q4_quality():
    """Export a model to Q8_0, dequantize and re-export to Q4_0. The
    Q4_0 result must be roughly the same quality as a direct Q4_0
    export from the original — not "double quantization noise" stacking."""
    import torch
    from dataset_sorter.gguf_export import export_safetensors_to_gguf
    from safetensors.torch import save_file

    torch.manual_seed(7)
    weight = torch.randn(128, 128)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        save_file({"transformer.x.weight": weight}, str(src))

        # Direct Q4_0 export — baseline quality
        direct_q4 = td / "direct_q4.gguf"
        export_safetensors_to_gguf(src, direct_q4, "Q4_0", arch="flux")
        direct_arr = _load_gguf_tensor_as_fp32(direct_q4, "transformer.x.weight")

        # Round-trip: Q8_0 → dequantize → Q4_0
        q8_step = td / "step_q8.gguf"
        export_safetensors_to_gguf(src, q8_step, "Q8_0", arch="flux")
        q8_arr = _load_gguf_tensor_as_fp32(q8_step, "transformer.x.weight")

        # Resave the dequantized Q8 as safetensors, then export to Q4_0
        intermediate = td / "intermediate.safetensors"
        save_file(
            {"transformer.x.weight": torch.from_numpy(q8_arr)},
            str(intermediate),
        )
        rt_q4 = td / "roundtrip_q4.gguf"
        export_safetensors_to_gguf(intermediate, rt_q4, "Q4_0", arch="flux")
        rt_arr = _load_gguf_tensor_as_fp32(rt_q4, "transformer.x.weight")

        original = weight.numpy().astype(np.float32)
        direct_err = np.abs(direct_arr - original).mean() / (
            np.abs(original).mean() + 1e-9
        )
        rt_err = np.abs(rt_arr - original).mean() / (
            np.abs(original).mean() + 1e-9
        )

        # Round-trip should be at most 50% worse than direct (Q8 noise
        # adds a tiny extra error but Q4 dominates).
        assert rt_err < direct_err * 1.5, (
            f"Round-trip Q8→Q4 ({rt_err:.4f}) significantly worse than "
            f"direct Q4 ({direct_err:.4f}) — quantization stacking issue"
        )


# ─────────────────────────────────────────────────────────────────────────
# Unusual tensor shapes — fallback to F16
# ─────────────────────────────────────────────────────────────────────────


def test_1d_tensor_falls_back_to_f16():
    """Block-quantized formats need a 2D inner dim divisible by 32. 1D
    biases / norms should silently fall back to F16."""
    import torch
    from dataset_sorter.gguf_export import export_safetensors_to_gguf
    from safetensors.torch import save_file
    import gguf

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / "out.gguf"
        save_file({
            "transformer.bias": torch.randn(64),  # 1D
            "transformer.weight": torch.randn(64, 64),  # 2D, would quantize
        }, str(src))

        result = export_safetensors_to_gguf(src, out, "Q8_0", arch="flux")
        # Bias kept as F16, weight quantized as Q8_0
        assert result.n_tensors_kept_f16 >= 1
        assert result.n_tensors_quantized >= 1

        reader = gguf.GGUFReader(str(out))
        types = {t.name: t.tensor_type for t in reader.tensors}
        assert types["transformer.bias"] == gguf.GGMLQuantizationType.F16
        assert types["transformer.weight"] == gguf.GGMLQuantizationType.Q8_0


def test_inner_dim_smaller_than_block_size_falls_back():
    """A 2D tensor whose inner dim is < 32 cannot be Q8_0-quantized
    (block size is 32). Must fall back to F16, not crash."""
    import torch
    from dataset_sorter.gguf_export import export_safetensors_to_gguf
    from safetensors.torch import save_file
    import gguf

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / "out.gguf"
        # 32×16 — inner dim 16 too small for Q8_0 blocks of 32
        save_file({"transformer.tiny": torch.randn(32, 16)}, str(src))
        result = export_safetensors_to_gguf(src, out, "Q8_0", arch="flux")

        reader = gguf.GGUFReader(str(out))
        t = next(iter(reader.tensors))
        assert t.tensor_type == gguf.GGMLQuantizationType.F16, (
            f"Inner dim < block size should fall back to F16; got {t.tensor_type}"
        )
        assert result.n_tensors_kept_f16 == 1


def test_inner_dim_not_multiple_of_block_size_falls_back():
    """64×33 — 33 isn't a multiple of 32. Must use F16, not crash."""
    import torch
    from dataset_sorter.gguf_export import export_safetensors_to_gguf
    from safetensors.torch import save_file
    import gguf

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / "out.gguf"
        save_file({"transformer.odd": torch.randn(64, 33)}, str(src))
        result = export_safetensors_to_gguf(src, out, "Q4_0", arch="flux")

        reader = gguf.GGUFReader(str(out))
        t = next(iter(reader.tensors))
        assert t.tensor_type == gguf.GGMLQuantizationType.F16


# ─────────────────────────────────────────────────────────────────────────
# Cancellation
# ─────────────────────────────────────────────────────────────────────────


def test_cancel_check_aborts_export_and_removes_partial():
    """When ``cancel_check`` returns True between tensors, the export
    must raise and remove the partial output file (no silent garbage
    on disk)."""
    import torch
    from dataset_sorter.gguf_export import export_safetensors_to_gguf
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.safetensors"
        out = td / "out.gguf"
        # Many tensors so the loop has multiple cancel-check points
        save_file({
            f"transformer.layer_{i}.weight": torch.randn(64, 64)
            for i in range(20)
        }, str(src))

        # Cancel after the 3rd tensor
        call_count = [0]

        def _cancel():
            call_count[0] += 1
            return call_count[0] > 3

        with pytest.raises(RuntimeError, match="cancel"):
            export_safetensors_to_gguf(
                src, out, "Q8_0", arch="flux",
                cancel_check=_cancel,
            )

        # Partial output must NOT exist
        assert not out.exists(), (
            "Cancelled export left a partial GGUF file on disk"
        )
