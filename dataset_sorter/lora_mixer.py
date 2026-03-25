"""
Module: lora_mixer.py
========================
Fusion de plusieurs adaptateurs LoRA en un seul fichier merged.

Role in DataBuilder:
    - Combines LoRAs of different styles/concepts with per-LoRA weighting
      (e.g. style.safetensors:0.7 + concept.safetensors:0.3)
    - Supporte trois modes de fusion : weighted_average, add, svd_merge
    - Automatically detects and normalizes divergent key formats
      (PEFT, diffusers, kohya) to a canonical internal format before merging
    - Expose une API Python (LoRAMixer) et une interface CLI

Classes/Fonctions principales:
    - LoRAMixer            : API principale — add_lora(), validate(), mix()
    - _detect_format()     : Detects the key format (kohya/peft/diffusers)
    - _normalize_key()     : Converts keys to a canonical internal format
    - _kohya_path_to_dotted(): Reconvertit les chemins kohya underscore en points

Modes de fusion:
    - weighted_average : tensor = sum(t_i * w_i) / sum(w_i)  — recommended
    - add              : tensor = sum(t_i * w_i)  — pour combinaison additive
    - svd_merge        : reconstructs W_delta = up @ down, then re-decomposes
                         via truncated SVD — better quality, slower

Dependencies: torch, safetensors

Usage CLI:
    python -m dataset_sorter.lora_mixer lora1.safetensors:0.7 lora2.safetensors:0.3 -o mixed.safetensors
    python -m dataset_sorter.lora_mixer lora1.safetensors lora2.safetensors --mode add -o out.safetensors
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Literal

import torch

log = logging.getLogger(__name__)

MergeMode = Literal["weighted_average", "add", "svd_merge"]

# ============================================================
# SECTION: Regex patterns for LoRA key formats
# ============================================================

# The three major tools (kohya-ss, PEFT, diffusers) use incompatible
# naming conventions. We normalize to a
# canonical <component>.<path>.<suffix> format before any operation.

# kohya: lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
_KOHYA_RE = re.compile(r"^lora_(unet|te|te1|te2)_(.+)\.(lora_down|lora_up|alpha)(?:\.weight)?$")

# PEFT: base_model.model.down_blocks.0.attentions.0.proj_in.lora_A.weight
_PEFT_RE = re.compile(r"^base_model\.model\.(.+)\.(lora_A|lora_B)\.weight$")

# diffusers: unet.down_blocks.0.attentions.0.proj_in.lora_down.weight
_DIFFUSERS_RE = re.compile(r"^(unet|text_encoder|text_encoder_2|transformer)\.(.+)\.(lora_down|lora_up|alpha)(?:\.weight)?$")


def _detect_format(keys: list[str]) -> str:
    """Detect the key format of a LoRA file."""
    sample = keys[:20]
    kohya_count = sum(1 for k in sample if _KOHYA_RE.match(k))
    peft_count = sum(1 for k in sample if _PEFT_RE.match(k))
    diffusers_count = sum(1 for k in sample if _DIFFUSERS_RE.match(k))

    counts = {"kohya": kohya_count, "peft": peft_count, "diffusers": diffusers_count}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "unknown"
    return best


def _normalize_key(key: str, fmt: str) -> str:
    """Normalize a key to a canonical internal format.

    Canonical form: ``<component>.<path>.<suffix>``
    where suffix ∈ {lora_down, lora_up, alpha}.
    """
    if fmt == "kohya":
        m = _KOHYA_RE.match(key)
        if m:
            component, path, suffix = m.group(1), m.group(2), m.group(3)
            path = _kohya_path_to_dotted(path)
            return f"{component}.{path}.{suffix}"
    elif fmt == "peft":
        m = _PEFT_RE.match(key)
        if m:
            path, ab = m.group(1), m.group(2)
            suffix = "lora_down" if ab == "lora_A" else "lora_up"
            return f"unet.{path}.{suffix}"
    elif fmt == "diffusers":
        m = _DIFFUSERS_RE.match(key)
        if m:
            component, path, suffix = m.group(1), m.group(2), m.group(3)
            return f"{component}.{path}.{suffix}"
    # Fallback: return as-is
    return key


def _load_lora(path: str) -> dict[str, torch.Tensor]:
    """Load a LoRA file into a dict of tensors (CPU, float32)."""
    from safetensors.torch import load_file

    tensors = load_file(path, device="cpu")
    return {k: v.float() for k, v in tensors.items()}


def _get_rank(tensors: dict[str, torch.Tensor], norm_key: str) -> int | None:
    """Return the rank for a given normalized key by looking at lora_down shape."""
    down_key = norm_key if norm_key.endswith("lora_down") else None
    if down_key and down_key in tensors:
        t = tensors[down_key]
        return t.shape[0] if t.ndim >= 1 else None
    return None


# ============================================================
# SECTION: Classe principale LoRAMixer
# ============================================================


class LoRAMixer:
    """Combine multiple LoRA adapters into a single merged LoRA.

    Example::

        mixer = LoRAMixer()
        mixer.add_lora("style.safetensors", weight=0.7)
        mixer.add_lora("concept.safetensors", weight=0.3)
        mixer.mix("merged.safetensors", mode="weighted_average")
    """

    def __init__(self) -> None:
        # Each entry: (path, weight, name, raw_tensors, fmt, normalized_tensors)
        self._loras: list[tuple[str, float, str, dict, str, dict]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_lora(self, path: str, weight: float = 1.0, name: str | None = None) -> None:
        """Add a LoRA with an optional blending weight."""
        if not Path(path).exists():
            raise FileNotFoundError(f"LoRA not found: {path}")
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")

        raw = _load_lora(path)
        fmt = _detect_format(list(raw.keys()))
        normalized = {_normalize_key(k, fmt): v for k, v in raw.items()}

        display_name = name or Path(path).stem
        self._loras.append((path, weight, display_name, raw, fmt, normalized))
        log.info("Added LoRA '%s' (format=%s, keys=%d, weight=%.3f)", display_name, fmt, len(raw), weight)

    def remove_lora(self, index: int) -> None:
        """Remove the LoRA at *index*."""
        if index < 0 or index >= len(self._loras):
            raise IndexError(f"Index {index} out of range (have {len(self._loras)} LoRAs)")
        name = self._loras[index][2]
        self._loras.pop(index)
        log.info("Removed LoRA '%s' at index %d", name, index)

    def set_weight(self, index: int, weight: float) -> None:
        """Update the blending weight for the LoRA at *index*."""
        if index < 0 or index >= len(self._loras):
            raise IndexError(f"Index {index} out of range")
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        path, _, name, raw, fmt, norm = self._loras[index]
        self._loras[index] = (path, weight, name, raw, fmt, norm)

    def get_common_keys(self) -> list[str]:
        """Return normalized keys present in **all** loaded LoRAs."""
        if not self._loras:
            return []
        sets = [set(entry[5].keys()) for entry in self._loras]
        common = sets[0].intersection(*sets[1:])
        return sorted(common)

    def get_info(self) -> dict:
        """Return metadata for each loaded LoRA."""
        result = []
        for path, weight, name, raw, fmt, normalized in self._loras:
            down_keys = [k for k in normalized if k.endswith("lora_down")]
            ranks = {}
            for dk in down_keys:
                t = normalized[dk]
                r = t.shape[0] if t.ndim >= 1 else None
                if r is not None:
                    ranks[r] = ranks.get(r, 0) + 1
            result.append({
                "name": name,
                "path": path,
                "weight": weight,
                "format": fmt,
                "num_keys": len(raw),
                "num_lora_layers": len(down_keys),
                "ranks": ranks,
            })
        return {"loras": result, "common_keys": len(self.get_common_keys())}

    def validate(self) -> list[str]:
        """Check compatibility across loaded LoRAs. Returns list of warning strings."""
        warnings: list[str] = []
        if len(self._loras) < 2:
            return warnings

        common = self.get_common_keys()
        if not common:
            warnings.append("No common keys found — LoRAs may be incompatible (different architectures).")
            return warnings

        # Warn about rank mismatches
        all_ranks: dict[str, set[int]] = {}
        for _, _, name, _, _, norm in self._loras:
            for k in norm:
                if k.endswith("lora_down") or k.endswith("lora_down.weight"):
                    r = norm[k].shape[0]
                    all_ranks.setdefault(k, set()).add(r)

        mismatched = {k: ranks for k, ranks in all_ranks.items() if len(ranks) > 1}
        if mismatched:
            warnings.append(
                f"{len(mismatched)} key(s) have mismatched ranks across LoRAs. "
                "Mix will use the largest rank (zero-padded). "
                f"Example: {next(iter(mismatched))!r} has ranks {next(iter(mismatched.values()))}"
            )

        # Check coverage: keys only in some LoRAs
        all_keys_union: set[str] = set()
        for entry in self._loras:
            all_keys_union |= entry[5].keys()
        partial = all_keys_union - set(common)
        if partial:
            warnings.append(
                f"{len(partial)} key(s) are not present in all LoRAs and will be excluded from the mix."
            )

        return warnings

    def mix(self, output_path: str, mode: MergeMode = "weighted_average") -> None:
        """Merge all loaded LoRAs and save to *output_path* (safetensors).

        Args:
            output_path: Destination ``.safetensors`` file.
            mode: One of ``weighted_average``, ``add``, or ``svd_merge``.

        Raises:
            ValueError: If fewer than 2 LoRAs are loaded or mode is unknown.
            RuntimeError: If no common keys are found.
        """
        if len(self._loras) < 2:
            raise ValueError("Need at least 2 LoRAs to mix.")
        if mode not in ("weighted_average", "add", "svd_merge"):
            raise ValueError(f"Unknown mode '{mode}'. Use: weighted_average, add, svd_merge")

        warnings = self.validate()
        for w in warnings:
            log.warning(w)

        common = self.get_common_keys()
        if not common:
            raise RuntimeError("No common keys found across LoRAs — cannot merge.")

        log.info("Merging %d LoRAs, mode=%s, %d common keys", len(self._loras), mode, len(common))

        if mode == "weighted_average":
            merged = self._merge_weighted_average(common)
        elif mode == "add":
            merged = self._merge_add(common)
        else:  # svd_merge
            merged = self._merge_svd(common)

        self._save(merged, output_path)
        log.info("Saved merged LoRA to %s", output_path)

    # ── Merge implementations ─────────────────────────────────────────────────

    def _merge_weighted_average(self, common: list[str]) -> dict[str, torch.Tensor]:
        """Weighted average: tensor = sum(t_i * w_i) / sum(w_i)."""
        total_weight = sum(e[1] for e in self._loras)
        merged: dict[str, torch.Tensor] = {}

        for key in common:
            tensors_and_weights = [(e[5][key], e[1]) for e in self._loras]
            target_shape = _broadcast_shape([t.shape for t, _ in tensors_and_weights])
            acc = torch.zeros(target_shape, dtype=torch.float32)
            for t, w in tensors_and_weights:
                log.debug("weighted_avg key=%s shape=%s weight=%.4f", key, tuple(t.shape), w)
                acc += _pad_to_shape(t, target_shape) * w
            merged[key] = acc / total_weight

        return merged

    def _merge_add(self, common: list[str]) -> dict[str, torch.Tensor]:
        """Additive merge: tensor = sum(t_i * w_i)."""
        merged: dict[str, torch.Tensor] = {}

        for key in common:
            tensors_and_weights = [(e[5][key], e[1]) for e in self._loras]
            target_shape = _broadcast_shape([t.shape for t, _ in tensors_and_weights])
            acc = torch.zeros(target_shape, dtype=torch.float32)
            for t, w in tensors_and_weights:
                log.debug("add key=%s shape=%s weight=%.4f", key, tuple(t.shape), w)
                acc += _pad_to_shape(t, target_shape) * w
            merged[key] = acc

        return merged

    def _merge_svd(self, common: list[str]) -> dict[str, torch.Tensor]:
        # SVD merge algorithm:
        # 1. For each (lora_down, lora_up) pair, reconstruct the full delta
        #    matrix: W_delta = up @ down * (alpha / rank)
        # 2. Accumulate the weighted sum of W_delta across all LoRAs
        # 3. Re-decompose the result with truncated SVD at the max rank found:
        #    U, S, Vh = svd(W_acc) ; new_up = U[:, :r] * sqrt(S[:r])
        #    new_down = Vh[:r, :] * sqrt(S[:r])  (equal split)
        # This approach produces a least-squares "optimal" LoRA
        # but is more memory-intensive (temporary full matrix).
        """SVD-based merge: reconstruct full delta weights then re-decompose.

        For each lora_down/lora_up pair, reconstructs W_delta = up @ down,
        accumulates weighted sum, then re-decomposes with truncated SVD to
        the maximum rank found across inputs. Alpha keys are averaged.
        """
        total_weight = sum(e[1] for e in self._loras)
        merged: dict[str, torch.Tensor] = {}

        # Group keys by their layer path (strip lora_down/lora_up/alpha suffix)
        layers: dict[str, list[str]] = {}
        for key in common:
            base = _layer_base(key)
            layers.setdefault(base, []).append(key)

        for base, suffixes in layers.items():
            has_down = any(s.endswith("lora_down") for s in suffixes)
            has_up = any(s.endswith("lora_up") for s in suffixes)

            if not (has_down and has_up):
                # Not a complete pair — fall back to weighted average
                for key in suffixes:
                    tensors_and_weights = [(e[5][key], e[1]) for e in self._loras]
                    target_shape = _broadcast_shape([t.shape for t, _ in tensors_and_weights])
                    acc = torch.zeros(target_shape, dtype=torch.float32)
                    for t, w in tensors_and_weights:
                        acc += _pad_to_shape(t, target_shape) * w
                    merged[key] = acc / total_weight
                continue

            down_key = f"{base}.lora_down"
            up_key = f"{base}.lora_up"
            alpha_key = f"{base}.alpha"

            # Determine max rank
            ranks = [e[5][down_key].shape[0] for e in self._loras if down_key in e[5]]
            max_rank = max(ranks)

            # Accumulate full delta W = up @ down (weighted)
            delta_acc: torch.Tensor | None = None
            alpha_acc = 0.0

            for entry in self._loras:
                _, w, _, _, _, norm = entry
                down = norm[down_key]  # (rank, in)
                up = norm[up_key]      # (out, rank)

                # Handle conv weights: reshape to 2D
                down_2d = down.flatten(1) if down.ndim > 2 else down
                up_2d = up.flatten(1) if up.ndim > 2 else up

                alpha_val = norm.get(alpha_key, torch.tensor(float(down.shape[0]))).item()
                scale = alpha_val / down.shape[0]

                delta = (up_2d @ down_2d) * scale * w  # (out, in)
                delta_acc = delta if delta_acc is None else delta_acc + delta

                alpha_acc += alpha_val * w

            delta_acc = delta_acc / total_weight  # normalize
            alpha_acc /= total_weight

            # Re-decompose with truncated SVD
            try:
                U, S, Vh = torch.linalg.svd(delta_acc, full_matrices=False)
                rank = min(max_rank, S.shape[0])
                U_r = U[:, :rank]          # (out, rank)
                S_r = S[:rank]             # (rank,)
                Vh_r = Vh[:rank, :]        # (rank, in)

                new_up = U_r * S_r.sqrt().unsqueeze(0)   # (out, rank)
                new_down = Vh_r * S_r.sqrt().unsqueeze(1)  # (rank, in)

                # Restore conv shape if original was conv
                orig_down = self._loras[0][5][down_key]
                orig_up = self._loras[0][5][up_key]
                if orig_down.ndim == 4:
                    new_down = new_down.view(rank, *orig_down.shape[1:])
                if orig_up.ndim == 4:
                    new_up = new_up.view(orig_up.shape[0], rank, *orig_up.shape[2:])

                merged[down_key] = new_down
                merged[up_key] = new_up
                if alpha_key in common:
                    merged[alpha_key] = torch.tensor(alpha_acc)

            except Exception as exc:
                log.warning("SVD decomposition failed for %s (%s), falling back to weighted avg", base, exc)
                for key in suffixes:
                    tensors_and_weights = [(e[5][key], e[1]) for e in self._loras if key in e[5]]
                    target_shape = _broadcast_shape([t.shape for t, _ in tensors_and_weights])
                    acc = torch.zeros(target_shape, dtype=torch.float32)
                    for t, w in tensors_and_weights:
                        acc += _pad_to_shape(t, target_shape) * w
                    merged[key] = acc / total_weight

        return merged

    # ── I/O ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _save(tensors: dict[str, torch.Tensor], path: str) -> None:
        from safetensors.torch import save_file

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert back to float16 to match typical LoRA storage; ensure contiguous
        tensors_f16 = {k: v.half().contiguous() for k, v in tensors.items()}
        save_file(tensors_f16, path)


# ============================================================
# SECTION: Fonctions utilitaires internes
# ============================================================


def _kohya_path_to_dotted(path: str) -> str:
    """Convert kohya underscore path back to dot-separated, preserving compound words.

    Kohya converts ``down_blocks.0.attentions.0.proj_in`` →
    ``down_blocks_0_attentions_0_proj_in``.  We reverse this by treating the
    underscores adjacent to numeric tokens as ``.`` separators and keeping
    intra-word underscores as-is.

    Examples::

        down_blocks_0_attentions_0_proj_in → down_blocks.0.attentions.0.proj_in
        to_q → to_q
        encoder_layers_3_self_attn → encoder_layers.3.self_attn
    """
    tokens = path.split("_")
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        prev_digit = i > 0 and tokens[i - 1].isdigit()
        curr_digit = tok.isdigit()
        if i == 0:
            parts.append(tok)
        elif curr_digit or prev_digit:
            parts.append(".")
            parts.append(tok)
        else:
            parts.append("_")
            parts.append(tok)
    return "".join(parts)


def _layer_base(key: str) -> str:
    """Strip the trailing .lora_down / .lora_up / .alpha suffix."""
    for suffix in (".lora_down", ".lora_up", ".alpha"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _broadcast_shape(shapes: list[torch.Size]) -> torch.Size:
    """Return the element-wise max shape (for zero-padding mismatched ranks)."""
    if not shapes:
        raise ValueError("Empty shape list")
    ndim = shapes[0].__len__()
    if any(len(s) != ndim for s in shapes):
        # Different ndims — cannot broadcast, return first
        return shapes[0]
    return torch.Size([max(s[i] for s in shapes) for i in range(ndim)])


def _pad_to_shape(tensor: torch.Tensor, target: torch.Size) -> torch.Tensor:
    """Zero-pad *tensor* to *target* shape along all dimensions."""
    if tensor.shape == target:
        return tensor
    if tensor.ndim != len(target):
        return tensor  # incompatible ndims — return as-is
    result = torch.zeros(target, dtype=tensor.dtype)
    idx = tuple(slice(0, s) for s in tensor.shape)
    result[idx] = tensor
    return result


# ============================================================
# SECTION: Interface ligne de commande (CLI)
# ============================================================


def _parse_lora_arg(arg: str) -> tuple[str, float]:
    """Parse ``path:weight`` or just ``path`` (weight defaults to 1.0)."""
    if ":" in arg:
        path, _, weight_str = arg.rpartition(":")
        return path, float(weight_str)
    return arg, 1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix multiple LoRA adapters into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m dataset_sorter.lora_mixer a.safetensors:0.7 b.safetensors:0.3 -o mixed.safetensors\n"
            "  python -m dataset_sorter.lora_mixer a.safetensors b.safetensors --mode add -o out.safetensors\n"
        ),
    )
    parser.add_argument("loras", nargs="+", metavar="PATH[:WEIGHT]", help="LoRA files with optional weight")
    parser.add_argument("-o", "--output", required=True, help="Output safetensors file")
    parser.add_argument(
        "--mode",
        choices=["weighted_average", "add", "svd_merge"],
        default="weighted_average",
        help="Merge mode (default: weighted_average)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    mixer = LoRAMixer()
    for arg in args.loras:
        path, weight = _parse_lora_arg(arg)
        mixer.add_lora(path, weight=weight)

    warnings = mixer.validate()
    for w in warnings:
        log.warning("VALIDATION: %s", w)

    info = mixer.get_info()
    log.info("Common keys: %d", info["common_keys"])
    for lora_info in info["loras"]:
        log.info("  %s (weight=%.3f, keys=%d, layers=%d, ranks=%s)",
                 lora_info["name"], lora_info["weight"],
                 lora_info["num_keys"], lora_info["num_lora_layers"],
                 lora_info["ranks"])

    mixer.mix(args.output, mode=args.mode)
    print(f"Merged LoRA saved to: {args.output}")


if __name__ == "__main__":
    main()
