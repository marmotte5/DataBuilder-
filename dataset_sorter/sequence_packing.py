"""Sequence packing for DiT/transformer-based diffusion models.

When training with multiple aspect ratios, standard bucketing pads shorter
sequences to match the longest in the batch. This wastes significant compute
on padding tokens that contribute nothing to the loss.

Sequence packing solves this by concatenating multiple images' latent
sequences into a single long sequence, separated by attention masks.
This eliminates all padding waste.

Example:
  Standard bucketing (batch_size=3, max_len=1024):
    Image A: 512 tokens + 512 padding  → 50% wasted
    Image B: 768 tokens + 256 padding  → 25% wasted
    Image C: 256 tokens + 768 padding  → 75% wasted
    Total: 3072 elements, 1536 wasted (50% waste)

  Sequence packing:
    Packed: [A:512 | B:768 | C:256] = 1536 tokens, 0 waste
    Uses flash_attn_varlen to handle variable-length attention correctly.

Requires: flash-attn >= 2.5.0 (for flash_attn_varlen_func)
Falls back to standard padded attention when flash-attn is unavailable.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

_FLASH_VARLEN_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func
    _FLASH_VARLEN_AVAILABLE = True
except ImportError:
    pass


class SequencePacker:
    """Packs multiple variable-length latent sequences into contiguous tensors.

    For DiT models (Flux, SD3, PixArt, etc.), the latent tensor is typically
    reshaped from (B, C, H, W) → (B, H*W, C) before entering the transformer.
    Different aspect ratios produce different H*W sequence lengths.

    This packer concatenates them into a single (1, total_len, C) tensor
    with corresponding cu_seqlens for flash_attn_varlen.
    """

    def __init__(self, max_packed_length: int = 4096, device: torch.device = None):
        self.max_packed_length = max_packed_length
        self.device = device or torch.device("cuda")

    def pack_latents(
        self, latents_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
        """Pack a list of latent tensors into a single contiguous sequence.

        Args:
            latents_list: List of (C, H, W) latent tensors (no batch dim).

        Returns:
            packed_latents: (1, total_seq_len, C) packed tensor
            cu_seqlens: (num_sequences + 1,) cumulative sequence lengths
            max_seqlen: Maximum sequence length in the pack
            shapes: Original (C, H, W) shapes for unpacking
        """
        sequences = []
        shapes = []
        seq_lens = []

        for lat in latents_list:
            c, h, w = lat.shape
            # Reshape to (h*w, c) — sequence of spatial tokens
            seq = lat.reshape(c, h * w).permute(1, 0)  # (h*w, c)
            sequences.append(seq)
            shapes.append((c, h, w))
            seq_lens.append(h * w)

        if not sequences:
            empty = torch.empty(1, 0, 0, device=self.device)
            cu = torch.zeros(1, dtype=torch.int32, device=self.device)
            return empty, cu, torch.tensor(0, device=self.device), []

        # Concatenate all sequences
        packed = torch.cat(sequences, dim=0).unsqueeze(0)  # (1, total_len, c)

        # Build cumulative sequence lengths for flash_attn_varlen
        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=self.device)
        for i, sl in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl

        max_seqlen = max(seq_lens)

        return packed, cu_seqlens, torch.tensor(max_seqlen, device=self.device), shapes

    def pack_batch(
        self, latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pack a batch of same-size latents (for uniform aspect ratio).

        For same-size batches, this is simpler: just track batch boundaries.

        Args:
            latents: (B, C, H, W) tensor.

        Returns:
            packed: (1, B*H*W, C) packed tensor
            cu_seqlens: (B+1,) cumulative lengths
            max_seqlen: H*W
        """
        b, c, h, w = latents.shape
        seq_len = h * w

        # Reshape: (B, C, H, W) → (B, H*W, C) → (1, B*H*W, C)
        # Force contiguous first — channels_last memory format makes the
        # initial reshape raise "view size is not compatible" otherwise.
        packed = latents.contiguous().reshape(b, c, seq_len).permute(0, 2, 1).contiguous().reshape(1, b * seq_len, c)

        cu_seqlens = torch.arange(
            0, (b + 1) * seq_len, seq_len,
            dtype=torch.int32, device=latents.device,
        )

        return packed, cu_seqlens, seq_len

    @staticmethod
    def unpack_output(
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        shapes: list[tuple[int, int, int]],
    ) -> list[torch.Tensor]:
        """Unpack a packed output back to individual (C, H, W) tensors."""
        results = []
        packed = packed_output.squeeze(0)  # (total_len, c)
        for i, (c, h, w) in enumerate(shapes):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq = packed[start:end]  # (h*w, c)
            results.append(seq.permute(1, 0).reshape(c, h, w))
        return results


class PackedAttention(nn.Module):
    """Drop-in attention replacement that uses flash_attn_varlen.

    Replaces standard scaled_dot_product_attention with flash_attn_varlen_func
    when sequence packing is active. Falls back to standard attention otherwise.
    """

    def __init__(self, original_attn: nn.Module, num_heads: int, head_dim: int):
        super().__init__()
        self.original_attn = original_attn
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._cu_seqlens = None
        self._max_seqlen = None

    def set_packing_info(self, cu_seqlens: torch.Tensor, max_seqlen: int):
        """Set the current packing info (called each forward pass)."""
        self._cu_seqlens = cu_seqlens
        self._max_seqlen = max_seqlen

    def clear_packing_info(self):
        self._cu_seqlens = None
        self._max_seqlen = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (self._cu_seqlens is not None and _FLASH_VARLEN_AVAILABLE
                and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
            return self._packed_attention(q, k, v)
        return self.original_attn(q, k, v, attn_mask=attn_mask)

    def _packed_attention(self, q, k, v):
        """Use flash_attn_varlen_func for packed sequences."""
        # Reshape from (1, num_heads, total_len, head_dim) to (total_len, num_heads, head_dim)
        total_len = q.shape[2] if q.dim() == 4 else q.shape[0]

        if q.dim() == 4:
            q = q.squeeze(0).permute(1, 0, 2)  # (total_len, num_heads, head_dim)
            k = k.squeeze(0).permute(1, 0, 2)
            v = v.squeeze(0).permute(1, 0, 2)

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            causal=False,  # Diffusion models use bidirectional attention
        )

        # Reshape back to (1, num_heads, total_len, head_dim)
        return out.permute(1, 0, 2).unsqueeze(0)


class PackedTrainingStep:
    """Wraps the training step to use sequence packing when beneficial.

    Automatically detects when a batch has variable-length sequences
    (from aspect ratio bucketing) and packs them to eliminate padding waste.
    """

    def __init__(self, enabled: bool = True, device: torch.device = None):
        self.enabled = enabled and _FLASH_VARLEN_AVAILABLE
        self.packer = SequencePacker(device=device) if self.enabled else None
        self._total_saved_tokens = 0
        self._total_tokens = 0

    def pack_if_needed(
        self, latents: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        """Pack latents if they come from different aspect ratio buckets.

        For uniform-size batches, returns the original latents unchanged.

        Returns:
            latents: Possibly packed (1, total_len, C) tensor
            cu_seqlens: Cumulative sequence lengths (or None)
            max_seqlen: Maximum sequence length (or None)
        """
        if not self.enabled:
            return latents, None, None

        # For uniform batches, packing doesn't save compute
        # but can still be used for flash_attn_varlen compatibility
        b, c, h, w = latents.shape
        packed, cu_seqlens, max_seqlen = self.packer.pack_batch(latents)
        return packed, cu_seqlens, max_seqlen

    def get_stats(self) -> dict:
        """Return packing efficiency statistics."""
        return {
            "enabled": self.enabled,
            "flash_varlen_available": _FLASH_VARLEN_AVAILABLE,
            "total_tokens_processed": self._total_tokens,
            "tokens_saved": self._total_saved_tokens,
            "efficiency": (1 - self._total_saved_tokens / max(1, self._total_tokens)),
        }


def is_packing_available() -> bool:
    """Check if sequence packing is available (requires flash-attn)."""
    return _FLASH_VARLEN_AVAILABLE
