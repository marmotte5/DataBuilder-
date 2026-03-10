"""Attention Map Debugger — Visualize cross-attention heatmaps on training images.

Hooks into UNet cross-attention layers to capture attention maps,
then generates heatmap overlays on the original training images to
debug why certain concepts aren't being learned.

Usage:
    debugger = AttentionMapDebugger(tokenizer)
    debugger.attach(unet)
    # ... run forward pass ...
    maps = debugger.get_attention_maps()
    overlay = debugger.create_heatmap_overlay(image, maps, token_index=3)
    debugger.detach()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class AttentionHook:
    """Hook that captures cross-attention maps from a single attention layer."""

    def __init__(self, name: str):
        self.name = name
        self.attention_map: Optional[torch.Tensor] = None
        self._handle = None

    def hook_fn(self, module, input_args, output):
        """Capture attention weights from the cross-attention layer.

        For diffusers UNet, cross-attention modules compute:
            attn_weights = softmax(Q @ K^T / sqrt(d))
        We intercept the output and reconstruct the attention map
        from Q and K stored in the module during forward.
        """
        # The hook captures the module's output after the attention computation.
        # We extract Q, K from the input to reconstruct attention probabilities.
        pass

    def clear(self):
        self.attention_map = None


def _compute_cross_attention(query, key):
    """Compute cross-attention probabilities from Q and K tensors.

    Args:
        query: (batch, heads, seq_q, dim)
        key: (batch, heads, seq_k, dim)

    Returns:
        attention_probs: (batch, heads, seq_q, seq_k)
    """
    scale = query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_probs = torch.softmax(attn_weights, dim=-1)
    return attn_probs


class CrossAttentionCapture(nn.Module):
    """Wrapper that captures cross-attention maps from an attention processor."""

    def __init__(self):
        super().__init__()
        self.captured_maps: list[torch.Tensor] = []
        self._enabled = False

    def clear(self):
        self.captured_maps.clear()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if not value:
            self.clear()


class AttentionMapDebugger:
    """Extract and visualize cross-attention maps from UNet/transformer models.

    Attaches lightweight hooks to cross-attention layers to capture
    where the model "looks" for each text token. Generates heatmap
    overlays on training images for debugging concept learning.
    """

    def __init__(self, tokenizer=None):
        """Initialize the debugger.

        Args:
            tokenizer: HuggingFace tokenizer for decoding token IDs to text.
        """
        self.tokenizer = tokenizer
        self._hooks: list[tuple[str, torch.utils.hooks.RemovableHook]] = []
        self._attention_maps: dict[str, torch.Tensor] = {}
        self._attached = False

    def attach(self, unet: nn.Module):
        """Attach attention capture hooks to all cross-attention layers.

        Args:
            unet: The UNet or transformer model to hook into.
        """
        if self._attached:
            self.detach()

        count = 0
        for name, module in unet.named_modules():
            # Match diffusers cross-attention processor modules
            if _is_cross_attention(name, module):
                handle = module.register_forward_hook(
                    self._make_hook(name),
                )
                self._hooks.append((name, handle))
                count += 1

        self._attached = True
        log.info(f"AttentionMapDebugger: attached to {count} cross-attention layers")

    def detach(self):
        """Remove all hooks from the model."""
        for name, handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._attention_maps.clear()
        self._attached = False

    def _make_hook(self, layer_name: str):
        """Create a forward hook that captures cross-attention maps."""
        debugger = self

        def hook_fn(module, args, output):
            # For Attention modules in diffusers, we capture the attention
            # output and compute an approximate attention map from the
            # hidden states shapes. The actual attention computation
            # happens inside the processor.
            if hasattr(module, '_attn_map'):
                # Some custom processors store attention maps
                debugger._attention_maps[layer_name] = module._attn_map.detach().cpu()
            elif isinstance(output, torch.Tensor) and len(args) >= 2:
                # Standard case: reconstruct from query/key if available
                hidden = args[0] if isinstance(args[0], torch.Tensor) else None
                encoder_hidden = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else None

                if hidden is not None and encoder_hidden is not None:
                    # This is a cross-attention layer
                    try:
                        attn_map = _approximate_attention_map(
                            module, hidden, encoder_hidden,
                        )
                        if attn_map is not None:
                            debugger._attention_maps[layer_name] = attn_map.detach().cpu()
                    except Exception:
                        pass  # Skip layers we can't capture

        return hook_fn

    def get_attention_maps(self) -> dict[str, torch.Tensor]:
        """Return captured attention maps from the last forward pass.

        Returns:
            Dict mapping layer names to attention tensors of shape
            (batch, heads, spatial_tokens, text_tokens).
        """
        return dict(self._attention_maps)

    def get_aggregated_map(
        self,
        token_index: int,
        resolution: Optional[tuple[int, int]] = None,
        layer_filter: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Aggregate attention maps across layers for a specific token.

        Args:
            token_index: Index of the text token to visualize.
            resolution: Target (H, W) for the output map. If None, uses
                the largest spatial resolution found.
            layer_filter: Optional substring to filter layer names (e.g. "mid" or "up").

        Returns:
            Normalized heatmap as numpy array of shape (H, W) with values in [0, 1],
            or None if no maps are available.
        """
        maps = self._attention_maps
        if not maps:
            return None

        aggregated = []
        for name, attn in maps.items():
            if layer_filter and layer_filter not in name:
                continue
            if attn.dim() < 3:
                continue
            # attn shape: (batch, heads, spatial, text) or (batch, spatial, text)
            if attn.dim() == 4:
                # Average over batch and heads
                m = attn.float().mean(dim=(0, 1))  # (spatial, text)
            else:
                m = attn.float().mean(dim=0)  # (spatial, text)

            if token_index >= m.shape[-1]:
                continue

            token_attn = m[:, token_index]  # (spatial,)
            # Try to reshape to 2D
            spatial_size = token_attn.shape[0]
            h = w = int(spatial_size ** 0.5)
            if h * w == spatial_size:
                token_attn = token_attn.view(h, w)
                aggregated.append(token_attn.numpy())

        if not aggregated:
            return None

        # Resize all maps to the target resolution and average
        if resolution is None:
            max_h = max(m.shape[0] for m in aggregated)
            max_w = max(m.shape[1] for m in aggregated)
            resolution = (max_h, max_w)

        from PIL import Image as PILImage

        resized = []
        for m in aggregated:
            if m.shape[0] != resolution[0] or m.shape[1] != resolution[1]:
                img = PILImage.fromarray(m)
                img = img.resize((resolution[1], resolution[0]), PILImage.Resampling.BILINEAR)
                m = np.array(img)
            resized.append(m)

        result = np.mean(resized, axis=0)
        # Normalize to [0, 1]
        rmin, rmax = result.min(), result.max()
        if rmax > rmin:
            result = (result - rmin) / (rmax - rmin)
        elif rmax > 0:
            # Uniform non-zero attention — treat as full attention everywhere
            result = np.ones_like(result)
        else:
            result = np.zeros_like(result)

        return result

    def create_heatmap_overlay(
        self,
        image: "PILImage",
        token_index: int,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Optional["PILImage"]:
        """Create a heatmap overlay on an image for a specific token.

        Args:
            image: PIL Image to overlay the heatmap on.
            token_index: Text token index to visualize.
            alpha: Overlay transparency (0 = image only, 1 = heatmap only).
            colormap: Colormap name ("jet", "hot", "viridis").

        Returns:
            PIL Image with heatmap overlay, or None if no maps available.
        """
        from PIL import Image as PILImage

        heatmap = self.get_aggregated_map(
            token_index, resolution=(image.height, image.width),
        )
        if heatmap is None:
            return None

        # Apply colormap
        colored_heatmap = _apply_colormap(heatmap, colormap)

        # Convert to PIL
        heatmap_img = PILImage.fromarray(colored_heatmap)
        heatmap_img = heatmap_img.resize(
            (image.width, image.height), PILImage.Resampling.BILINEAR,
        )

        # Blend with original image
        image_rgb = image.convert("RGB")
        overlay = PILImage.blend(image_rgb, heatmap_img, alpha)
        return overlay

    def tokenize_caption(self, caption: str) -> list[str]:
        """Tokenize a caption and return human-readable token strings.

        Args:
            caption: The text caption to tokenize.

        Returns:
            List of token strings (e.g. ["<|startoftext|>", "a", "photo", "of", ...]).
        """
        if self.tokenizer is None:
            return []
        tokens = self.tokenizer(
            caption, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        return [
            self.tokenizer.decode([tid]).strip()
            for tid in tokens.input_ids
            if tid != self.tokenizer.pad_token_id
        ]

    def generate_debug_report(
        self,
        image: "PILImage",
        caption: str,
        output_dir: Path,
        step: int = 0,
        top_k_tokens: int = 5,
    ) -> list[Path]:
        """Generate a full debug report with attention overlays for key tokens.

        Saves heatmap overlays for the top-K most attended tokens to disk.

        Args:
            image: Training image to overlay.
            caption: Caption text for this image.
            output_dir: Directory to save debug images.
            step: Current training step (for filenames).
            top_k_tokens: Number of tokens to visualize.

        Returns:
            List of saved file paths.
        """
        from PIL import Image as PILImage

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        tokens = self.tokenize_caption(caption) if self.tokenizer else []
        maps = self.get_attention_maps()
        if not maps:
            log.warning("No attention maps captured — cannot generate debug report")
            return saved

        # Compute total attention per token across all layers
        token_attention_scores = _compute_token_importance(maps)
        if token_attention_scores is None:
            return saved

        # Select top-K tokens by total attention
        num_tokens = min(len(tokens), len(token_attention_scores)) if tokens else len(token_attention_scores)
        if num_tokens == 0:
            return saved

        scores = token_attention_scores[:num_tokens]
        top_indices = np.argsort(scores)[::-1][:top_k_tokens]

        for rank, tidx in enumerate(top_indices):
            overlay = self.create_heatmap_overlay(image, int(tidx))
            if overlay is None:
                continue

            token_str = tokens[tidx] if tidx < len(tokens) else f"token_{tidx}"
            safe_token = "".join(c if c.isalnum() else "_" for c in token_str)
            filename = f"attn_step{step:06d}_rank{rank}_{safe_token}.png"
            path = output_dir / filename
            overlay.save(str(path))
            saved.append(path)

        log.info(f"Attention debug report: {len(saved)} overlays saved to {output_dir}")
        return saved

    def clear(self):
        """Clear all captured attention maps."""
        self._attention_maps.clear()


def _is_cross_attention(name: str, module: nn.Module) -> bool:
    """Check if a module is a cross-attention layer."""
    cls_name = type(module).__name__.lower()
    # Match diffusers Attention modules that do cross-attention
    if "attention" in cls_name and "cross" in name.lower():
        return True
    if "attention" in cls_name and "attn2" in name.lower():
        return True
    # Generic transformer cross-attention
    if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
        # Heuristic: if module name suggests cross-attention
        if "attn2" in name or "cross" in name or "encoder_attn" in name:
            return True
    return False


def _approximate_attention_map(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Approximate attention map by computing Q @ K^T from module weights.

    Args:
        module: The attention module (must have to_q, to_k attributes).
        hidden_states: Query input (batch, seq_q, dim).
        encoder_hidden_states: Key input (batch, seq_k, dim).

    Returns:
        Attention probabilities (batch, heads, seq_q, seq_k) or None.
    """
    if not hasattr(module, 'to_q') or not hasattr(module, 'to_k'):
        return None

    with torch.no_grad():
        q = module.to_q(hidden_states)
        k = module.to_k(encoder_hidden_states)

        # Determine number of heads
        heads = getattr(module, 'heads', None)
        if heads is None:
            heads = getattr(module, 'num_heads', 1)

        batch_size, seq_len, inner_dim = q.shape
        head_dim = inner_dim // heads

        q = q.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        attn_probs = _compute_cross_attention(q, k)
        return attn_probs


def _compute_token_importance(
    attention_maps: dict[str, torch.Tensor],
) -> Optional[np.ndarray]:
    """Compute per-token importance scores aggregated across all layers.

    Returns:
        1D numpy array of importance scores per token, or None.
    """
    all_scores = []

    for name, attn in attention_maps.items():
        if attn.dim() < 3:
            continue
        # Average over batch, heads, and spatial dimensions to get per-token score
        if attn.dim() == 4:
            # (batch, heads, spatial, text) -> (text,)
            scores = attn.float().mean(dim=(0, 1, 2)).numpy()
        else:
            # (batch, spatial, text) -> (text,)
            scores = attn.float().mean(dim=(0, 1)).numpy()
        all_scores.append(scores)

    if not all_scores:
        return None

    # Pad to same length and average
    max_len = max(s.shape[0] for s in all_scores)
    padded = []
    for s in all_scores:
        if s.shape[0] < max_len:
            s = np.pad(s, (0, max_len - s.shape[0]))
        padded.append(s)

    return np.mean(padded, axis=0)


def _apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Apply a colormap to a normalized [0, 1] heatmap.

    Args:
        heatmap: 2D numpy array with values in [0, 1].
        colormap: Name of colormap ("jet", "hot", "viridis").

    Returns:
        RGB image as uint8 numpy array of shape (H, W, 3).
    """
    # Simple built-in colormaps without matplotlib dependency
    h = np.clip(heatmap, 0, 1)

    if colormap == "hot":
        r = np.clip(h * 3.0, 0, 1)
        g = np.clip(h * 3.0 - 1.0, 0, 1)
        b = np.clip(h * 3.0 - 2.0, 0, 1)
    elif colormap == "viridis":
        # Simplified viridis approximation
        r = np.clip(0.267 + h * 0.735 * h, 0, 1)
        g = np.clip(0.004 + h * (1.0 - 0.15 * h), 0, 1)
        b = np.clip(0.329 + h * (0.4 - 0.73 * h), 0, 1)
    else:  # jet
        r = np.clip(1.5 - np.abs(h * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(h * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(h * 4 - 1), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def analyze_attention_for_concept(
    attention_maps: dict[str, torch.Tensor],
    token_indices: list[int],
    threshold: float = 0.1,
) -> dict:
    """Analyze whether attention focuses on concept tokens.

    Useful for diagnosing why a concept isn't being learned — checks
    if the model's cross-attention is actually attending to the trigger
    word tokens.

    Args:
        attention_maps: Captured attention maps from AttentionMapDebugger.
        token_indices: Indices of concept/trigger tokens to check.
        threshold: Minimum attention fraction to consider "focused".

    Returns:
        Dict with diagnostic info:
        - concept_attention_ratio: fraction of attention on concept tokens
        - per_layer_ratios: dict of per-layer attention ratios
        - is_focused: whether attention exceeds threshold
        - diagnosis: human-readable diagnosis string
    """
    if not attention_maps or not token_indices:
        return {
            "concept_attention_ratio": 0.0,
            "per_layer_ratios": {},
            "is_focused": False,
            "diagnosis": "No attention maps or token indices provided.",
        }

    per_layer = {}
    total_concept_attn = 0.0
    total_attn = 0.0

    for name, attn in attention_maps.items():
        if attn.dim() < 3:
            continue

        if attn.dim() == 4:
            # (batch, heads, spatial, text) -> average over batch, heads, spatial
            per_token = attn.float().mean(dim=(0, 1, 2))  # (text,)
        else:
            per_token = attn.float().mean(dim=(0, 1))

        total = per_token.sum().item()
        if total == 0:
            continue

        concept_sum = sum(
            per_token[i].item()
            for i in token_indices
            if i < len(per_token)
        )
        ratio = concept_sum / total
        per_layer[name] = ratio
        total_concept_attn += concept_sum
        total_attn += total

    overall_ratio = total_concept_attn / total_attn if total_attn > 0 else 0.0
    is_focused = overall_ratio >= threshold

    if overall_ratio < 0.05:
        diagnosis = (
            "Very low attention on concept tokens. The model is barely "
            "attending to your trigger words. Consider: (1) increasing "
            "token weights, (2) simplifying captions, (3) training longer."
        )
    elif overall_ratio < threshold:
        diagnosis = (
            f"Moderate attention ({overall_ratio:.1%}) on concept tokens, "
            f"below threshold ({threshold:.1%}). The model is learning "
            f"but may need more epochs or higher token weights."
        )
    else:
        diagnosis = (
            f"Good attention focus ({overall_ratio:.1%}) on concept tokens. "
            f"The model appears to be learning the concept well."
        )

    return {
        "concept_attention_ratio": overall_ratio,
        "per_layer_ratios": per_layer,
        "is_focused": is_focused,
        "diagnosis": diagnosis,
    }
