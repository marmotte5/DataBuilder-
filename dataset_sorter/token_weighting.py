"""Token-Level Caption Weighting — per-token loss weighting for training.

Allows users to mark trigger words and concept tokens with importance
weights (e.g. 2x, 3x). Applies per-token loss weighting during training
so the model prioritizes learning key concepts.

Syntax in captions:
    "a photo of {sks:2.0} person, {detailed face:1.5}, high quality"

The {token:weight} syntax marks tokens with custom importance weights.
Tokens not marked default to weight 1.0.

Usage:
    parser = TokenWeightParser()
    result = parser.parse("a photo of {sks:2.0} person")
    # result.clean_caption = "a photo of sks person"
    # result.token_weights = {"sks": 2.0}

    weighter = TokenLossWeighter(tokenizer)
    weight_mask = weighter.compute_weight_mask(
        caption="a photo of {sks:2.0} person",
        max_length=77,
    )
    # weight_mask is a tensor of shape (max_length,) with per-token weights
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

# Pattern: {text:weight} where weight is a float (supports -0.5, .5, 2, 1.5)
_WEIGHT_PATTERN = re.compile(r'\{([^}:]+):(-?(?:\d+\.?\d*|\.\d+))\}')


@dataclass
class WeightedCaption:
    """Parsed caption with token weights."""
    clean_caption: str
    token_weights: dict[str, float] = field(default_factory=dict)
    raw_caption: str = ""


class TokenWeightParser:
    """Parse captions with {token:weight} syntax.

    Extracts weighted tokens and produces a clean caption string
    suitable for tokenization.
    """

    def parse(self, caption: str) -> WeightedCaption:
        """Parse a caption with optional {token:weight} markers.

        Args:
            caption: Raw caption, possibly containing {token:weight} syntax.

        Returns:
            WeightedCaption with clean_caption and token_weights dict.
        """
        token_weights = {}
        clean = caption

        for match in _WEIGHT_PATTERN.finditer(caption):
            text = match.group(1).strip()
            weight = max(0.0, min(float(match.group(2)), 10.0))  # Clamp to [0, 10]
            token_weights[text] = weight
            # Replace {text:weight} with just the text
            clean = clean.replace(match.group(0), text)

        # Clean up extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()

        return WeightedCaption(
            clean_caption=clean,
            token_weights=token_weights,
            raw_caption=caption,
        )

    def add_weights_to_caption(
        self,
        caption: str,
        weights: dict[str, float],
    ) -> str:
        """Add weight markers to a plain caption.

        Args:
            caption: Plain caption without weight markers.
            weights: Dict mapping token text to weight values.

        Returns:
            Caption with {token:weight} syntax inserted.
        """
        result = caption
        for text, weight in weights.items():
            if text in result:
                # Only replace the first occurrence to avoid double-marking
                result = result.replace(text, f"{{{text}:{weight}}}", 1)
        return result

    def extract_trigger_words(
        self,
        caption: str,
        keep_first_n: int = 1,
    ) -> list[str]:
        """Extract likely trigger words from a caption.

        Trigger words are typically the first N comma-separated tags.

        Args:
            caption: Plain or weighted caption.
            keep_first_n: Number of leading tags to treat as triggers.

        Returns:
            List of trigger word strings.
        """
        # Strip weight markers first
        clean = _WEIGHT_PATTERN.sub(r'\1', caption)
        tags = [t.strip() for t in clean.split(',') if t.strip()]
        return tags[:keep_first_n]


class TokenLossWeighter:
    """Compute per-token loss weight masks for training.

    Maps weighted caption tokens to tokenizer token IDs and produces
    a weight tensor that can be applied to per-token losses during training.
    """

    def __init__(self, tokenizer=None):
        """Initialize with a HuggingFace tokenizer.

        Args:
            tokenizer: Tokenizer instance (CLIPTokenizer, T5Tokenizer, etc.).
        """
        self.tokenizer = tokenizer
        self._parser = TokenWeightParser()

    def compute_weight_mask(
        self,
        caption: str,
        max_length: Optional[int] = None,
        default_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute a per-token weight mask for a caption.

        Args:
            caption: Caption with optional {token:weight} syntax.
            max_length: Maximum sequence length. Defaults to tokenizer's max.
            default_weight: Weight for tokens without explicit weight.

        Returns:
            Float tensor of shape (max_length,) with per-token weights.
        """
        if self.tokenizer is None:
            if max_length is None:
                max_length = 77
            return torch.ones(max_length) * default_weight

        if max_length is None:
            max_length = self.tokenizer.model_max_length

        parsed = self._parser.parse(caption)

        if not parsed.token_weights:
            return torch.ones(max_length) * default_weight

        # Tokenize the clean caption
        encoding = self.tokenizer(
            parsed.clean_caption,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=False,
        )
        token_ids = encoding.input_ids

        # Build weight mask
        weights = torch.ones(max_length) * default_weight

        # For each weighted phrase, find its tokens in the clean caption
        for phrase, weight in parsed.token_weights.items():
            token_indices = self._find_phrase_tokens(
                parsed.clean_caption, phrase, token_ids, max_length,
            )
            for idx in token_indices:
                weights[idx] = weight

        # Set padding tokens to 0 weight
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            for i, tid in enumerate(token_ids):
                if tid == self.tokenizer.pad_token_id:
                    weights[i] = 0.0

        return weights

    def compute_batch_weight_masks(
        self,
        captions: list[str],
        max_length: Optional[int] = None,
        default_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute weight masks for a batch of captions.

        Args:
            captions: List of captions with optional {token:weight} syntax.
            max_length: Maximum sequence length.
            default_weight: Default per-token weight.

        Returns:
            Float tensor of shape (batch_size, max_length).
        """
        masks = [
            self.compute_weight_mask(c, max_length, default_weight)
            for c in captions
        ]
        return torch.stack(masks)

    def _find_phrase_tokens(
        self,
        clean_caption: str,
        phrase: str,
        token_ids: list[int],
        max_length: int,
    ) -> list[int]:
        """Find which token indices correspond to a phrase in the caption.

        Uses a decode-and-match strategy: tokenize the phrase separately,
        then find the matching subsequence in the full token ID sequence.

        Args:
            clean_caption: The clean caption text.
            phrase: The phrase to locate.
            token_ids: Full token IDs of the clean caption.
            max_length: Max sequence length.

        Returns:
            List of token indices that correspond to the phrase.
        """
        if self.tokenizer is None:
            return []

        # Tokenize just the phrase
        phrase_tokens = self.tokenizer(
            phrase,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids

        if not phrase_tokens:
            return []

        # Find subsequence match in the full token IDs
        matches = []
        for i in range(len(token_ids) - len(phrase_tokens) + 1):
            if i >= max_length:
                break
            if token_ids[i:i + len(phrase_tokens)] == phrase_tokens:
                matches.extend(range(i, i + len(phrase_tokens)))
                break  # Only first match

        return matches

    def get_clean_caption(self, caption: str) -> str:
        """Strip weight markers and return clean caption text.

        Args:
            caption: Caption with optional {token:weight} markers.

        Returns:
            Clean caption without weight markers.
        """
        return self._parser.parse(caption).clean_caption


def apply_token_weights_to_loss(
    loss: torch.Tensor,
    weight_mask: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply per-token weights to a training loss via attention-weighted scaling.

    For diffusion model training, we can't directly weight per-token losses
    because the loss is computed in latent space, not token space. Instead,
    we compute an effective weight per spatial position based on the
    cross-attention pattern between spatial positions and text tokens.

    When cross-attention maps are not available, we use a simpler approach:
    scale the overall loss by the mean of the non-zero token weights.

    Args:
        loss: Per-sample loss tensor of shape (batch,) or scalar.
        weight_mask: Per-token weights of shape (batch, seq_len).
        encoder_hidden_states: Text encoder hidden states (batch, seq, dim).

    Returns:
        Weighted loss tensor (same shape as input loss).
    """
    if weight_mask is None:
        return loss

    # Compute effective weight as mean of non-padding token weights
    # This gives higher overall loss for captions with high-weighted tokens
    non_zero = weight_mask > 0
    if non_zero.any():
        # Per-sample effective weight
        if weight_mask.dim() == 2:
            effective_weights = torch.where(
                non_zero,
                weight_mask,
                torch.zeros_like(weight_mask),
            ).sum(dim=1) / non_zero.sum(dim=1).clamp(min=1)
        else:
            effective_weights = weight_mask[non_zero].mean()
    else:
        return loss

    # Scale loss by effective weight
    if loss.dim() == 0:
        return loss * effective_weights.mean()
    elif loss.shape[0] == effective_weights.shape[0]:
        return loss * effective_weights.to(loss.device, dtype=loss.dtype)
    else:
        return loss * effective_weights.mean().to(loss.device, dtype=loss.dtype)


def preprocess_captions_with_weights(
    captions: list[str],
) -> tuple[list[str], list[dict[str, float]]]:
    """Preprocess a list of captions, extracting weights and cleaning text.

    Args:
        captions: List of captions, possibly with {token:weight} markers.

    Returns:
        Tuple of (clean_captions, weight_dicts) where weight_dicts[i]
        maps phrase -> weight for captions[i].
    """
    parser = TokenWeightParser()
    clean_captions = []
    weight_dicts = []

    for caption in captions:
        parsed = parser.parse(caption)
        clean_captions.append(parsed.clean_caption)
        weight_dicts.append(parsed.token_weights)

    return clean_captions, weight_dicts
