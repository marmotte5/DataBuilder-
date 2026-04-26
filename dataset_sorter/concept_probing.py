"""Concept Probing & Adaptive Tag Weighting — automated concept learning optimization.

Three interconnected systems for ensuring every concept is well-learned:

1. **ConceptProber**: Pre-training analysis that generates images from the base
   model using each concept tag, then measures CLIP similarity to assess which
   concepts the model already knows vs. doesn't know. Unknown concepts get
   automatically boosted weights.

2. **AdaptiveTagWeighter**: During training, tracks per-tag loss decomposition
   across steps. Tags where the model keeps failing get their weight dynamically
   increased; tags already learned get reduced. This enables "focus more on blue
   tire than red car" automatically.

3. **AttentionGuidedRebalancer**: Uses cross-attention maps to detect which
   caption tokens the model is ignoring, and boosts those tokens' weights
   in real-time. Works with the existing AttentionMapDebugger.

Usage:

    # Pre-training: probe base model knowledge
    prober = ConceptProber(pipeline, device)
    knowledge = prober.probe_concepts(["blue tire", "red car", "sks person"])
    # knowledge["blue tire"].score = 0.12  (unknown!)
    # knowledge["red car"].score = 0.85    (already known)

    # Convert to initial tag weights
    weights = prober.compute_initial_weights(knowledge)
    # weights = {"blue tire": 3.5, "red car": 1.0, "sks person": 2.8}

    # During training: adaptive weighting
    weighter = AdaptiveTagWeighter(initial_weights=weights)
    for step, batch in enumerate(dataloader):
        loss = training_step(batch)
        per_tag_losses = decompose_loss_by_tag(batch, loss)
        weighter.update(per_tag_losses)
        current_weights = weighter.get_weights()  # dynamically adjusted
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONCEPT PROBING — Base Model Knowledge Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConceptKnowledge:
    """Result of probing a single concept."""
    concept: str
    similarity_score: float = 0.0  # 0.0 = unknown, 1.0 = perfectly known
    generated_quality: float = 0.0  # How well the base model generates this concept
    is_known: bool = False  # Whether the model already knows this concept
    suggested_weight: float = 1.0  # Suggested training weight


@dataclass
class ProbeResult:
    """Complete probing results for all concepts."""
    concepts: dict[str, ConceptKnowledge] = field(default_factory=dict)
    unknown_concepts: list[str] = field(default_factory=list)
    known_concepts: list[str] = field(default_factory=list)
    suggested_weights: dict[str, float] = field(default_factory=dict)


class ConceptProber:
    """Analyze what the base model knows before training starts.

    Strategy:
    1. For each concept tag, generate N images from the base model using
       that tag as the prompt.
    2. Use CLIP to measure text-image similarity between the concept tag
       and the generated images.
    3. Low similarity = the model doesn't know this concept = needs higher
       training weight.

    This works because if a model already knows "red car", it will generate
    good red cars. If it doesn't know "sks person", the generated images
    will be random and have low CLIP similarity to the text "sks person".
    """

    def __init__(
        self,
        device: torch.device,
        num_images_per_concept: int = 2,
        num_inference_steps: int = 15,
        knowledge_threshold: float = 0.25,
        max_weight: float = 5.0,
        min_weight: float = 0.5,
    ):
        self.device = device
        self.num_images = num_images_per_concept
        self.num_steps = num_inference_steps
        self.knowledge_threshold = knowledge_threshold
        self.max_weight = max_weight
        self.min_weight = min_weight

    @torch.no_grad()
    def probe_concepts(
        self,
        concepts: list[str],
        pipeline,
        tokenizer=None,
        text_encoder=None,
        progress_fn=None,
    ) -> ProbeResult:
        """Probe the base model's knowledge of each concept.

        Args:
            concepts: List of concept tags to probe.
            pipeline: Diffusion pipeline for image generation.
            tokenizer: CLIP tokenizer (optional, for similarity scoring).
            text_encoder: CLIP text encoder (optional, for similarity scoring).
            progress_fn: Optional callback(current, total, message).

        Returns:
            ProbeResult with per-concept knowledge scores and suggested weights.
        """
        result = ProbeResult()

        # If we have CLIP components, use text-image similarity
        # Otherwise fall back to generation-quality heuristic
        use_clip = tokenizer is not None and text_encoder is not None

        for idx, concept in enumerate(concepts):
            if progress_fn:
                progress_fn(idx, len(concepts), f"Probing: {concept}")

            knowledge = self._probe_single_concept(
                concept, pipeline, tokenizer, text_encoder, use_clip,
            )
            result.concepts[concept] = knowledge

            if knowledge.is_known:
                result.known_concepts.append(concept)
            else:
                result.unknown_concepts.append(concept)

        # Compute suggested weights
        result.suggested_weights = self._compute_weights(result.concepts)

        if progress_fn:
            progress_fn(len(concepts), len(concepts), "Probing complete")

        log.info(
            f"Concept probing: {len(result.known_concepts)} known, "
            f"{len(result.unknown_concepts)} unknown out of {len(concepts)}"
        )
        return result

    def _probe_single_concept(
        self,
        concept: str,
        pipeline,
        tokenizer,
        text_encoder,
        use_clip: bool,
    ) -> ConceptKnowledge:
        """Probe a single concept via generation + similarity."""
        knowledge = ConceptKnowledge(concept=concept)

        try:
            # Generate images using the concept as prompt
            prompt = f"a photo of {concept}, high quality, detailed"
            images = []
            for i in range(self.num_images):
                gen_device = "cpu" if self.device.type == "mps" else self.device
                img = pipeline(
                    prompt=prompt,
                    num_inference_steps=self.num_steps,
                    guidance_scale=7.0,
                    generator=torch.Generator(device=gen_device).manual_seed(42 + i),
                ).images[0]
                images.append(img)

            if use_clip and images:
                score = self._compute_clip_similarity(
                    concept, images, tokenizer, text_encoder,
                )
                knowledge.similarity_score = score
                knowledge.generated_quality = score
            elif images:
                # Fallback: use image variance as a rough quality heuristic
                # Low-variance images = model produced generic/blurry output = unknown concept
                score = self._compute_quality_heuristic(images)
                knowledge.similarity_score = score
                knowledge.generated_quality = score

            knowledge.is_known = knowledge.similarity_score >= self.knowledge_threshold

        except Exception as e:
            log.warning(f"Failed to probe concept '{concept}': {e}")
            # Assume unknown on failure → higher weight (safe default)
            knowledge.similarity_score = 0.0
            knowledge.is_known = False

        return knowledge

    def _compute_clip_similarity(
        self,
        concept: str,
        images: list,
        tokenizer,
        text_encoder,
    ) -> float:
        """Compute CLIP text-image similarity between concept and generated images.

        Attempts to load a full CLIP model (text + vision) for proper
        text-image similarity scoring. Falls back to quality heuristic
        if CLIP vision encoder is not available.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            log.debug("CLIP model not available, using quality heuristic")
            return self._compute_quality_heuristic(images)

        try:
            # Try to load a lightweight CLIP model for proper text-image similarity
            _clip_repo = "openai/clip-vit-base-patch32"
            try:
                clip_model = CLIPModel.from_pretrained(
                    _clip_repo, torch_dtype=torch.float16, local_files_only=True,
                ).to(self.device).eval()
                clip_processor = CLIPProcessor.from_pretrained(
                    _clip_repo, local_files_only=True,
                )
            except Exception:
                clip_model = CLIPModel.from_pretrained(
                    _clip_repo, torch_dtype=torch.float16,
                ).to(self.device).eval()
                clip_processor = CLIPProcessor.from_pretrained(_clip_repo)

            # Compute text-image similarity using full CLIP
            inputs = clip_processor(
                text=[concept], images=images,
                return_tensors="pt", padding=True,
            )
            # CLIP model is loaded in fp16; pixel_values must match model dtype
            # (input_ids/attention_mask are integer tensors, unaffected)
            inputs = {
                k: v.to(self.device, dtype=torch.float16) if v.is_floating_point()
                   else v.to(self.device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = clip_model(**inputs)
                # outputs.logits_per_text is the temperature-scaled cosine
                # similarity: ``logit_scale.exp() * cosine_sim`` where
                # logit_scale.exp() ≈ 14–100 depending on the CLIP variant.
                # Earlier code used ``logits.sigmoid()``: sigmoid(14) ≈ 1.0
                # and sigmoid(-14) ≈ 0.0 saturate so hard that any "similar"
                # image scored ≈ 1.0 regardless of HOW similar — the probe
                # silently lost almost all gradation. Softmax across images
                # also fails because it forces scores to sum to 1 (a single
                # image always gets 1.0).
                #
                # Use the un-scaled cosine similarity directly: divide by
                # the learned temperature to recover ``cos ∈ [-1, 1]``,
                # then linearly map to ``[0, 1]``. Preserves the ranking
                # AND the relative magnitude of matches.
                logit_scale = float(clip_model.logit_scale.exp().detach())
                cosine = outputs.logits_per_text / max(logit_scale, 1e-6)
                # Cosine in [-1, 1] → [0, 1]
                similarities = (cosine.clamp(-1.0, 1.0) + 1.0) * 0.5
                score = float(similarities.mean())

            # Clean up CLIP model to free VRAM
            del clip_model, clip_processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return score

        except Exception as e:
            log.debug(f"CLIP similarity failed ({e}), using quality heuristic")
            return self._compute_quality_heuristic(images)

    def _compute_quality_heuristic(self, images: list) -> float:
        """Heuristic quality score based on image characteristics.

        High-quality, concept-specific images have:
        - High color variance (not blurry/gray)
        - Sharp edges (not blurred out)
        - Diverse content across seeds (not mode-collapsed)

        Returns score in [0, 1] where higher = model knows the concept better.
        """
        if not images:
            return 0.0

        scores = []
        for img in images:
            arr = np.array(img.resize((64, 64))).astype(np.float32) / 255.0

            # Color variance: diverse colors = more specific generation
            color_var = arr.var(axis=(0, 1)).mean()

            # Spatial variance: high = sharp/detailed, low = blurry/uniform
            spatial_var = arr.var(axis=(0,)).mean() + arr.var(axis=(1,)).mean()

            # Edge density: Sobel-like gradient magnitude
            dx = np.abs(np.diff(arr, axis=1)).mean()
            dy = np.abs(np.diff(arr, axis=0)).mean()
            edge_density = (dx + dy) / 2

            # Combined score (each in roughly [0, 0.3] range)
            score = min(1.0, (color_var * 3.0 + spatial_var * 1.5 + edge_density * 3.0))
            scores.append(score)

        # Cross-image consistency: if images are similar to each other despite
        # different seeds, the model has a coherent concept representation
        if len(scores) >= 2:
            consistency = 1.0 - np.std(scores) / max(np.mean(scores), 1e-6)
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5

        return float(np.mean(scores) * 0.7 + consistency * 0.3)

    def _compute_weights(
        self,
        concepts: dict[str, ConceptKnowledge],
    ) -> dict[str, float]:
        """Convert knowledge scores to training weights.

        Unknown concepts → high weight (model needs to learn them).
        Known concepts → normal weight (maintain existing knowledge).

        Weight = max_weight * (1 - similarity_score)^2 + min_weight

        The quadratic curve aggressively boosts unknown concepts:
        - score=0.0 (unknown) → weight=max_weight
        - score=0.5 (partial) → weight≈0.75*max
        - score=1.0 (known)   → weight=min_weight
        """
        weights = {}
        for concept, knowledge in concepts.items():
            s = knowledge.similarity_score
            w = self.max_weight * (1.0 - s) ** 2 + self.min_weight
            w = max(self.min_weight, min(self.max_weight, w))
            knowledge.suggested_weight = w
            weights[concept] = w
        return weights


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ADAPTIVE TAG WEIGHTER — Dynamic Per-Concept Loss Weighting
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveTagWeighter:
    """Dynamically adjusts per-tag training weights based on loss feedback.

    Tracks per-tag loss across training steps using EMA. Tags with
    persistently high loss (poorly learned) get weight increased;
    tags with low loss (already learned) get weight decreased.

    This implements "focus more on blue tire than red car" automatically:
    if the model quickly learns "red car" but struggles with "blue tire",
    the weighter will increase the weight for "blue tire" tokens.

    The adjustment is smooth and bounded to prevent instability.

    Integration:
    1. After each training step, call update() with per-tag losses.
    2. Before each step, call get_weight_for_caption() to get a weight
       mask that can multiply the token-level loss.
    """

    def __init__(
        self,
        initial_weights: dict[str, float] | None = None,
        momentum: float = 0.95,
        min_weight: float = 0.3,
        max_weight: float = 5.0,
        adjustment_rate: float = 0.5,
        warmup_steps: int = 50,
    ):
        """Initialize the adaptive weighter.

        Args:
            initial_weights: Pre-computed weights from ConceptProber (optional).
            momentum: EMA momentum for loss tracking (higher = smoother).
            min_weight: Minimum allowed weight for any tag.
            max_weight: Maximum allowed weight for any tag.
            adjustment_rate: How aggressively to adjust weights (0-1).
            warmup_steps: Steps of uniform weighting before adapting.
        """
        self._initial_weights = dict(initial_weights or {})
        self._momentum = momentum
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._adjustment_rate = adjustment_rate
        self._warmup_steps = warmup_steps

        # Per-tag EMA of loss
        self._loss_ema: dict[str, float] = {}
        # Per-tag step count (how many times we've seen this tag)
        self._seen_count: dict[str, int] = defaultdict(int)
        # Current adaptive weights
        self._weights: dict[str, float] = dict(self._initial_weights)
        # Global step counter
        self._step = 0

    def update(self, per_tag_losses: dict[str, float]):
        """Update per-tag loss EMA after a training step.

        Args:
            per_tag_losses: Dict mapping tag -> loss value for tags
                present in this batch's captions.
        """
        self._step += 1

        for tag, loss_val in per_tag_losses.items():
            if tag not in self._loss_ema:
                self._loss_ema[tag] = loss_val
            else:
                self._loss_ema[tag] = (
                    self._momentum * self._loss_ema[tag]
                    + (1.0 - self._momentum) * loss_val
                )
            self._seen_count[tag] += 1

        # Recompute adaptive weights periodically (every 10 steps after warmup)
        if self._step >= self._warmup_steps and self._step % 10 == 0:
            self._recompute_weights()

    def _recompute_weights(self):
        """Recompute per-tag weights based on current loss EMA."""
        if not self._loss_ema:
            return

        # Compute mean loss across all tracked tags
        all_losses = list(self._loss_ema.values())
        mean_loss = np.mean(all_losses) if all_losses else 1.0
        std_loss = np.std(all_losses) if len(all_losses) > 1 else 0.0

        if mean_loss < 1e-8:
            return  # All losses near zero, nothing to rebalance

        for tag, ema_loss in self._loss_ema.items():
            # Relative loss: how much harder is this tag vs average?
            # z-score: positive = harder than average, negative = easier
            if std_loss > 1e-8:
                z_score = (ema_loss - mean_loss) / std_loss
            else:
                z_score = 0.0

            # Convert z-score to weight adjustment:
            # z=+2 (very hard) → boost significantly
            # z=0  (average)   → no change
            # z=-2 (very easy) → reduce
            adjustment = 1.0 + self._adjustment_rate * np.tanh(z_score)

            # Apply adjustment to initial weight (or 1.0 if no initial)
            base = self._initial_weights.get(tag, 1.0)
            new_weight = base * adjustment

            # Clamp to bounds
            self._weights[tag] = max(self._min_weight, min(self._max_weight, new_weight))

    def get_weight(self, tag: str) -> float:
        """Get the current adaptive weight for a single tag."""
        if self._step < self._warmup_steps:
            return self._initial_weights.get(tag, 1.0)
        return self._weights.get(tag, self._initial_weights.get(tag, 1.0))

    def get_all_weights(self) -> dict[str, float]:
        """Get all current adaptive weights."""
        if self._step < self._warmup_steps:
            return dict(self._initial_weights)
        return dict(self._weights)

    def get_caption_weight(
        self,
        caption: str,
        deleted_tags: set[str] | None = None,
    ) -> float:
        """Compute an effective weight for an entire caption.

        The caption's weight is the mean of its tags' weights, giving
        higher overall loss to captions containing hard-to-learn concepts.

        Args:
            caption: Comma-separated tag string.
            deleted_tags: Tags to skip.

        Returns:
            Effective caption weight (>1 = boost, <1 = reduce).
        """
        tags = [t.strip() for t in caption.split(",") if t.strip()]
        if not tags:
            return 1.0

        weights = []
        for tag in tags:
            if deleted_tags and tag in deleted_tags:
                continue
            weights.append(self.get_weight(tag))

        return float(np.mean(weights)) if weights else 1.0

    def get_stats(self) -> dict:
        """Return diagnostic statistics for logging."""
        if not self._loss_ema:
            return {
                "step": self._step,
                "active": self._step >= self._warmup_steps,
                "tracked_tags": 0,
            }

        all_weights = list(self._weights.values()) if self._weights else [1.0]
        all_losses = list(self._loss_ema.values())

        # Top 5 hardest and easiest tags
        sorted_by_loss = sorted(self._loss_ema.items(), key=lambda x: x[1], reverse=True)
        hardest = sorted_by_loss[:5]
        easiest = sorted_by_loss[-5:] if len(sorted_by_loss) > 5 else []

        return {
            "step": self._step,
            "active": self._step >= self._warmup_steps,
            "tracked_tags": len(self._loss_ema),
            "weight_mean": float(np.mean(all_weights)),
            "weight_min": float(np.min(all_weights)),
            "weight_max": float(np.max(all_weights)),
            "loss_mean": float(np.mean(all_losses)),
            "loss_std": float(np.std(all_losses)) if len(all_losses) > 1 else 0.0,
            "hardest_tags": [(t, round(l, 4)) for t, l in hardest],
            "easiest_tags": [(t, round(l, 4)) for t, l in easiest],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION-GUIDED REBALANCER — Focus on Ignored Tokens
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionGuidedRebalancer:
    """Detects which caption tokens the model ignores and boosts their weight.

    Uses cross-attention maps from the AttentionMapDebugger to identify
    tokens that receive very low attention (the model is "not looking at"
    parts of the caption). These tokens get their loss weight boosted.

    This solves the "blue tire" problem directly: if the model attends
    strongly to "red car" but ignores "blue tire", this rebalancer
    will detect the imbalance and boost "blue tire" tokens.

    Works in conjunction with the existing TokenLossWeighter by providing
    dynamic attention-based adjustments on top of static/adaptive weights.
    """

    def __init__(
        self,
        tokenizer=None,
        attention_threshold: float = 0.15,
        boost_factor: float = 2.0,
        ema_momentum: float = 0.9,
        update_every_n_steps: int = 5,
    ):
        """Initialize the rebalancer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding token IDs.
            attention_threshold: Tokens with attention below this fraction
                of the mean are considered "ignored".
            boost_factor: How much to boost ignored tokens (multiplicative).
            ema_momentum: Smoothing for attention tracking.
            update_every_n_steps: How often to recompute boosts.
        """
        self.tokenizer = tokenizer
        self.attention_threshold = attention_threshold
        self.boost_factor = boost_factor
        self.ema_momentum = ema_momentum
        self.update_every = update_every_n_steps

        # Per-token attention EMA (token_text -> attention_score)
        self._token_attention_ema: dict[str, float] = {}
        self._token_boost: dict[str, float] = {}
        self._step = 0

    def update_from_attention_maps(
        self,
        attention_maps: dict[str, torch.Tensor],
        caption: str,
    ):
        """Update token attention tracking from captured attention maps.

        Args:
            attention_maps: Dict from AttentionMapDebugger.get_attention_maps().
            caption: The caption that was used for this forward pass.
        """
        self._step += 1

        if not attention_maps or self.tokenizer is None:
            return

        # Compute per-token attention scores
        token_scores = self._compute_per_token_attention(attention_maps)
        if token_scores is None:
            return

        # Decode tokens to text for tag-level tracking
        tokens = self.tokenizer(
            caption, padding=False, truncation=True,
            max_length=77, return_tensors="pt",
        )
        token_ids = tokens.input_ids[0].tolist()

        for i, tid in enumerate(token_ids):
            if i >= len(token_scores):
                break
            if tid == self.tokenizer.pad_token_id:
                continue

            token_text = self.tokenizer.decode([tid]).strip()
            if not token_text:
                continue

            score = float(token_scores[i])
            if token_text not in self._token_attention_ema:
                self._token_attention_ema[token_text] = score
            else:
                self._token_attention_ema[token_text] = (
                    self.ema_momentum * self._token_attention_ema[token_text]
                    + (1.0 - self.ema_momentum) * score
                )

        # Recompute boosts periodically
        if self._step % self.update_every == 0:
            self._recompute_boosts()

    def _compute_per_token_attention(
        self,
        attention_maps: dict[str, torch.Tensor],
    ) -> Optional[np.ndarray]:
        """Aggregate attention across layers to get per-token scores."""
        all_scores = []

        for name, attn in attention_maps.items():
            if attn.dim() < 3:
                continue
            if attn.dim() == 4:
                # (batch, heads, spatial, text) -> (text,)
                scores = attn.float().mean(dim=(0, 1, 2)).cpu().numpy()
            else:
                scores = attn.float().mean(dim=(0, 1)).cpu().numpy()
            all_scores.append(scores)

        if not all_scores:
            return None

        # Pad and average
        max_len = max(s.shape[0] for s in all_scores)
        padded = [np.pad(s, (0, max_len - s.shape[0])) for s in all_scores]
        return np.mean(padded, axis=0)

    def _recompute_boosts(self):
        """Recompute per-token boost factors based on attention EMA."""
        if not self._token_attention_ema:
            return

        all_scores = list(self._token_attention_ema.values())
        mean_attn = np.mean(all_scores) if all_scores else 1.0

        if mean_attn < 1e-8:
            return

        threshold = mean_attn * self.attention_threshold

        self._token_boost.clear()
        for token, score in self._token_attention_ema.items():
            if score < threshold:
                # Token is being ignored → boost its weight
                # Boost strength proportional to how ignored it is
                deficit = (threshold - score) / threshold
                boost = 1.0 + self.boost_factor * deficit
                self._token_boost[token] = min(boost, self.boost_factor + 1.0)
            else:
                self._token_boost[token] = 1.0

    def get_token_boost(self, token_text: str) -> float:
        """Get the attention-based boost for a specific token."""
        return self._token_boost.get(token_text, 1.0)

    def get_caption_boosts(self, caption: str) -> dict[str, float]:
        """Get per-tag boosts for all tags in a caption.

        Returns dict mapping tag -> boost_factor. Tags being ignored
        by the model get boost > 1.0.
        """
        tags = [t.strip() for t in caption.split(",") if t.strip()]
        boosts = {}
        for tag in tags:
            # A tag may span multiple tokens; use the max boost among them
            tag_tokens = tag.lower().split()
            tag_boost = max(
                (self._token_boost.get(t, 1.0) for t in tag_tokens),
                default=1.0,
            )
            boosts[tag] = tag_boost
        return boosts

    def get_stats(self) -> dict:
        """Return diagnostic info about attention-based boosts."""
        if not self._token_boost:
            return {"step": self._step, "tracked_tokens": 0, "boosted_tokens": 0}

        boosted = {t: b for t, b in self._token_boost.items() if b > 1.05}
        return {
            "step": self._step,
            "tracked_tokens": len(self._token_boost),
            "boosted_tokens": len(boosted),
            "top_boosted": sorted(
                boosted.items(), key=lambda x: x[1], reverse=True,
            )[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOSS DECOMPOSITION — Per-Tag Loss from Batch Loss
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_loss_by_tag(
    captions: list[str],
    per_sample_losses: torch.Tensor | list[float],
    deleted_tags: set[str] | None = None,
) -> dict[str, float]:
    """Decompose per-sample batch losses into per-tag loss estimates.

    Each sample has a set of tags in its caption. We attribute the
    sample's loss equally to all its tags (uniform attribution).

    For more accurate attribution, use attention maps to weight
    each tag's contribution (see AttentionGuidedRebalancer).

    Args:
        captions: List of comma-separated caption strings.
        per_sample_losses: Per-sample loss values (one per caption).
        deleted_tags: Tags to ignore.

    Returns:
        Dict mapping tag -> estimated loss for that tag.
    """
    if isinstance(per_sample_losses, torch.Tensor):
        losses = per_sample_losses.detach().cpu().tolist()
    else:
        losses = list(per_sample_losses)

    tag_loss_sum: dict[str, float] = defaultdict(float)
    tag_count: dict[str, int] = defaultdict(int)
    deleted = deleted_tags or set()

    for caption, loss_val in zip(captions, losses):
        tags = [t.strip() for t in caption.split(",") if t.strip()]
        active_tags = [t for t in tags if t not in deleted]
        if not active_tags:
            continue

        # Uniform attribution: each tag gets equal share of the loss
        per_tag = loss_val / len(active_tags)
        for tag in active_tags:
            tag_loss_sum[tag] += per_tag
            tag_count[tag] += 1

    # Average across batch
    return {
        tag: tag_loss_sum[tag] / max(tag_count[tag], 1)
        for tag in tag_loss_sum
    }


def compute_combined_weight(
    tag: str,
    adaptive_weighter: Optional[AdaptiveTagWeighter] = None,
    attention_rebalancer: Optional[AttentionGuidedRebalancer] = None,
    probe_weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute the final combined weight for a tag from all three systems.

    Multiplies weights from:
    1. Concept probing (pre-training knowledge gaps)
    2. Adaptive tag weighting (training-time loss feedback)
    3. Attention-guided rebalancing (attention map feedback)

    The multiplicative combination means a tag that is both unknown
    (probe=3x) and being ignored (attention=2x) gets a 6x boost,
    which is exactly what we want — maximum focus on concepts
    that the model both doesn't know AND isn't learning.
    """
    weight = 1.0

    # Layer 1: Probe-based initial weight
    if probe_weights and tag in probe_weights:
        weight *= probe_weights[tag]

    # Layer 2: Adaptive loss-based weight
    if adaptive_weighter is not None:
        weight *= adaptive_weighter.get_weight(tag)

    # Layer 3: Attention-based boost
    if attention_rebalancer is not None:
        weight *= attention_rebalancer.get_token_boost(tag)

    return max(0.1, min(10.0, weight))
