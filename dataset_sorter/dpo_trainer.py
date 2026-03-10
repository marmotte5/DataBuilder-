"""DPO (Direct Preference Optimization) trainer for image generation.

Implements the DPO loss for fine-tuning diffusion models based on human
preference data. Supports three loss variants:
- Sigmoid: standard DPO (Rafailov et al. 2023)
- Hinge: max-margin preference loss
- IPO: Identity Preference Optimization (robust to noisy labels)

The core idea: given a pair of images (chosen, rejected) for the same prompt,
compute the implicit reward difference under the trained model vs. a frozen
reference model, and optimise the model to prefer the chosen image.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── DPO Loss Functions ────────────────────────────────────────────────

def dpo_loss_sigmoid(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Standard sigmoid DPO loss (Rafailov et al. 2023).

    L = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected)
                             - log_ref(chosen) + log_ref(rejected))))
    """
    log_ratio = beta * (
        (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
    )

    if label_smoothing > 0:
        # Soft labels: (1 - eps) * loss_chosen + eps * loss_rejected
        loss = (
            (1 - label_smoothing) * -F.logsigmoid(log_ratio)
            + label_smoothing * -F.logsigmoid(-log_ratio)
        )
    else:
        loss = -F.logsigmoid(log_ratio)

    return loss.mean()


def dpo_loss_hinge(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Hinge (max-margin) DPO loss.

    L = max(0, 1 - beta * (r_chosen - r_rejected))
    """
    log_ratio = beta * (
        (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
    )
    loss = torch.clamp(1.0 - log_ratio, min=0.0)
    return loss.mean()


def dpo_loss_ipo(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """IPO (Identity Preference Optimization) loss — robust to noise.

    L = (log_ratio - 1/(2*beta))^2
    """
    log_ratio = (
        (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
    )
    loss = (log_ratio - 1.0 / (2.0 * beta)) ** 2
    return loss.mean()


DPO_LOSS_FNS = {
    "sigmoid": dpo_loss_sigmoid,
    "hinge": dpo_loss_hinge,
    "ipo": dpo_loss_ipo,
}


# ── Preference Data ──────────────────────────────────────────────────

@dataclass
class PreferencePair:
    """A single preference annotation: chosen image preferred over rejected."""
    prompt: str
    chosen_path: str       # Path to the preferred image
    rejected_path: str     # Path to the dispreferred image
    step: int = 0          # Training step when preference was collected
    round_idx: int = 0     # RLHF round index
    metadata: dict = field(default_factory=dict)


class PreferenceStore:
    """Manages human preference data for RLHF/DPO training.

    Stores preferences as JSON for persistence and provides iteration
    for DPO training loops.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.preferences_dir = output_dir / "rlhf_preferences"
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        self._pairs: list[PreferencePair] = []
        self._load_existing()

    def _load_existing(self):
        """Load any existing preference data from disk."""
        prefs_file = self.preferences_dir / "preferences.json"
        if not prefs_file.exists():
            return
        try:
            data = json.loads(prefs_file.read_text(encoding="utf-8"))
            for entry in data.get("pairs", []):
                self._pairs.append(PreferencePair(
                    prompt=entry["prompt"],
                    chosen_path=entry["chosen_path"],
                    rejected_path=entry["rejected_path"],
                    step=entry.get("step", 0),
                    round_idx=entry.get("round_idx", 0),
                    metadata=entry.get("metadata", {}),
                ))
            log.info(f"Loaded {len(self._pairs)} existing preference pairs.")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning(f"Could not load preferences: {e}")

    def add_pair(self, pair: PreferencePair):
        """Add a preference pair and persist to disk."""
        self._pairs.append(pair)
        self._save()

    def add_pairs(self, pairs: list[PreferencePair]):
        """Add multiple preference pairs and persist."""
        self._pairs.extend(pairs)
        self._save()

    def _save(self):
        """Write all preferences to disk."""
        data = {
            "total_pairs": len(self._pairs),
            "pairs": [
                {
                    "prompt": p.prompt,
                    "chosen_path": p.chosen_path,
                    "rejected_path": p.rejected_path,
                    "step": p.step,
                    "round_idx": p.round_idx,
                    "metadata": p.metadata,
                }
                for p in self._pairs
            ],
        }
        prefs_file = self.preferences_dir / "preferences.json"
        try:
            prefs_file.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError as e:
            log.warning(f"Could not save preferences: {e}")

    @property
    def pairs(self) -> list[PreferencePair]:
        return list(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def get_pairs_for_round(self, round_idx: int) -> list[PreferencePair]:
        """Get pairs from a specific RLHF round."""
        return [p for p in self._pairs if p.round_idx == round_idx]


# ── DPO Training Step ────────────────────────────────────────────────

def compute_image_log_probs(
    backend,
    latents: torch.Tensor,
    te_out: tuple,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute log-probability of an image under the model.

    For diffusion models, this is approximated as the negative MSE loss
    between the predicted noise and the actual noise at given timesteps.
    Lower loss = higher implicit "log-probability" under the model.
    """
    noisy_latents = backend.noise_scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden = te_out[0]
    pooled = te_out[1] if len(te_out) > 1 else None
    bsz = latents.shape[0]

    added_cond = backend.get_added_cond(bsz, pooled=pooled, te_out=te_out)

    fwd_kwargs = {}
    if added_cond is not None:
        fwd_kwargs["added_cond_kwargs"] = added_cond

    noise_pred = backend.unet(
        noisy_latents, timesteps, encoder_hidden,
        **fwd_kwargs,
    ).sample

    # Negative MSE as log-probability proxy
    per_sample_loss = F.mse_loss(
        noise_pred.float(), noise.float(), reduction="none"
    ).mean(dim=list(range(1, len(noise_pred.shape))))

    return -per_sample_loss  # Higher = model assigns higher probability


def dpo_training_step(
    backend,
    chosen_latents: torch.Tensor,
    rejected_latents: torch.Tensor,
    te_out: tuple,
    ref_backend,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Execute one DPO training step.

    Args:
        backend: The model being trained (policy)
        chosen_latents: VAE latents of preferred images
        rejected_latents: VAE latents of rejected images
        te_out: Text encoder outputs (shared for both)
        ref_backend: Frozen reference model
        beta: KL penalty coefficient
        loss_type: 'sigmoid', 'hinge', or 'ipo'
        label_smoothing: smoothing for robust training

    Returns:
        DPO loss tensor
    """
    bsz = chosen_latents.shape[0]

    # Share the same noise and timesteps for fair comparison
    noise = torch.randn_like(chosen_latents)
    timesteps = torch.randint(
        0, backend.noise_scheduler.config.num_train_timesteps,
        (bsz,), device=device or chosen_latents.device,
    ).long()

    # Compute log-probs under the trained model (policy)
    chosen_logps = compute_image_log_probs(
        backend, chosen_latents, te_out, noise, timesteps, device, dtype
    )
    rejected_logps = compute_image_log_probs(
        backend, rejected_latents, te_out, noise, timesteps, device, dtype
    )

    # Compute log-probs under the reference model (frozen)
    with torch.no_grad():
        ref_chosen_logps = compute_image_log_probs(
            ref_backend, chosen_latents, te_out, noise, timesteps, device, dtype
        )
        ref_rejected_logps = compute_image_log_probs(
            ref_backend, rejected_latents, te_out, noise, timesteps, device, dtype
        )

    # Compute DPO loss
    loss_fn = DPO_LOSS_FNS.get(loss_type, dpo_loss_sigmoid)
    loss = loss_fn(
        chosen_logps, rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    return loss


# ── Candidate Generation for RLHF ───────────────────────────────────

def generate_candidate_pairs(
    backend,
    prompts: list[str],
    num_pairs: int = 4,
    num_steps: int = 28,
    cfg_scale: float = 7.0,
    device: torch.device = None,
) -> list[dict]:
    """Generate pairs of candidate images for human preference annotation.

    For each prompt, generates 2 images with different seeds.
    Returns list of dicts with {prompt, image_a, image_b, seed_a, seed_b}.
    """
    import random

    candidates = []
    used_prompts = prompts[:num_pairs] if len(prompts) >= num_pairs else (
        prompts * math.ceil(num_pairs / max(1, len(prompts)))
    )[:num_pairs]

    for prompt in used_prompts:
        seed_a = random.randint(0, 2**32 - 1)
        seed_b = random.randint(0, 2**32 - 1)

        img_a = backend.generate_sample(prompt, seed_a)
        img_b = backend.generate_sample(prompt, seed_b)

        if img_a is not None and img_b is not None:
            candidates.append({
                "prompt": prompt,
                "image_a": img_a,
                "image_b": img_b,
                "seed_a": seed_a,
                "seed_b": seed_b,
            })

    return candidates
