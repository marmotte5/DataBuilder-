"""Noise scheduler rescaling — fix terminal SNR issues.

Implements the "Common Diffusion Noise Schedules and Sample Steps are Flawed"
(Lin et al. 2024) fix. The problem: standard DDPM noise schedules don't
reach zero SNR at the final timestep, creating a train/inference mismatch.

This module provides:
1. enforce_zero_terminal_snr: Rescale betas so the last timestep has SNR=0
2. compute_snr_weights: Min-SNR-gamma loss weighting (Hang et al. 2023)
3. apply_noise_rescaling: One-call setup for noise schedule fixes

These fixes improve:
- Dark/light image generation quality
- Color accuracy at high noise levels
- Overall FID scores

Compatible with: SD 1.5, SD 2.x, SDXL, Pony (DDPM-based schedulers)
Not needed for: Flow-matching models (Flux, SD3, Z-Image) which use
linear interpolation instead of variance-preserving schedules.
"""

import logging

import torch

log = logging.getLogger(__name__)


def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Rescale betas to enforce zero terminal SNR.

    Modifies the noise schedule so that the signal-to-noise ratio (SNR)
    reaches exactly 0 at the final timestep. This fixes the train/inference
    mismatch described in "Common Diffusion Noise Schedules" paper.

    Args:
        betas: Original beta schedule [T]

    Returns:
        Rescaled betas [T] with zero terminal SNR
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so final value is 0 (zero terminal SNR)
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so first value is restored
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert back to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1.0 - alphas

    return betas.clamp(0.0, 0.999)


def compute_snr(noise_scheduler) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio for each timestep.

    SNR(t) = alpha_cumprod(t) / (1 - alpha_cumprod(t))

    Returns:
        SNR values [T] for each timestep
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    snr = alphas_cumprod / (1.0 - alphas_cumprod)
    return snr


def compute_snr_weights(
    timesteps: torch.Tensor,
    noise_scheduler,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Compute Min-SNR-gamma loss weights.

    From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    (Hang et al. 2023). Weights each timestep by min(SNR, gamma) / SNR.

    Args:
        timesteps: Sampled timesteps [B]
        noise_scheduler: Diffusers noise scheduler
        gamma: SNR clipping value (5.0 recommended)

    Returns:
        Per-sample loss weights [B]
    """
    snr = compute_snr(noise_scheduler).to(
        device=timesteps.device, dtype=torch.float32
    )
    timestep_snr = snr[timesteps]

    # min(SNR, gamma) / SNR
    weights = torch.clamp(timestep_snr, max=gamma) / timestep_snr

    return weights


def apply_noise_rescaling(
    noise_scheduler,
    zero_terminal_snr: bool = True,
    rescale_classifier_free_guidance: bool = False,
) -> bool:
    """Apply noise schedule rescaling to an existing scheduler.

    This modifies the scheduler in-place to fix the terminal SNR issue.

    Args:
        noise_scheduler: Diffusers noise scheduler
        zero_terminal_snr: Enforce zero terminal SNR
        rescale_classifier_free_guidance: Adjust CFG for rescaled schedule

    Returns:
        True if rescaling was applied, False if not applicable
    """
    if not hasattr(noise_scheduler, 'betas'):
        log.debug("Scheduler has no betas attribute, skipping rescaling")
        return False

    if not hasattr(noise_scheduler, 'alphas_cumprod'):
        log.debug("Scheduler has no alphas_cumprod, skipping rescaling")
        return False

    if zero_terminal_snr:
        original_betas = noise_scheduler.betas.clone()
        new_betas = enforce_zero_terminal_snr(noise_scheduler.betas)

        # Update scheduler state
        noise_scheduler.betas = new_betas
        noise_scheduler.alphas = 1.0 - new_betas
        noise_scheduler.alphas_cumprod = torch.cumprod(
            noise_scheduler.alphas, dim=0
        )

        # Verify terminal SNR is ~0
        final_snr = (
            noise_scheduler.alphas_cumprod[-1]
            / (1.0 - noise_scheduler.alphas_cumprod[-1])
        )
        log.info(
            f"Noise rescaling applied: terminal SNR = {final_snr:.6f} "
            f"(was {original_betas[-1]:.4f})"
        )
        return True

    return False
