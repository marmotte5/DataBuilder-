"""VRAM estimation for training configurations.

Estimates GPU memory usage based on model architecture, resolution,
batch size, precision, and optimization settings. Helps users choose
appropriate settings before starting a training run.
"""

import logging

log = logging.getLogger(__name__)

# Base model memory (GB) at bf16 for UNet/transformer only
_MODEL_VRAM_BF16 = {
    "sd15":     1.7,
    "sd2":      1.7,
    "sdxl":     5.1,
    "pony":     5.1,
    "flux":     12.0,
    "flux2":    12.0,
    "sd3":      4.5,
    "sd35":     4.5,
    "zimage":   5.1,
    "pixart":   2.5,
    "cascade":  3.6,
    "hunyuan":  3.0,
    "kolors":   5.1,
    "auraflow": 6.0,
    "sana":     2.8,
    "hidream":  8.0,
    "chroma":   12.0,
}

# VAE memory (GB) at bf16
_VAE_VRAM = {
    "sd15": 0.3, "sd2": 0.3, "sdxl": 0.4, "pony": 0.4,
    "flux": 0.4, "flux2": 0.4, "sd3": 0.4, "sd35": 0.4,
    "zimage": 0.4, "pixart": 0.4, "cascade": 0.4,
    "hunyuan": 0.4, "kolors": 0.4, "auraflow": 0.4,
    "sana": 0.4, "hidream": 0.4, "chroma": 0.4,
}

# Text encoder memory (GB) at bf16
_TE_VRAM = {
    "sd15": 0.5, "sd2": 0.5, "sdxl": 1.6, "pony": 1.6,
    "flux": 9.5, "flux2": 9.5, "sd3": 5.5, "sd35": 5.5,
    "zimage": 1.6, "pixart": 4.5, "cascade": 0.5,
    "hunyuan": 3.0, "kolors": 1.6, "auraflow": 4.5,
    "sana": 4.5, "hidream": 5.5, "chroma": 9.5,
}


def get_base_model_key(model_type: str) -> str:
    """Strip _lora/_full suffix to get base model key."""
    for suffix in ("_lora", "_full"):
        if model_type.endswith(suffix):
            return model_type[: -len(suffix)]
    return model_type


def estimate_vram(config) -> dict:
    """Estimate VRAM usage for a training configuration.

    Returns a dict with:
        - total_gb: Estimated total VRAM (GB)
        - breakdown: Dict of component -> GB
        - fits_gpu: Whether it fits in config.vram_gb
        - warnings: List of warning strings
        - suggestions: List of suggestions to reduce VRAM
    """
    base = get_base_model_key(config.model_type)
    is_lora = config.model_type.endswith("_lora")

    breakdown = {}
    warnings = []
    suggestions = []

    # 1. Model weights
    base_model_gb = _MODEL_VRAM_BF16.get(base, 5.0)  # Unmodified for LoRA calc
    model_gb = base_model_gb
    if config.fp8_base_model:
        model_gb *= 0.5
    if config.mixed_precision == "fp32":
        model_gb *= 2.0
    breakdown["Model weights"] = round(model_gb, 1)

    # 2. VAE (offloaded to CPU if latents cached)
    vae_gb = _VAE_VRAM.get(base, 0.4)
    if config.cache_latents:
        vae_gb = 0.0  # Offloaded after caching
    breakdown["VAE"] = round(vae_gb, 1)

    # 3. Text encoder (offloaded if TE outputs cached)
    te_gb = _TE_VRAM.get(base, 1.0)
    if config.cache_text_encoder:
        te_gb = 0.0  # Offloaded after caching
    elif not config.train_text_encoder:
        te_gb *= 0.5  # Inference only, no gradients
    # Apply quantization savings
    te_quant = getattr(config, 'quantize_text_encoder', 'none')
    if te_gb > 0 and te_quant == "int8":
        te_gb *= 0.5
    elif te_gb > 0 and te_quant == "int4":
        te_gb *= 0.25
    breakdown["Text encoder"] = round(te_gb, 1)

    # 4. LoRA adapter weights
    if is_lora:
        # LoRA adds ~rank/dim * model_params — use base (bf16) size, not
        # precision-adjusted model_gb, since adapter size depends on param count
        lora_gb = base_model_gb * (config.lora_rank / 1024) * 0.5
        if config.use_dora:
            lora_gb *= 1.3  # DoRA has extra magnitude vector
        breakdown["LoRA adapter"] = round(lora_gb, 2)
    else:
        breakdown["LoRA adapter"] = 0.0

    # 5. Optimizer states
    opt = config.optimizer
    trainable_gb = lora_gb if is_lora else model_gb
    if opt in ("AdamW", "AdamW8bit"):
        # AdamW: 2 states (mean + variance) per parameter
        opt_mult = 1.0 if "8bit" in opt else 2.0
        opt_gb = trainable_gb * opt_mult
    elif opt == "Adafactor":
        # Adafactor: ~1 state (row/col factored)
        opt_gb = trainable_gb * 0.5
    elif opt in ("Prodigy", "DAdaptAdam"):
        opt_gb = trainable_gb * 2.5
    elif opt == "SGD":
        opt_gb = trainable_gb * 0.1
    elif opt in ("Lion", "SOAP", "Muon"):
        opt_gb = trainable_gb * 1.0
    elif opt in ("CAME", "AdamWScheduleFree"):
        opt_gb = trainable_gb * 1.5
    else:
        opt_gb = trainable_gb * 2.0

    if config.fused_backward_pass and opt == "Adafactor":
        opt_gb *= 0.3  # Fused backward dramatically reduces peak
    breakdown["Optimizer states"] = round(opt_gb, 1)

    # 6. Gradient memory
    grad_gb = trainable_gb * 1.0
    if config.gradient_checkpointing:
        grad_gb *= 0.3  # Recompute activations, save memory
    breakdown["Gradients"] = round(grad_gb, 1)

    # 7. Activation memory (scales with resolution and batch size)
    res_factor = (config.resolution / 1024) ** 2
    act_gb = 1.5 * config.batch_size * res_factor
    if config.gradient_checkpointing:
        act_gb *= 0.4
    breakdown["Activations"] = round(act_gb, 1)

    # 8. EMA weights
    ema_gb = 0.0
    if config.use_ema:
        ema_gb = trainable_gb
        if config.ema_cpu_offload:
            ema_gb = 0.0  # Offloaded to CPU RAM
    breakdown["EMA weights"] = round(ema_gb, 1)

    # 9. CUDA overhead / fragmentation
    overhead = 0.8
    breakdown["CUDA overhead"] = overhead

    total_gb = sum(breakdown.values())
    total_gb = round(total_gb, 1)

    # Check fit
    fits_gpu = total_gb <= config.vram_gb

    if not fits_gpu:
        excess = total_gb - config.vram_gb
        warnings.append(
            f"Estimated {total_gb} GB exceeds your {config.vram_gb} GB GPU by ~{excess:.1f} GB"
        )

        # Generate suggestions
        if not config.cache_latents:
            suggestions.append("Enable 'Cache VAE Latents' to offload VAE from GPU")
        if not config.cache_text_encoder:
            suggestions.append("Enable 'Cache Text Encoder' to offload TE from GPU")
        if not config.gradient_checkpointing:
            suggestions.append("Enable 'Gradient Checkpointing' (saves ~40% activation memory)")
        if config.batch_size > 1:
            suggestions.append(f"Reduce batch size from {config.batch_size} to 1")
        if not config.fp8_base_model and not is_lora:
            suggestions.append("Enable 'fp8 Base Model' to halve model weight memory")
        te_quant_s = getattr(config, 'quantize_text_encoder', 'none')
        if te_quant_s == "none" and not config.cache_text_encoder:
            suggestions.append("Set 'TE Quantization' to INT8 or INT4 to reduce text encoder VRAM")
        if config.use_ema and not config.ema_cpu_offload:
            suggestions.append("Enable 'EMA CPU Offload' to move EMA weights to RAM")
        if not is_lora:
            suggestions.append("Use LoRA instead of full finetune for much lower VRAM")
        if config.optimizer in ("AdamW", "Prodigy", "DAdaptAdam"):
            suggestions.append("Switch to Adafactor optimizer (lower memory)")
        if config.resolution > 512 and base in ("sd15", "sd2"):
            suggestions.append(f"Reduce resolution from {config.resolution} to 512")
        if config.optimizer == "Adafactor" and not config.fused_backward_pass:
            suggestions.append("Enable 'Fused Backward Pass' with Adafactor (saves ~14 GB)")

    return {
        "total_gb": total_gb,
        "breakdown": breakdown,
        "fits_gpu": fits_gpu,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def detect_gpu_vram() -> int:
    """Auto-detect GPU VRAM and return closest VRAM tier.

    Returns the detected VRAM in GB, or 0 if no GPU found.
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return round(vram_bytes / 1024**3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon shares system RAM; report unified memory
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    total_bytes = int(result.stdout.strip())
                    # Report ~75% of unified memory as usable for GPU
                    return round(total_bytes * 0.75 / 1024**3)
            except Exception:
                pass
            return 16  # Conservative default for Apple Silicon
    except (ImportError, OSError):
        pass
    return 0


def auto_detect_model_vram(model) -> float:
    """Auto-detect VRAM usage of a loaded model by counting parameters.

    Works with any model (custom, merged, pruned) — not just known architectures.
    Returns estimated VRAM in GB at bf16 precision.
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        # bf16 = 2 bytes per param
        vram_gb = (total_params * 2) / (1024 ** 3)
        return round(vram_gb, 2)
    except Exception as e:
        log.warning(f"Could not auto-detect model VRAM: {e}")
        return 0.0


class AdaptiveVRAMMonitor:
    """Monitors actual VRAM usage during training and adjusts settings.

    Samples VRAM at configurable intervals and can dynamically adjust
    batch size / gradient accumulation to prevent OOM crashes.
    """

    def __init__(self, total_vram_gb: float, safety_margin_gb: float = 1.5):
        self.total_vram_gb = total_vram_gb
        self.safety_margin_gb = safety_margin_gb
        self._samples: list[float] = []
        self._peak_gb: float = 0.0
        self._oom_count: int = 0

    def sample(self) -> float:
        """Sample current VRAM usage. Returns usage in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                self._samples.append(reserved)
                self._peak_gb = max(self._peak_gb, reserved)
                return reserved
        except (ImportError, RuntimeError):
            pass
        return 0.0

    def record_oom(self):
        """Record an OOM event for adaptive adjustments."""
        self._oom_count += 1

    @property
    def peak_gb(self) -> float:
        return self._peak_gb

    @property
    def avg_gb(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def headroom_gb(self) -> float:
        return self.total_vram_gb - self._peak_gb

    def should_reduce_batch(self) -> bool:
        """Check if batch size should be reduced based on VRAM pressure."""
        if self._oom_count > 0:
            return True
        return self.headroom_gb < self.safety_margin_gb

    def should_increase_batch(self) -> bool:
        """Check if batch size can be safely increased."""
        return self.headroom_gb > self.safety_margin_gb * 3 and self._oom_count == 0

    def suggest_adjustments(self, config) -> list[str]:
        """Return a list of suggested adjustments based on observed VRAM usage."""
        suggestions = []
        if self.should_reduce_batch() and config.batch_size > 1:
            suggestions.append(
                f"Reduce batch size from {config.batch_size} to {max(1, config.batch_size - 1)} "
                f"(peak VRAM: {self._peak_gb:.1f}/{self.total_vram_gb:.1f} GB)"
            )
        if self.should_reduce_batch() and config.batch_size == 1 and not config.gradient_checkpointing:
            suggestions.append("Enable gradient checkpointing to reduce VRAM usage")
        if self.should_increase_batch():
            suggestions.append(
                f"Batch size can be increased from {config.batch_size} to {config.batch_size + 1} "
                f"({self.headroom_gb:.1f} GB headroom available)"
            )
        return suggestions

    def get_report(self) -> str:
        """Get a summary report of VRAM monitoring."""
        if not self._samples:
            return "No VRAM samples collected."
        return (
            f"VRAM Monitor: peak={self._peak_gb:.1f} GB, avg={self.avg_gb:.1f} GB, "
            f"headroom={self.headroom_gb:.1f} GB, OOMs={self._oom_count}, "
            f"samples={len(self._samples)}"
        )


def format_vram_estimate(estimate: dict) -> str:
    """Format VRAM estimate as a human-readable string."""
    lines = []
    lines.append(f"Estimated VRAM: {estimate['total_gb']:.1f} GB")
    lines.append("")
    for component, gb in estimate["breakdown"].items():
        if gb > 0:
            lines.append(f"  {component}: {gb:.1f} GB")

    if estimate["warnings"]:
        lines.append("")
        for w in estimate["warnings"]:
            lines.append(f"WARNING: {w}")

    if estimate["suggestions"]:
        lines.append("")
        lines.append("Suggestions to reduce VRAM:")
        for s in estimate["suggestions"]:
            lines.append(f"  - {s}")

    return "\n".join(lines)
