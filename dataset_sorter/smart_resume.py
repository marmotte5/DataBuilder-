"""Smart Resume — loss curve analysis and hyperparameter adjustment.

When resuming interrupted training, analyses the previous run's loss curve
and adjusts hyperparameters (LR, batch size, warmup, etc.) to improve
the next phase of training.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class LossAnalysis:
    """Result of analysing a training loss history."""
    total_steps: int = 0
    final_loss: float = 0.0
    min_loss: float = 0.0
    max_loss: float = 0.0
    mean_loss: float = 0.0

    # Trend detection
    trend: str = "unknown"          # improving, plateau, diverging, oscillating
    improvement_rate: float = 0.0   # average loss decrease per 100 steps
    plateau_start_step: int = 0     # step where plateau was detected (0 = none)
    divergence_start_step: int = 0  # step where divergence started (0 = none)
    oscillation_amplitude: float = 0.0
    _first_step: int = 0            # absolute first step (for plateau duration calc)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    adjustments: dict = field(default_factory=dict)  # param_name -> new_value
    confidence: float = 0.0         # 0-1, how confident we are in recommendations


def save_loss_history(
    output_dir: Path,
    loss_history: list[tuple[int, float]],
    lr_history: list[tuple[int, float]] | None = None,
    config_snapshot: dict | None = None,
):
    """Save loss history and metadata to the training output directory."""
    history_path = output_dir / "loss_history.json"

    # Load existing history (accumulate across resumes)
    existing = []
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            existing = data.get("runs", [])
        except (json.JSONDecodeError, OSError):
            pass

    run = {
        "loss_points": [{"step": s, "loss": l} for s, l in loss_history],
    }
    if lr_history:
        run["lr_points"] = [{"step": s, "lr": lr} for s, lr in lr_history]
    if config_snapshot:
        run["config"] = config_snapshot

    existing.append(run)

    data = {"runs": existing}
    try:
        history_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
    except OSError as e:
        log.warning(f"Could not save loss history: {e}")


def load_loss_history(output_dir: Path) -> list[tuple[int, float]]:
    """Load all loss points from all runs, concatenated."""
    history_path = output_dir / "loss_history.json"
    if not history_path.exists():
        return []

    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    all_points = []
    for run in data.get("runs", []):
        for pt in run.get("loss_points", []):
            try:
                all_points.append((pt["step"], pt["loss"]))
            except (KeyError, TypeError):
                continue  # Skip malformed entries

    return sorted(all_points, key=lambda x: x[0])


def analyze_loss_curve(
    loss_history: list[tuple[int, float]],
    window_size: int = 50,
) -> LossAnalysis:
    """Analyse a loss curve and detect trends.

    Args:
        loss_history: list of (step, loss) tuples
        window_size: number of steps to use for moving-average smoothing

    Returns:
        LossAnalysis with detected trend and recommendations
    """
    result = LossAnalysis()
    if len(loss_history) < 10:
        result.trend = "insufficient_data"
        result.recommendations.append("Not enough training history for analysis (need 10+ steps).")
        return result

    steps = [s for s, _ in loss_history]
    losses = [l for _, l in loss_history]

    result.total_steps = steps[-1] - steps[0]
    result._first_step = steps[0]
    result.final_loss = losses[-1]
    result.min_loss = min(losses)
    result.max_loss = max(losses)
    result.mean_loss = sum(losses) / len(losses)

    # Compute moving average for smoothing
    def moving_avg(vals, w):
        if len(vals) <= w:
            return vals
        out = []
        for i in range(len(vals)):
            start = max(0, i - w // 2)
            end = min(len(vals), i + w // 2 + 1)
            out.append(sum(vals[start:end]) / (end - start))
        return out

    smoothed = moving_avg(losses, window_size)

    # Split into segments for trend analysis
    n = len(smoothed)
    seg_size = max(1, n // 4)

    seg1 = smoothed[:seg_size]              # early
    seg2 = smoothed[seg_size:2*seg_size]    # mid-early
    seg3 = smoothed[2*seg_size:3*seg_size]  # mid-late
    seg4 = smoothed[3*seg_size:]            # late

    avg1 = sum(seg1) / max(1, len(seg1))
    avg2 = sum(seg2) / max(1, len(seg2))
    avg3 = sum(seg3) / max(1, len(seg3))
    avg4 = sum(seg4) / max(1, len(seg4))

    # Compute improvement rate (loss decrease per 100 steps)
    if result.total_steps > 0:
        result.improvement_rate = (losses[0] - losses[-1]) / result.total_steps * 100

    # Detect oscillation amplitude
    if n >= 20:
        recent = losses[-min(n, 100):]
        recent_mean = sum(recent) / len(recent)
        result.oscillation_amplitude = (
            sum(abs(l - recent_mean) for l in recent) / len(recent)
        )

    # ── Trend detection ──

    late_improving = avg4 < avg3 * 0.98  # still decreasing in last segment
    # Use abs() for threshold to handle near-zero or negative losses correctly
    plateau = (abs(avg4 - avg3) < abs(avg3) * 0.02 + 1e-8
               and abs(avg3 - avg2) < abs(avg2) * 0.02 + 1e-8)
    diverging = avg4 > avg3 * 1.05  # loss increasing in last segment

    # Check for oscillation (high variance relative to mean in recent loss)
    oscillating = False
    if n >= 20:
        recent = losses[-min(n, 100):]
        recent_mean = sum(recent) / len(recent)
        variance = sum((l - recent_mean) ** 2 for l in recent) / len(recent)
        std = math.sqrt(variance)
        # Use abs(recent_mean) to handle near-zero/negative losses
        oscillating = std > max(abs(recent_mean) * 0.15, 1e-8)

    # Classify
    if diverging:
        result.trend = "diverging"
        result.divergence_start_step = steps[3 * seg_size] if 3 * seg_size < len(steps) else steps[-1]
        result.confidence = 0.85
    elif oscillating:
        result.trend = "oscillating"
        result.confidence = 0.75
    elif plateau:
        result.trend = "plateau"
        # Find where plateau started
        for i in range(n - 1, 0, -1):
            if abs(smoothed[i] - smoothed[-1]) > abs(smoothed[-1]) * 0.03 + 1e-8:
                result.plateau_start_step = steps[min(i, len(steps) - 1)]
                break
        result.confidence = 0.8
    elif late_improving:
        result.trend = "improving"
        result.confidence = 0.9
    else:
        result.trend = "stable"
        result.confidence = 0.6

    return result


def compute_adjustments(
    analysis: LossAnalysis,
    current_lr: float,
    current_batch_size: int,
    current_epochs: int,
    current_warmup: int,
    current_optimizer: str,
    total_steps_remaining: int,
) -> LossAnalysis:
    """Given a loss analysis, compute recommended hyperparameter adjustments.

    Mutates the analysis.recommendations and analysis.adjustments in-place.
    """
    recs = analysis.recommendations
    adj = analysis.adjustments

    if analysis.trend == "diverging":
        # Loss is increasing — LR is too high or training is unstable
        new_lr = current_lr * 0.3
        adj["learning_rate"] = new_lr
        adj["text_encoder_lr"] = new_lr * 0.5
        recs.append(f"Loss is diverging. Reducing LR from {current_lr:.2e} to {new_lr:.2e} (0.3x).")

        if current_warmup < 200:
            adj["warmup_steps"] = min(200, total_steps_remaining // 5)
            recs.append(f"Adding warmup steps ({adj['warmup_steps']}) to stabilise after resume.")

        adj["max_grad_norm"] = 0.5
        recs.append("Tightening gradient clipping to 0.5 to prevent explosions.")

    elif analysis.trend == "oscillating":
        # High variance — LR too aggressive or batch too small
        new_lr = current_lr * 0.5
        adj["learning_rate"] = new_lr
        adj["text_encoder_lr"] = new_lr * 0.5
        recs.append(f"Loss is oscillating. Reducing LR from {current_lr:.2e} to {new_lr:.2e} (0.5x).")

        if current_batch_size < 4:
            # Suggest gradient accumulation increase instead of actual batch size
            adj["gradient_accumulation"] = min(8, current_batch_size * 2)
            recs.append(
                f"Increasing effective batch size via gradient accumulation "
                f"to {adj['gradient_accumulation']} to smooth gradients."
            )

    elif analysis.trend == "plateau":
        # Loss has flattened — several strategies
        # plateau_start_step is absolute; total_steps is the step *range*
        # (steps[-1] - steps[0]).  Compute duration correctly.
        steps_in_plateau = max(0, analysis.total_steps - (
            analysis.plateau_start_step - analysis._first_step
        ))

        if steps_in_plateau > analysis.total_steps * 0.3:
            # Long plateau — significant LR reduction to find lower minimum
            new_lr = current_lr * 0.2
            adj["learning_rate"] = new_lr
            adj["text_encoder_lr"] = new_lr * 0.5
            recs.append(
                f"Extended plateau detected ({steps_in_plateau} steps). "
                f"Reducing LR from {current_lr:.2e} to {new_lr:.2e} (0.2x) "
                f"to probe for a lower minimum."
            )
        else:
            # Short plateau — moderate reduction
            new_lr = current_lr * 0.5
            adj["learning_rate"] = new_lr
            adj["text_encoder_lr"] = new_lr * 0.5
            recs.append(
                f"Plateau detected at step {analysis.plateau_start_step}. "
                f"Reducing LR to {new_lr:.2e} (0.5x)."
            )

        # Switch scheduler to cosine for smoother decay
        adj["lr_scheduler"] = "cosine"
        recs.append("Switching to cosine LR scheduler for smoother decay.")

        # Add warmup to smoothly transition
        adj["warmup_steps"] = min(50, total_steps_remaining // 10)
        recs.append(f"Adding {adj['warmup_steps']} warmup steps for smooth resume transition.")

    elif analysis.trend == "improving":
        # Still improving — gentle adjustments
        recs.append(
            f"Training is still improving (rate: {analysis.improvement_rate:.4f} loss/100 steps). "
            f"Keeping current hyperparameters."
        )

        if 0 < analysis.improvement_rate < 0.001 and current_lr > 1e-6:
            # Improving but slowly — slight LR boost.
            # Guard: only boost if improvement_rate is positive (truly improving).
            # A negative rate means loss is increasing, which should be caught
            # by the diverging/oscillating checks above.
            new_lr = current_lr * 1.2
            adj["learning_rate"] = new_lr
            recs.append(
                f"Improvement rate is slow. Slight LR increase to {new_lr:.2e} (1.2x) "
                f"to accelerate convergence."
            )

    elif analysis.trend == "stable":
        recs.append("Loss is stable. No adjustments recommended.")

    elif analysis.trend == "insufficient_data":
        recs.append("Not enough data for analysis. Continuing with current settings.")

    return analysis


def format_analysis_report(analysis: LossAnalysis) -> str:
    """Format a human-readable analysis report."""
    lines = [
        "=" * 50,
        "  SMART RESUME — Loss Curve Analysis",
        "=" * 50,
        "",
        f"  Steps analyzed:  {analysis.total_steps}",
        f"  Final loss:      {analysis.final_loss:.6f}",
        f"  Min loss:        {analysis.min_loss:.6f}",
        f"  Mean loss:       {analysis.mean_loss:.6f}",
        f"  Trend:           {analysis.trend.upper()}",
        f"  Improvement:     {analysis.improvement_rate:.4f} loss/100 steps",
        f"  Confidence:      {analysis.confidence:.0%}",
        "",
    ]

    if analysis.plateau_start_step > 0:
        lines.append(f"  Plateau started:  step {analysis.plateau_start_step}")
    if analysis.divergence_start_step > 0:
        lines.append(f"  Divergence at:    step {analysis.divergence_start_step}")
    if analysis.oscillation_amplitude > 0:
        lines.append(f"  Oscillation amp:  {analysis.oscillation_amplitude:.6f}")

    lines.append("")
    lines.append("  RECOMMENDATIONS:")
    lines.append("  " + "-" * 46)
    for rec in analysis.recommendations:
        lines.append(f"  - {rec}")

    if analysis.adjustments:
        lines.append("")
        lines.append("  ADJUSTMENTS TO APPLY:")
        lines.append("  " + "-" * 46)
        for param, value in analysis.adjustments.items():
            if isinstance(value, float) and value < 0.01:
                lines.append(f"    {param}: {value:.2e}")
            else:
                lines.append(f"    {param}: {value}")

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)


def apply_adjustments_to_config(config, adjustments: dict):
    """Apply recommended adjustments to a TrainingConfig."""
    for key, value in adjustments.items():
        if hasattr(config, key):
            setattr(config, key, value)
            log.info(f"Smart Resume: {key} = {value}")
    return config
