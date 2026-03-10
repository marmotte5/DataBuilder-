"""LR schedule preview — compute LR values for visualization.

Generates LR curves without requiring actual model parameters,
using a dummy optimizer + the real scheduler factory.
"""

import math


def compute_lr_schedule(
    scheduler_type: str,
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    num_cycles: int = 1,
) -> list[tuple[int, float]]:
    """Compute LR values at each step for a given scheduler.

    Returns list of (step, lr) tuples for plotting/display.
    Works without torch/diffusers by reimplementing schedule math.
    """
    if total_steps <= 0:
        return [(0, learning_rate)]

    points = []

    for step in range(total_steps + 1):
        lr = _compute_lr_at_step(
            scheduler_type, learning_rate, step, total_steps,
            warmup_steps, num_cycles,
        )
        points.append((step, lr))

    return points


def _compute_lr_at_step(
    scheduler_type: str,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    num_cycles: int,
) -> float:
    """Compute LR at a specific step (pure math, no torch needed)."""
    # Warmup phase
    if step < warmup_steps and warmup_steps > 0:
        warmup_factor = step / warmup_steps
        if scheduler_type == "constant":
            return base_lr * warmup_factor
        if scheduler_type == "constant_with_warmup":
            return base_lr * warmup_factor

    # Post-warmup
    if scheduler_type == "constant" or scheduler_type == "constant_with_warmup":
        return base_lr

    # Progress after warmup
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)

    if scheduler_type == "linear":
        return base_lr * (1.0 - progress)

    if scheduler_type == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    if scheduler_type == "cosine_with_restarts":
        return base_lr * 0.5 * (
            1.0 + math.cos(math.pi * ((progress * num_cycles) % 1.0))
        )

    if scheduler_type == "polynomial":
        power = 2.0
        return base_lr * (1.0 - progress) ** power

    if scheduler_type == "rex":
        # REX: reciprocal schedule lr * 1/(1 + t/T)
        if total_steps > warmup_steps:
            t = step - warmup_steps
            T = total_steps - warmup_steps
            return base_lr / (1.0 + t / T)
        return base_lr

    # Fallback
    return base_lr


def format_lr_ascii_graph(
    points: list[tuple[int, float]],
    width: int = 60,
    height: int = 15,
) -> str:
    """Render an ASCII art graph of the LR schedule.

    Returns a multi-line string suitable for display in a text widget.
    """
    if not points:
        return "No data"

    steps = [p[0] for p in points]
    lrs = [p[1] for p in points]

    max_lr = max(lrs) if lrs else 1.0
    min_lr = min(lrs) if lrs else 0.0
    lr_range = max_lr - min_lr if max_lr > min_lr else 1e-10
    max_step = max(steps) if steps else 1

    # Sample points to fit width
    if len(points) > width:
        sample_indices = [int(i * (len(points) - 1) / (width - 1)) for i in range(width)]
        sampled = [points[i] for i in sample_indices]
    else:
        sampled = points

    # Build grid
    grid = [[" " for _ in range(len(sampled))] for _ in range(height)]

    for col, (step, lr) in enumerate(sampled):
        row = int((lr - min_lr) / lr_range * (height - 1))
        row = height - 1 - row  # Flip: top = high
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    # Format output
    lines = []
    lines.append(f"LR Schedule Preview ({len(points)} steps)")
    lines.append(f"Max LR: {max_lr:.2e}  Min LR: {min_lr:.2e}")
    lines.append("")

    for r in range(height):
        if r == 0:
            label = f"{max_lr:.1e}"
        elif r == height - 1:
            label = f"{min_lr:.1e}"
        else:
            label = "        "
        row_str = "".join(grid[r])
        lines.append(f"{label:>9} |{row_str}")

    lines.append(f"          +{'─' * len(sampled)}")
    lines.append(f"          0{' ' * (len(sampled) - len(str(max_step)) - 1)}{max_step}")

    return "\n".join(lines)
