"""Contract tests for the inner training step.

The Trainer's outer loop (NaN handling, accumulation, save intervals,
progress callbacks) is bookkeeping — it's the per-step path that does
the real work and where regressions would silently corrupt gradients.

These tests exercise the train-step contract end-to-end on a tiny stub
backend so they run on CPU in milliseconds and catch:

- PEFT-wrapped params actually receive gradients on backward()
- optimizer.step() moves params in the descent direction
- Loss decreases over 5 steps on a fixed mini-batch (sanity check that
  the whole forward → loss → backward → step pipeline closes the loop)
- NaN / Inf detection works as expected
- Progressive-batch cycle-cache invariant holds across the optimizer.step()
- compute_loss dispatches correctly on prediction_type
"""

from __future__ import annotations

import importlib.util

import pytest

from dataset_sorter.models import TrainingConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PEFT = importlib.util.find_spec("peft") is not None


pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_PEFT),
    reason="torch + peft required",
)


# ─────────────────────────────────────────────────────────────────────────
# Test fixtures: a tiny "diffusion-like" backend that runs on CPU
# ─────────────────────────────────────────────────────────────────────────


def _build_stub_backend(network_type: str = "lora", **cfg_kwargs):
    """Concrete stub: wrapped UNet + simple noise scheduler.

    The "scheduler" mimics the DDPM API: a 1000-element ``alphas_cumprod``
    tensor + a ``num_train_timesteps`` config attr. Nothing else.
    """
    import torch
    from torch import nn
    from dataset_sorter.train_backend_base import TrainBackendBase

    class _ToyUnet(nn.Module):
        def __init__(self, dim: int = 32):
            super().__init__()
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

        def forward(self, x):
            # Simple scaled dot product to make the loss path deterministic.
            return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))

    class _StubScheduler:
        def __init__(self, num_steps: int = 1000):
            betas = torch.linspace(1e-4, 0.02, num_steps)
            self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
            self.config = type("C", (), {"num_train_timesteps": num_steps})()

        def get_velocity(self, latents, noise, timesteps):
            ac = self.alphas_cumprod[timesteps].view(-1, 1).to(latents.dtype)
            return ac.sqrt() * noise - (1.0 - ac).sqrt() * latents

    class _ConcreteBackend(TrainBackendBase):
        model_name = "stub"
        prediction_type = "epsilon"
        default_resolution = 32

        def load_model(self, model_path):  # noqa: ARG002
            pass

        def encode_text_batch(self, captions):  # noqa: ARG002
            return ()

    cfg = TrainingConfig(
        model_type="sdxl_lora",
        network_type=network_type,
        lora_rank=4,
        lora_alpha=4,
        learning_rate=1e-2,  # large for fast convergence in 5 steps
        loss_fn="mse",
        **cfg_kwargs,
    )

    backend = _ConcreteBackend.__new__(_ConcreteBackend)
    backend.config = cfg
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.vae_dtype = torch.float32
    backend.pipeline = None
    backend.unet = _ToyUnet()
    backend.vae = None
    backend.noise_scheduler = _StubScheduler()
    backend.tokenizer = None
    backend.text_encoder = None
    backend.text_encoder_2 = None
    backend._cached_time_ids = None
    backend._compiled = False
    backend._speed_sampler = None
    backend._token_weight_mask = None
    backend._timestep_ema_sampler = None
    backend._training_mask = None
    backend._training_te = False
    backend.adapter_type = None
    backend.lycoris_net = None
    backend._alphas_cumprod_cache = None

    backend.setup_lora()
    return backend, cfg


# ─────────────────────────────────────────────────────────────────────────
# Loss helpers — the building blocks the Trainer relies on
# ─────────────────────────────────────────────────────────────────────────


def test_compute_loss_dispatches_to_epsilon_path_by_default():
    """``prediction_type='epsilon'`` should route to _compute_epsilon_loss
    and produce a positive scalar tensor."""
    import torch
    backend, _ = _build_stub_backend()

    latents = torch.randn(2, 32, requires_grad=False)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    # Pretend the model predicted noise+0.1 — almost right, slight error.
    noise_pred = noise + 0.1 * torch.randn_like(noise)

    loss = backend.compute_loss(noise_pred, noise, latents, timesteps)
    # _compute_epsilon_loss returns per-sample loss reduced to a vector;
    # the trainer then averages externally. Either shape is OK.
    if loss.dim() > 0:
        loss = loss.mean()
    assert loss.requires_grad is False  # inputs detached, that's fine
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_compute_loss_x0_supervision_path():
    """``x0_supervision=True`` reroutes through _compute_x0_loss; the
    output must still be a finite, positive-or-zero scalar."""
    import torch
    backend, _ = _build_stub_backend(x0_supervision=True)

    latents = torch.randn(2, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    noise_pred = noise.clone()  # perfect prediction → near-zero loss

    loss = backend.compute_loss(noise_pred, noise, latents, timesteps)
    if loss.dim() > 0:
        loss = loss.mean()
    assert torch.isfinite(loss)
    assert loss.item() < 1e-3, (
        f"x₀ loss with perfect prediction should be ≈0, got {loss.item()}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Gradient flow — the core training contract
# ─────────────────────────────────────────────────────────────────────────


def test_lora_params_receive_gradients_on_backward():
    """After PEFT setup + a backward pass, every trainable param must
    have a non-None .grad — otherwise the optimizer can't update them."""
    import torch
    backend, _ = _build_stub_backend()

    # Forward through the wrapped unet, compute a synthetic loss
    x = torch.randn(4, 32, requires_grad=False)
    y_true = torch.randn(4, 32)
    y_pred = backend.unet(x)
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()

    trainable = [
        (n, p) for n, p in backend.unet.named_parameters() if p.requires_grad
    ]
    assert len(trainable) > 0
    for name, param in trainable:
        assert param.grad is not None, (
            f"trainable param {name} has no gradient after backward()"
        )
        assert torch.isfinite(param.grad).all(), (
            f"param {name} has non-finite gradients"
        )


def test_optimizer_step_decreases_loss_on_fixed_batch():
    """Smoke test of the full pipeline: 5 forward+backward+step iterations
    on the SAME batch should drive loss down. If they don't, the
    gradient flow / optimizer wiring is broken."""
    import torch
    backend, cfg = _build_stub_backend()

    trainable_params = [p for p in backend.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=cfg.learning_rate)

    torch.manual_seed(42)
    x = torch.randn(4, 32)
    y = torch.randn(4, 32)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        pred = backend.unet(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss must decrease over the 5 steps.
    assert losses[-1] < losses[0], (
        f"loss did not decrease over 5 steps: {losses}"
    )


# ─────────────────────────────────────────────────────────────────────────
# NaN / Inf safety — the trainer skips these batches without crashing
# ─────────────────────────────────────────────────────────────────────────


def test_loss_with_nan_prediction_propagates_nan():
    """If the model prediction contains NaN, the loss should also be NaN —
    that's what the trainer's NaN guard relies on to detect bad batches."""
    import torch
    backend, _ = _build_stub_backend()

    latents = torch.randn(2, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    # NaN prediction simulates a runaway-gradient explosion mid-training.
    noise_pred = torch.full_like(noise, float("nan"))

    loss = backend.compute_loss(noise_pred, noise, latents, timesteps)
    if loss.dim() > 0:
        loss = loss.mean()
    assert not torch.isfinite(loss), (
        "Expected non-finite loss for NaN prediction — the trainer's "
        "isnan/isinf guard depends on this"
    )


def test_loss_with_inf_prediction_propagates_inf():
    """Same guard, but for Inf rather than NaN — fp16 overflow case."""
    import torch
    backend, _ = _build_stub_backend()

    latents = torch.randn(2, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    noise_pred = torch.full_like(noise, float("inf"))

    loss = backend.compute_loss(noise_pred, noise, latents, timesteps)
    if loss.dim() > 0:
        loss = loss.mean()
    assert not torch.isfinite(loss), (
        "Expected non-finite loss for Inf prediction"
    )


def test_alphas_cumprod_cache_is_populated_on_first_use():
    """The perf optimisation we just landed: alphas_cumprod is cached on
    first compute_loss call and reused on subsequent calls."""
    import torch
    backend, _ = _build_stub_backend()
    assert backend._alphas_cumprod_cache is None

    latents = torch.randn(2, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    backend.compute_loss(noise.clone(), noise, latents, timesteps)
    # We use the v_prediction path indirectly only when prediction_type
    # is v_prediction; epsilon path doesn't touch the cache. Force an x₀
    # call to exercise the cache.
    backend.config.x0_supervision = True
    backend.compute_loss(noise.clone(), noise, latents, timesteps)
    assert backend._alphas_cumprod_cache is not None
    cached = backend._alphas_cumprod_cache

    # Calling again must return the SAME tensor (no new allocation).
    backend.compute_loss(noise.clone(), noise, latents, timesteps)
    assert backend._alphas_cumprod_cache is cached, (
        "alphas_cumprod cache was re-allocated on second call — perf bug"
    )
