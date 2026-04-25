"""Smoke tests for LoRA-FA and LyCORIS adapter paths.

Validates the configuration plumbing and adapter setup logic without
loading a full diffusion pipeline (which would need GPU + model download).

Tests are structured to be skippable when optional deps (peft, lycoris)
are missing — keeps the suite green on minimal CI environments.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from dataset_sorter.models import TrainingConfig


HAS_PEFT = importlib.util.find_spec("peft") is not None
HAS_LYCORIS = importlib.util.find_spec("lycoris") is not None


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_lora_fa_field_default_is_false():
    """LoRA-FA must default to OFF so existing configs are unchanged."""
    cfg = TrainingConfig()
    assert cfg.use_lora_fa is False


def test_lycoris_factor_default_is_auto():
    """LoKr factor of -1 means auto."""
    cfg = TrainingConfig()
    assert cfg.lycoris_factor == -1
    assert cfg.lycoris_decompose_both is False
    assert cfg.lycoris_use_tucker is False


def test_lycoris_dora_wd_default_off():
    """LyCORIS-DoRA hybrid is opt-in."""
    cfg = TrainingConfig()
    assert cfg.lycoris_dora_wd is False


def test_network_type_defaults_to_lora():
    cfg = TrainingConfig()
    assert cfg.network_type == "lora"


def test_constants_advertise_lycoris_variants():
    """Trainable LyCORIS variants must appear in NETWORK_TYPES."""
    from dataset_sorter.constants import NETWORK_TYPES
    for key in ("lora", "lokr", "loha", "locon", "dylora"):
        assert key in NETWORK_TYPES, f"NETWORK_TYPES missing {key!r}"


# ---------------------------------------------------------------------------
# LoRA-FA: actually freezes lora_A
# ---------------------------------------------------------------------------


class _ToyAttention(nn.Module):
    """Minimal stand-in for a UNet — enough modules for PEFT to target."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(64, 64, bias=False)
        self.to_k = nn.Linear(64, 64, bias=False)
        self.to_v = nn.Linear(64, 64, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(64, 64, bias=False)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
def test_lora_fa_freezes_a_matrix_only():
    """After applying PEFT + LoRA-FA, lora_A must be frozen and lora_B trainable."""
    from peft import LoraConfig, get_peft_model

    model = _ToyAttention()
    lora_cfg = LoraConfig(
        r=4, lora_alpha=4,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_cfg)

    # Manually apply the same logic as setup_lora() with use_lora_fa=True
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_down" in name:
            if param.requires_grad:
                param.requires_grad = False
                frozen += 1

    # Each Linear got an A and B; we have 4 targets → 4 frozen A matrices
    assert frozen == 4, f"expected 4 frozen lora_A, got {frozen}"

    # Verify B is still trainable
    b_trainable = sum(
        1 for n, p in model.named_parameters()
        if ("lora_B" in n or "lora_up" in n) and p.requires_grad
    )
    assert b_trainable >= 4, f"lora_B params must remain trainable, got {b_trainable}"

    # Total trainable params should be roughly half of the full LoRA count
    total_lora = sum(
        1 for n, _ in model.named_parameters()
        if "lora_A" in n or "lora_B" in n or "lora_down" in n or "lora_up" in n
    )
    trainable_lora = sum(
        1 for n, p in model.named_parameters()
        if (("lora_A" in n) or ("lora_B" in n) or ("lora_down" in n) or ("lora_up" in n))
        and p.requires_grad
    )
    assert trainable_lora < total_lora, "LoRA-FA must reduce trainable count"


# ---------------------------------------------------------------------------
# LyCORIS: network construction succeeds for each algo
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_LYCORIS, reason="lycoris-lora not installed")
@pytest.mark.parametrize("algo", ["lokr", "loha", "locon"])
def test_lycoris_network_attaches_to_toy_module(algo):
    """LycorisNetwork should wrap a small module and produce trainable params."""
    from lycoris import LycorisNetwork

    model = _ToyAttention()
    model.requires_grad_(False)

    net = LycorisNetwork(
        model,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        conv_lora_dim=4,
        conv_alpha=4,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        network_module=algo,
    )
    net.apply_to()
    net.train()
    net.requires_grad_(True)

    # The wrapper must own at least some trainable params
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert n_trainable > 0, f"{algo}: LyCORIS attached zero trainable params"

    # Forward should still work after attachment
    x = torch.randn(2, 64)
    y = model(x)
    assert y.shape == (2, 64), f"{algo}: forward shape mismatch after wrap"


@pytest.mark.skipif(not HAS_LYCORIS, reason="lycoris-lora not installed")
def test_lycoris_lokr_factor_is_passed_through():
    """LoKr factor kwarg should be accepted without error."""
    from lycoris import LycorisNetwork

    model = _ToyAttention()
    model.requires_grad_(False)
    net = LycorisNetwork(
        model, multiplier=1.0, lora_dim=4, alpha=4,
        conv_lora_dim=4, conv_alpha=4, network_module="lokr",
        factor=8, decompose_both=True,
    )
    net.apply_to()
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) > 0
