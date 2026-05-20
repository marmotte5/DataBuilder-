"""Regression tests for Flux / Flux 2 / Chroma latent packing.

These backends operate on packed 3D latent sequences (not 4D BCHW) and
require img_ids / txt_ids for RoPE. Without packing the transformer's
linear x_embedder crashes; without IDs the forward fails on None.ndim.

These tests pin the packing math to the diffusers reference so a future
refactor can't silently revert to the old crashing behavior.
"""

import pytest
import torch

from dataset_sorter.train_backend_chroma import ChromaBackend
from dataset_sorter.train_backend_flux import FluxBackend
from dataset_sorter.train_backend_flux2 import Flux2Backend


@pytest.mark.parametrize("backend_cls", [FluxBackend, Flux2Backend, ChromaBackend])
@pytest.mark.parametrize("shape", [
    (2, 16, 128, 128),  # 1024x1024
    (1, 16, 96, 128),   # 768x1024 (aspect ratio bucketing)
    (3, 16, 64, 64),    # 512x512
])
def test_pack_latents_shape(backend_cls, shape):
    """Packing 4D [B,C,H,W] -> 3D [B,(H/2)*(W/2),C*4]."""
    latents = torch.randn(*shape)
    packed = backend_cls._pack_latents(latents)
    B, C, H, W = shape
    assert packed.shape == (B, (H // 2) * (W // 2), C * 4)


def test_flux_pack_matches_diffusers_pipeline():
    """Flux packing must be byte-for-byte equivalent to FluxPipeline."""
    from diffusers import FluxPipeline
    B, C, H, W = 2, 16, 128, 96
    latents = torch.randn(B, C, H, W)
    official = FluxPipeline._pack_latents(latents.clone(), B, C, H, W)
    ours = FluxBackend._pack_latents(latents.clone())
    assert torch.allclose(official, ours)


def test_chroma_pack_matches_diffusers_pipeline():
    """Chroma packing must be byte-for-byte equivalent to ChromaPipeline."""
    from diffusers import ChromaPipeline
    B, C, H, W = 2, 16, 128, 128
    latents = torch.randn(B, C, H, W)
    official = ChromaPipeline._pack_latents(latents.clone(), B, C, H, W)
    ours = ChromaBackend._pack_latents(latents.clone())
    assert torch.allclose(official, ours)


@pytest.mark.parametrize("backend_cls", [FluxBackend, Flux2Backend, ChromaBackend])
def test_prepare_img_ids_shape(backend_cls):
    """img_ids is [h*w, 3] with channel 0 = 0, channel 1 = row, channel 2 = col."""
    ids = backend_cls._prepare_img_ids(64, 48, torch.device("cpu"), torch.float32)
    assert ids.shape == (64 * 48, 3)
    # First entry: row=0, col=0
    assert ids[0, 0].item() == 0
    assert ids[0, 1].item() == 0
    assert ids[0, 2].item() == 0
    # Last entry: row=63, col=47
    assert ids[-1, 1].item() == 63
    assert ids[-1, 2].item() == 47


def test_flux_prepare_img_ids_matches_diffusers():
    """img_ids construction must match FluxPipeline's _prepare_latent_image_ids."""
    from diffusers import FluxPipeline
    h, w = 64, 64
    device = torch.device("cpu")
    dtype = torch.float32
    official = FluxPipeline._prepare_latent_image_ids(1, h, w, device, dtype)
    ours = FluxBackend._prepare_img_ids(h, w, device, dtype)
    assert torch.allclose(official, ours)
