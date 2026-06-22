"""
Real-time diffusion engines — turn one camera frame into one filtered frame.

Two engines, both built on the pipeline the Generate tab already loaded (no
extra VRAM, no second model load):

* :class:`LeanRealtimeEngine` (default, ships working): few-step LCM/Turbo
  img2img. At 1-2 steps on an SD1.5 model this is already real-time on an
  8 GB RTX 3070 — the dominant cost is the handful of UNet forwards, and few
  steps means few forwards. Correct by construction: diffusers does the
  scheduler maths. Works for every img2img-capable architecture.

* :class:`StreamBatchEngine` (experimental, SD1.5/SD2): StreamDiffusion's
  "Stream Batch" — pipelines the denoising so each UNet forward advances a
  whole queue of frames at once, raising throughput when you want more steps.
  Marked experimental because the batched LCM update is hand-rolled and wants
  validation on a CUDA GPU.

Stochastic Similarity Filtering (skip near-identical frames) lives in the
worker, not here, so it applies to both engines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from dataset_sorter.realtime.stream_prompt import StreamPrompt

log = logging.getLogger(__name__)

# Architectures the lean engine knows how to build an img2img pipeline for.
# Flux/Flux2 are deliberately absent: their encode_prompt returns
# (prompt_embeds, pooled, text_ids) with no do_classifier_free_guidance arg, so
# the shared prompt path can't drive them, and at 12B they aren't real-time on
# 8 GB. SD3/SD3.5 share SDXL's 4-tuple encode_prompt, so they slot straight in.
_IMG2IMG_PIPELINE = {
    "sd15": "StableDiffusionImg2ImgPipeline",
    "sd2": "StableDiffusionImg2ImgPipeline",
    "sdxl": "StableDiffusionXLImg2ImgPipeline",
    "pony": "StableDiffusionXLImg2ImgPipeline",
    "sd3": "StableDiffusion3Img2ImgPipeline",
    "sd35": "StableDiffusion3Img2ImgPipeline",
}

# Flow-matching models keep their native FlowMatchEulerDiscrete scheduler —
# LCMScheduler lacks the mu/shift params the flow schedule needs (mirrors the
# FLOW_MATCHING_MODELS guard in generate_worker._load_scheduler). Forcing LCM on
# these produces garbage, so the few-step LCM swap must skip them.
_FLOW_MATCHING = {"sd3", "sd35", "flux", "flux2", "sana", "pixart",
                  "auraflow", "chroma", "hidream", "zimage"}
# Stream Batch is validated-by-design only for the SD1.5/SD2 UNet shape.
_STREAM_BATCH_OK = {"sd15", "sd2"}

# Tiny AutoEncoder (TAESD) repos — a ~10 MB VAE that encodes/decodes almost
# for free, which is the single biggest per-frame win for real-time (the full
# VAE is a large slice of few-step latency). This is what StreamDiffusion uses.
# Only listed for architectures the lean engine can actually drive (see
# _IMG2IMG_PIPELINE). taesd3 covers both SD3 and SD3.5 (shared latent space).
_TINY_VAE_REPO = {
    "sd15": "madebyollin/taesd",
    "sd2": "madebyollin/taesd",
    "sdxl": "madebyollin/taesdxl",
    "pony": "madebyollin/taesdxl",
    "sd3": "madebyollin/taesd3",
    "sd35": "madebyollin/taesd3",
}
# Cache loaded tiny VAEs by (repo, dtype-str) so pressing Start repeatedly
# doesn't re-download / re-instantiate them.
_tiny_vae_cache: dict = {}


@dataclass
class RealtimeParams:
    """Knobs for a real-time run. Mutable so the UI can tweak them live."""

    width: int = 512
    height: int = 512
    strength: float = 0.45          # how much the model reworks the frame (0-1)
    steps: int = 2                  # LCM/Turbo few-step count
    guidance_scale: float = 1.0     # 1.0 = no CFG (LCM real-time default)
    seed: int = -1                  # -1 = fresh noise each frame (temporal jitter)
    use_lcm_scheduler: bool = True  # force LCMScheduler for few-step denoising
    compile_unet: bool = False      # torch.compile the UNet (one-time cost, +20-30%)
    tiny_vae: bool = True           # swap in TAESD — near-free VAE, big real-time win
    channels_last: bool = True      # channels_last UNet/VAE memory layout (~10-20%)
    # Offload the text encoders to CPU between prompt changes. They aren't
    # needed per-frame (embeddings are cached), so this frees ~2 GB — the
    # difference between SDXL fitting or OOMing on an 8 GB card. "auto" enables
    # it only for SDXL/Pony, where it matters. Restored on stop.
    offload_text_encoders: str = "auto"   # "auto" | "on" | "off"


def build_engine(
    pipe,
    model_type: str,
    device,
    dtype,
    prompt: StreamPrompt,
    params: RealtimeParams,
    *,
    stream_batch: bool = False,
):
    """Pick and construct the engine for the loaded model.

    Falls back to the lean engine when Stream Batch isn't supported for the
    architecture, so the caller always gets a working engine.
    """
    if stream_batch and model_type in _STREAM_BATCH_OK:
        log.info("Real-time: using experimental Stream Batch engine (%s)", model_type)
        return StreamBatchEngine(pipe, model_type, device, dtype, prompt, params)
    if stream_batch:
        log.info(
            "Stream Batch not supported for %s — using lean few-step engine.",
            model_type,
        )
    return LeanRealtimeEngine(pipe, model_type, device, dtype, prompt, params)


class _BaseEngine:
    """Shared setup: LCM scheduler swap + prompt handle."""

    # Text-encoder attribute names across architectures (SDXL has two).
    _TE_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")

    def __init__(self, pipe, model_type, device, dtype, prompt, params):
        self.src_pipe = pipe
        self.model_type = model_type
        self.device = device
        self.dtype = self._resolve_dtype(dtype, pipe)
        self.prompt = prompt
        self.params = params
        self._do_cfg = params.guidance_scale > 1.0
        self._te_orig_devices: dict = {}     # for restore-on-stop
        self._te_offloaded = False

    @staticmethod
    def _resolve_dtype(dtype, pipe):
        """Return a real torch.dtype.

        GenerateWorker stores its dtype as a string ("torch.bfloat16"), so
        passing it straight to AutoencoderTiny.from_pretrained(torch_dtype=...)
        silently loads float32 and the VAE then mismatches the bf16 pipeline.
        Map known strings; otherwise fall back to the pipeline's actual
        compute dtype so the tiny VAE always matches.
        """
        import torch

        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            m = {
                "torch.bfloat16": torch.bfloat16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "torch.float16": torch.float16, "float16": torch.float16, "fp16": torch.float16,
                "torch.float32": torch.float32, "float32": torch.float32, "fp32": torch.float32,
            }
            if dtype in m:
                return m[dtype]
        # Fall back to the model's real dtype.
        mod = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        if mod is not None:
            try:
                return next(mod.parameters()).dtype
            except StopIteration:
                pass
        return torch.float32

    # ── text-encoder offload (frees ~2 GB on SDXL/8 GB) ──────────────────

    def _te_offload_active(self, pipe) -> bool:
        """Whether to manage text-encoder placement for this run."""
        mode = getattr(self.params, "offload_text_encoders", "auto")
        if mode == "off":
            return False
        if mode == "auto" and self.model_type not in ("sdxl", "pony"):
            return False
        # Don't fight diffusers' own model_cpu_offload hooks if they're active.
        for name in self._TE_ATTRS:
            te = getattr(pipe, name, None)
            if te is not None and hasattr(te, "_hf_hook"):
                log.info("Text-encoder offload skipped: pipeline cpu_offload is active.")
                return False
        return True

    def _set_te_device(self, pipe, device) -> None:
        for name in self._TE_ATTRS:
            te = getattr(pipe, name, None)
            if te is not None and hasattr(te, "to"):
                te.to(device)

    def _encode_managed(self, pipe):
        """Encode the prompt, bringing text encoders on-GPU only when a real
        (re)encode is needed, then offloading them again."""
        if not self._te_offload_active(pipe):
            return self.prompt.encode(pipe, self.device, self.dtype, do_cfg=self._do_cfg)

        if self.prompt.is_cached(self._do_cfg):
            return self.prompt.encode(pipe, self.device, self.dtype, do_cfg=self._do_cfg)

        # Stale → record original placement once, run the encode on-device,
        # then push the encoders back to CPU.
        if not self._te_orig_devices:
            import torch
            for name in self._TE_ATTRS:
                te = getattr(pipe, name, None)
                if te is not None:
                    try:
                        self._te_orig_devices[name] = next(te.parameters()).device
                    except StopIteration:
                        pass
        self._set_te_device(pipe, self.device)
        enc = self.prompt.encode(pipe, self.device, self.dtype, do_cfg=self._do_cfg)
        self._set_te_device(pipe, "cpu")
        self._te_offloaded = True
        try:
            import torch
            if self.device is not None and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return enc

    def teardown(self) -> None:
        """Restore text encoders to their original device so the Generate tab
        keeps working after a live session. Safe to call multiple times."""
        if not self._te_offloaded:
            return
        pipe = getattr(self, "_pipe", None) or self.src_pipe
        for name, dev in self._te_orig_devices.items():
            te = getattr(pipe, name, None)
            if te is not None and hasattr(te, "to"):
                try:
                    te.to(dev)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not restore %s to %s: %s", name, dev, exc)
        self._te_offloaded = False

    def _maybe_swap_to_lcm(self, pipe) -> None:
        """Swap in LCMScheduler when requested (few-step denoising needs it).

        Flow-matching models (SD3/SD3.5/Flux/...) keep their native scheduler:
        LCMScheduler lacks the mu/shift params their timestep schedule needs, so
        swapping it in produces garbage. Their few-step img2img runs on the
        native FlowMatchEuler scheduler instead.
        """
        if not self.params.use_lcm_scheduler:
            return
        if self.model_type in _FLOW_MATCHING:
            log.info("Keeping native scheduler for flow-matching model '%s'", self.model_type)
            return
        try:
            from diffusers import LCMScheduler
            if type(pipe.scheduler).__name__ != "LCMScheduler":
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not set LCMScheduler (keeping current): %s", exc)

    def _maybe_use_tiny_vae(self, pipe) -> None:
        """Swap the pipeline's VAE for TAESD — the key real-time speedup.

        TAESD is a tiny distilled autoencoder: its encode+decode cost is
        negligible next to the full VAE, which otherwise dominates few-step
        latency. Cached across Start presses. Any failure (offline, unknown
        architecture) silently keeps the full VAE so the filter still runs.
        """
        if not getattr(self.params, "tiny_vae", True):
            return
        repo = _TINY_VAE_REPO.get(self.model_type)
        if repo is None:
            return
        try:
            from diffusers import AutoencoderTiny

            key = (repo, str(self.dtype))
            tiny = _tiny_vae_cache.get(key)
            if tiny is None:
                tiny = AutoencoderTiny.from_pretrained(repo, torch_dtype=self.dtype)
                _tiny_vae_cache[key] = tiny
            pipe.vae = tiny
            if not self._has_cpu_offload(pipe):
                tiny.to(self.device)
            log.info("Real-time: using TAESD tiny VAE (%s)", repo)
        except Exception as exc:  # noqa: BLE001
            log.warning("TAESD tiny VAE unavailable (keeping full VAE): %s", exc)

    @staticmethod
    def _has_cpu_offload(pipe) -> bool:
        return hasattr(pipe, "_all_hooks") and len(getattr(pipe, "_all_hooks", [])) > 0

    def _rearm_cpu_offload(self, pipe) -> None:
        """Re-register model-cpu-offload hooks after swapping components.

        When the pipeline was loaded with enable_model_cpu_offload (VRAM < 16 GB)
        and we swap the VAE for TAESD, the new VAE has no offload hook, so
        diffusers never moves it to GPU and the forward call crashes. Calling
        enable_model_cpu_offload again re-registers hooks for all components
        including the new VAE.
        """
        if not self._has_cpu_offload(pipe):
            return
        try:
            pipe.enable_model_cpu_offload()
            log.info("Real-time: re-armed cpu_offload hooks after component swap")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not re-arm cpu_offload: %s", exc)

    def _maybe_channels_last(self, pipe) -> None:
        """Put the UNet/VAE in channels_last layout (~10-20% on conv-heavy nets)."""
        if not getattr(self.params, "channels_last", True):
            return
        try:
            import torch

            for name in ("unet", "vae"):
                mod = getattr(pipe, name, None)
                if mod is not None and hasattr(mod, "to"):
                    mod.to(memory_format=torch.channels_last)
        except Exception as exc:  # noqa: BLE001
            log.debug("channels_last not applied: %s", exc)

    def _maybe_compile_unet(self, pipe) -> None:
        """torch.compile the UNet for a steady-state speedup (LivePortrait-style).

        A fixed real-time resolution means the one-time compile cost is paid
        once and amortised over the whole session. Guarded: any failure (old
        torch, Windows/Triton gaps) silently keeps the eager UNet.
        """
        if not getattr(self.params, "compile_unet", False):
            return
        try:
            import torch

            unet = getattr(pipe, "unet", None)
            if unet is not None:
                pipe.unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
                log.info("Real-time: UNet compiled (first few frames will be slower)")
        except Exception as exc:  # noqa: BLE001
            log.warning("torch.compile of UNet skipped: %s", exc)

    def prepare(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def process(self, frame):  # pragma: no cover - overridden
        raise NotImplementedError


class LeanRealtimeEngine(_BaseEngine):
    """Few-step LCM/Turbo img2img — the default, always-works engine."""

    def __init__(self, pipe, model_type, device, dtype, prompt, params):
        super().__init__(pipe, model_type, device, dtype, prompt, params)
        self._pipe = None  # img2img pipeline sharing the source's weights

    def prepare(self) -> None:
        """Build an img2img pipeline from the loaded model's components.

        ``pipe.components`` shares the already-resident weights, so this adds
        no VRAM. Falls back to the source pipeline if a dedicated img2img
        class isn't available for the architecture.
        """
        import diffusers

        cls_name = _IMG2IMG_PIPELINE.get(self.model_type)
        target = self.src_pipe
        if cls_name is not None and hasattr(diffusers, cls_name):
            try:
                cls = getattr(diffusers, cls_name)
                target = cls(**self.src_pipe.components)
                target.set_progress_bar_config(disable=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("img2img pipeline build failed (%s) — using base", exc)
                target = self.src_pipe
        self._maybe_swap_to_lcm(target)
        self._maybe_use_tiny_vae(target)
        self._maybe_channels_last(target)
        self._maybe_compile_unet(target)
        self._rearm_cpu_offload(target)
        self._pipe = target

    def process(self, frame):
        """Run one img2img pass; returns a filtered RGB PIL.Image."""
        import torch

        p = self.params
        enc = self._encode_managed(self._pipe)
        frame = frame.convert("RGB").resize((p.width, p.height))

        gen = None
        if p.seed >= 0:
            gen = torch.Generator(device="cpu").manual_seed(p.seed)

        call = {
            "image": frame,
            "strength": max(0.05, min(1.0, p.strength)),
            "num_inference_steps": max(1, p.steps),
            "guidance_scale": p.guidance_scale,
            "prompt_embeds": enc.prompt_embeds,
            "output_type": "pil",
            "generator": gen,
        }
        if self._do_cfg and enc.negative_prompt_embeds is not None:
            call["negative_prompt_embeds"] = enc.negative_prompt_embeds
        # SDXL-class pipelines need the pooled embeddings.
        if enc.pooled_prompt_embeds is not None:
            call["pooled_prompt_embeds"] = enc.pooled_prompt_embeds
            if self._do_cfg and enc.negative_pooled_prompt_embeds is not None:
                call["negative_pooled_prompt_embeds"] = enc.negative_pooled_prompt_embeds

        with torch.inference_mode():
            out = self._pipe(**call)
        return out.images[0]


class StreamBatchEngine(_BaseEngine):
    """Experimental pipelined batched denoiser (StreamDiffusion Stream Batch).

    Keeps a queue of ``denoise_steps`` latents, each one step further along.
    Every call: VAE-encode the new frame, prepend it, run ONE batched UNet
    forward over the whole queue, apply the LCM update to all of them, emit the
    most-denoised (which then leaves the queue). Throughput approaches one
    decoded frame per UNet forward instead of per ``denoise_steps`` forwards.

    SD1.5/SD2 only (UNet conditioning is just ``encoder_hidden_states``).
    """

    def __init__(self, pipe, model_type, device, dtype, prompt, params):
        super().__init__(pipe, model_type, device, dtype, prompt, params)
        # Stream Batch runs a single UNet forward (no CFG branch), so the
        # prompt is always encoded without negatives.
        self._do_cfg = False
        self._pipe = pipe
        self._timesteps = None       # the img2img sub-schedule (tensor)
        self._buffer = None          # queued latents at successive noise levels
        self._c_skip = None          # per-step LCM boundary coefficients
        self._c_out = None
        self._alpha_sqrt = None
        self._beta_sqrt = None

    def prepare(self) -> None:
        import torch

        self._maybe_swap_to_lcm(self._pipe)
        self._maybe_use_tiny_vae(self._pipe)
        self._maybe_channels_last(self._pipe)
        self._maybe_compile_unet(self._pipe)
        self._rearm_cpu_offload(self._pipe)
        sched = self._pipe.scheduler
        n = max(1, self.params.steps)
        sched.set_timesteps(n, device=self.device)

        # img2img start index from strength (same rule diffusers uses).
        init = min(int(n * self.params.strength), n)
        t_start = max(n - init, 0)
        timesteps = sched.timesteps[t_start:]
        self._timesteps = timesteps

        # Precompute the LCM consistency coefficients per queued timestep so the
        # batched update is a pure vectorised op (no stateful scheduler.step).
        # Computed in fp32 for precision, then cast to the latent dtype so the
        # arithmetic in process() doesn't promote bf16/fp16 latents to fp32 and
        # then feed a mismatched dtype into the UNet.
        alphas_cumprod = sched.alphas_cumprod.to(self.device)
        sigma_data = 0.5
        ts = timesteps.long()
        alpha_prod = alphas_cumprod[ts]                       # (k,)
        self._alpha_sqrt = alpha_prod.sqrt().view(-1, 1, 1, 1).to(self.dtype)
        self._beta_sqrt = (1 - alpha_prod).sqrt().view(-1, 1, 1, 1).to(self.dtype)
        # LCM scaled timestep (the 0.1 follows the LCM paper / LCMScheduler).
        scaled = ts.float() * 0.1
        denom = scaled.pow(2) + sigma_data ** 2
        self._c_skip = (sigma_data ** 2 / denom).view(-1, 1, 1, 1).to(self.dtype)
        self._c_out = (scaled * sigma_data / denom.sqrt()).view(-1, 1, 1, 1).to(self.dtype)
        self._buffer = None

    def _is_tiny_vae(self) -> bool:
        return type(self._pipe.vae).__name__ == "AutoencoderTiny"

    def _vae_scale(self) -> float:
        # TAESD operates in unit scale (scaling_factor == 1.0); the full VAE
        # uses its configured scaling_factor.
        return float(getattr(self._pipe.vae.config, "scaling_factor", 1.0))

    def _encode_frame(self, frame):
        import torch

        p = self.params
        frame = frame.convert("RGB").resize((p.width, p.height))
        img = self._pipe.image_processor.preprocess(frame).to(self.device, self.dtype)
        with torch.inference_mode():
            enc = self._pipe.vae.encode(img)
            # AutoencoderTiny returns .latents; the full VAE returns .latent_dist.
            latent = enc.latents if self._is_tiny_vae() else enc.latent_dist.sample()
        return latent * self._vae_scale()

    def _decode_latent(self, latent):
        import torch

        with torch.inference_mode():
            img = self._pipe.vae.decode(latent / self._vae_scale()).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return self._pipe.image_processor.postprocess(img, output_type="pil")[0]

    def process(self, frame):
        import torch

        z0 = self._encode_frame(frame)                        # (1,C,H,W)
        k = len(self._timesteps)
        enc = self._encode_managed(self._pipe)

        # Noise the fresh frame to the first (noisiest) queued timestep.
        noise0 = torch.randn_like(z0)
        noisy0 = self._alpha_sqrt[0] * z0 + self._beta_sqrt[0] * noise0

        if self._buffer is None or k == 1:
            batch = noisy0
        else:
            batch = torch.cat([noisy0, self._buffer], dim=0)  # (k,C,H,W)

        t_batch = self._timesteps[: batch.shape[0]]
        embeds = enc.prompt_embeds.expand(batch.shape[0], -1, -1)

        with torch.inference_mode():
            model_out = self._pipe.unet(
                batch, t_batch, encoder_hidden_states=embeds
            ).sample

        # epsilon -> predicted x0, then LCM consistency boundary.
        n = batch.shape[0]
        a_sqrt = self._alpha_sqrt[:n]
        b_sqrt = self._beta_sqrt[:n]
        x0 = (batch - b_sqrt * model_out) / a_sqrt
        denoised = self._c_skip[:n] * batch + self._c_out[:n] * x0

        # The most-denoised entry (last) is finished and leaves the queue.
        finished = denoised[-1:].detach()
        out_img = self._decode_latent(finished)

        # Re-noise the rest forward to their NEXT (less noisy) timestep so they
        # are ready for the following UNet forward.
        if n > 1:
            keep = denoised[:-1]
            a_next = self._alpha_sqrt[1:n]
            b_next = self._beta_sqrt[1:n]
            noise = torch.randn_like(keep)
            self._buffer = (a_next * keep + b_next * noise).detach()
        else:
            self._buffer = None
        return out_img
