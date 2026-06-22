"""Tests for the real-time Live Filter feature (camera → diffusion → frame).

Covers the GPU-free logic: prompt-embedding caching, the engine factory's
architecture gating/fallback, params plumbing, camera enumeration degradation,
and the similarity filter. The actual diffusion forward needs CUDA and is not
exercised here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


# ── Camera source ────────────────────────────────────────────────────────────

class TestCameraSource:
    def test_device_label(self):
        from dataset_sorter.realtime.camera_source import CameraDevice
        assert CameraDevice(2, "Canon EOS R5").label == "2: Canon EOS R5"

    def test_enumeration_without_opencv_returns_empty(self, monkeypatch):
        """No opencv installed → graceful empty list, never a crash."""
        import builtins
        from dataset_sorter.realtime import camera_source

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "cv2":
                raise ImportError("no cv2")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert camera_source.list_camera_devices() == []


# ── StreamPrompt: encode only on change ──────────────────────────────────────

class _FakePipe:
    def __init__(self):
        self.calls = 0

    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1,
                      do_classifier_free_guidance=False, negative_prompt=None,
                      clip_skip=0):
        self.calls += 1
        return (f"EMB({prompt})", "NEG")


class TestStreamPrompt:
    def test_caches_until_text_changes(self):
        from dataset_sorter.realtime.stream_prompt import StreamPrompt
        pipe = _FakePipe()
        sp = StreamPrompt("a cat")

        sp.encode(pipe, "cpu", None, do_cfg=False)
        sp.encode(pipe, "cpu", None, do_cfg=False)
        assert pipe.calls == 1  # second call served from cache

        sp.set("a dog")
        out = sp.encode(pipe, "cpu", None, do_cfg=False)
        assert pipe.calls == 2  # recomputed after change
        assert out.prompt_embeds == "EMB(a dog)"

    def test_cfg_toggle_busts_cache(self):
        from dataset_sorter.realtime.stream_prompt import StreamPrompt
        pipe = _FakePipe()
        sp = StreamPrompt("x")
        sp.encode(pipe, "cpu", None, do_cfg=False)
        sp.encode(pipe, "cpu", None, do_cfg=True)
        assert pipe.calls == 2

    def test_sdxl_four_tuple_return(self):
        from dataset_sorter.realtime.stream_prompt import StreamPrompt

        class XLPipe:
            def encode_prompt(self, prompt, **k):
                return ("pe", "ne", "pooled", "neg_pooled")

        sp = StreamPrompt("p")
        out = sp.encode(XLPipe(), "cpu", None, do_cfg=True)
        assert out.prompt_embeds == "pe"
        assert out.pooled_prompt_embeds == "pooled"
        assert out.negative_pooled_prompt_embeds == "neg_pooled"


# ── Engine factory: gating + fallback ────────────────────────────────────────

class TestEngineFactory:
    def _make(self, model_type, stream_batch):
        from dataset_sorter.realtime.stream_engine import (
            RealtimeParams, build_engine,
        )
        from dataset_sorter.realtime.stream_prompt import StreamPrompt
        return build_engine(
            object(), model_type, "cpu", None, StreamPrompt(),
            RealtimeParams(), stream_batch=stream_batch,
        )

    def test_default_is_lean(self):
        from dataset_sorter.realtime.stream_engine import LeanRealtimeEngine
        assert isinstance(self._make("sd15", False), LeanRealtimeEngine)

    def test_stream_batch_for_sd15(self):
        from dataset_sorter.realtime.stream_engine import StreamBatchEngine
        assert isinstance(self._make("sd15", True), StreamBatchEngine)

    def test_stream_batch_falls_back_for_sdxl(self):
        from dataset_sorter.realtime.stream_engine import LeanRealtimeEngine
        # SDXL UNet conditioning isn't supported by the batch path → lean.
        assert isinstance(self._make("sdxl", True), LeanRealtimeEngine)

    def test_stream_batch_falls_back_for_flux(self):
        from dataset_sorter.realtime.stream_engine import LeanRealtimeEngine
        assert isinstance(self._make("flux", True), LeanRealtimeEngine)

    def test_sd3_gets_lean_engine(self):
        from dataset_sorter.realtime.stream_engine import LeanRealtimeEngine
        assert isinstance(self._make("sd3", False), LeanRealtimeEngine)

    def test_sd35_gets_lean_engine(self):
        from dataset_sorter.realtime.stream_engine import LeanRealtimeEngine
        assert isinstance(self._make("sd35", False), LeanRealtimeEngine)

    def test_img2img_pipeline_mapping(self):
        from dataset_sorter.realtime.stream_engine import _IMG2IMG_PIPELINE
        assert "sd3" in _IMG2IMG_PIPELINE
        assert "sd35" in _IMG2IMG_PIPELINE
        assert "Diffusion3" in _IMG2IMG_PIPELINE["sd3"]
        # Flux is intentionally NOT supported: its encode_prompt is incompatible
        # with the shared prompt path and it's too heavy for 8 GB real-time.
        assert "flux" not in _IMG2IMG_PIPELINE
        assert "flux2" not in _IMG2IMG_PIPELINE


class TestRealtimeParams:
    def test_defaults_are_realtime_friendly(self):
        from dataset_sorter.realtime.stream_engine import RealtimeParams
        p = RealtimeParams()
        assert p.width == 512 and p.height == 512
        assert p.steps <= 2          # few-step by default
        assert p.guidance_scale == 1.0  # no CFG → single forward
        assert p.compile_unet is False
        # Real-time speedups on by default (the whole point of the feature).
        assert p.tiny_vae is True
        assert p.channels_last is True


class TestTinyVAE:
    def test_repo_mapping(self):
        from dataset_sorter.realtime.stream_engine import (
            _TINY_VAE_REPO, _IMG2IMG_PIPELINE,
        )
        assert _TINY_VAE_REPO["sd15"] == "madebyollin/taesd"
        assert _TINY_VAE_REPO["sd2"] == "madebyollin/taesd"
        assert _TINY_VAE_REPO["sdxl"] == "madebyollin/taesdxl"
        assert _TINY_VAE_REPO["pony"] == "madebyollin/taesdxl"
        assert _TINY_VAE_REPO["sd3"] == "madebyollin/taesd3"
        assert _TINY_VAE_REPO["sd35"] == "madebyollin/taesd3"
        # Every TAESD entry must correspond to a drivable img2img architecture —
        # no tiny VAE for a model the engine can't actually run.
        assert set(_TINY_VAE_REPO).issubset(set(_IMG2IMG_PIPELINE))

    def test_tiny_vae_disabled_is_noop(self):
        """tiny_vae=False must not touch the pipeline VAE."""
        from dataset_sorter.realtime.stream_engine import (
            LeanRealtimeEngine, RealtimeParams,
        )
        from dataset_sorter.realtime.stream_prompt import StreamPrompt

        class FakePipe:
            vae = "ORIGINAL_VAE"

        eng = LeanRealtimeEngine(
            FakePipe(), "sd15", "cpu", None, StreamPrompt(),
            RealtimeParams(tiny_vae=False),
        )
        pipe = FakePipe()
        eng._maybe_use_tiny_vae(pipe)  # disabled → no change, no import
        assert pipe.vae == "ORIGINAL_VAE"

    def test_lcm_swap_skipped_for_flow_matching(self):
        """SD3/SD3.5 must keep their native FlowMatchEuler scheduler — forcing
        LCMScheduler (the default request) would corrupt flow-matching output."""
        from dataset_sorter.realtime.stream_engine import (
            LeanRealtimeEngine, RealtimeParams,
        )
        from dataset_sorter.realtime.stream_prompt import StreamPrompt

        class FakeSched:
            config = {}

        class FakePipe:
            scheduler = FakeSched()

        for mt in ("sd3", "sd35"):
            eng = LeanRealtimeEngine(
                FakePipe(), mt, "cpu", None, StreamPrompt(),
                RealtimeParams(use_lcm_scheduler=True),
            )
            pipe = FakePipe()
            orig = pipe.scheduler
            eng._maybe_swap_to_lcm(pipe)
            assert pipe.scheduler is orig, f"{mt} scheduler was swapped"

    def test_flow_matching_set_matches_generate_worker(self):
        """The engine's flow-matching guard must not drift from the generate
        worker's authoritative set (the reason the scheduler bug existed)."""
        from dataset_sorter.realtime.stream_engine import _FLOW_MATCHING
        from dataset_sorter.generate_worker import FLOW_MATCHING_MODELS
        # Every model generate_worker treats as flow-matching must also be
        # guarded here; epsilon models (sdxl/sd15) must be in neither.
        assert FLOW_MATCHING_MODELS.issubset(_FLOW_MATCHING)
        assert "sdxl" not in _FLOW_MATCHING
        assert "sd15" not in _FLOW_MATCHING

    def test_unknown_arch_skips_tiny_vae(self):
        from dataset_sorter.realtime.stream_engine import (
            LeanRealtimeEngine, RealtimeParams,
        )
        from dataset_sorter.realtime.stream_prompt import StreamPrompt

        class FakePipe:
            vae = "ORIGINAL_VAE"

        eng = LeanRealtimeEngine(
            FakePipe(), "kolors", "cpu", None, StreamPrompt(),
            RealtimeParams(tiny_vae=True),
        )
        pipe = FakePipe()
        eng._maybe_use_tiny_vae(pipe)  # no TAESD repo for kolors → unchanged
        assert pipe.vae == "ORIGINAL_VAE"

    def test_resolve_dtype_maps_generate_worker_string(self):
        """GenerateWorker stores dtype as a string; the engine must map it to a
        real torch.dtype or the tiny VAE loads float32 and mismatches bf16."""
        import torch
        from dataset_sorter.realtime.stream_engine import _BaseEngine

        class FakePipe:
            unet = None
        assert _BaseEngine._resolve_dtype("torch.bfloat16", FakePipe()) is torch.bfloat16
        assert _BaseEngine._resolve_dtype("torch.float16", FakePipe()) is torch.float16
        assert _BaseEngine._resolve_dtype("torch.float32", FakePipe()) is torch.float32

    def test_resolve_dtype_passes_through_real_dtype(self):
        import torch
        from dataset_sorter.realtime.stream_engine import _BaseEngine

        class FakePipe:
            unet = None
        assert _BaseEngine._resolve_dtype(torch.bfloat16, FakePipe()) is torch.bfloat16

    def test_resolve_dtype_falls_back_to_pipeline(self):
        """An unknown/None dtype falls back to the model's actual dtype."""
        import torch
        from dataset_sorter.realtime.stream_engine import _BaseEngine

        unet = torch.nn.Linear(2, 2).to(torch.float16)

        class FakePipe:
            pass
        pipe = FakePipe()
        pipe.unet = unet
        assert _BaseEngine._resolve_dtype(None, pipe) is torch.float16


# ── Text-encoder offload (makes SDXL fit on 8 GB) ────────────────────────────

class _FakeTE:
    def __init__(self, device="gpu"):
        self._device = device
        self.to_calls = []

    def to(self, device):
        self._device = device
        self.to_calls.append(str(device))
        return self

    def parameters(self):
        yield type("P", (), {"device": self._device})()


class _FakePromptPipe:
    def __init__(self):
        self.text_encoder = _FakeTE()
        self.text_encoder_2 = _FakeTE()
        self.calls = 0

    def encode_prompt(self, prompt, **k):
        self.calls += 1
        return ("pe", "ne", "pooled", "neg_pooled")  # SDXL-shape


class TestTextEncoderOffload:
    def _engine(self, model_type, mode):
        from dataset_sorter.realtime.stream_engine import (
            LeanRealtimeEngine, RealtimeParams,
        )
        from dataset_sorter.realtime.stream_prompt import StreamPrompt
        eng = LeanRealtimeEngine(
            _FakePromptPipe(), model_type, "gpu", None, StreamPrompt("neon"),
            RealtimeParams(offload_text_encoders=mode),
        )
        return eng

    def test_auto_active_for_sdxl_only(self):
        sdxl = self._engine("sdxl", "auto")
        sd15 = self._engine("sd15", "auto")
        assert sdxl._te_offload_active(sdxl.src_pipe) is True
        assert sd15._te_offload_active(sd15.src_pipe) is False

    def test_off_disables_even_for_sdxl(self):
        eng = self._engine("sdxl", "off")
        assert eng._te_offload_active(eng.src_pipe) is False

    def test_skips_when_cpu_offload_hooks_present(self):
        eng = self._engine("sdxl", "on")
        eng.src_pipe.text_encoder._hf_hook = object()  # simulate model_cpu_offload
        assert eng._te_offload_active(eng.src_pipe) is False

    def test_encode_offloads_then_teardown_restores(self):
        eng = self._engine("sdxl", "on")
        eng._pipe = eng.src_pipe
        te = eng.src_pipe.text_encoder

        enc = eng._encode_managed(eng.src_pipe)
        assert enc.prompt_embeds == "pe"
        # After a real encode the TE was pulled to gpu then pushed to cpu.
        assert te.to_calls[-1] == "cpu"
        assert eng._te_offloaded is True

        eng.teardown()
        assert te.to_calls[-1] == "gpu"   # original device restored
        assert eng._te_offloaded is False

    def test_cached_prompt_does_not_move_encoders(self):
        eng = self._engine("sdxl", "on")
        eng._pipe = eng.src_pipe
        eng._encode_managed(eng.src_pipe)          # first: encodes + offloads
        eng.src_pipe.text_encoder.to_calls.clear()
        eng._encode_managed(eng.src_pipe)          # cached: no TE movement
        assert eng.src_pipe.text_encoder.to_calls == []
        assert eng.src_pipe.calls == 1             # encode_prompt fired once


# ── Worker: configuration + similarity filter (no Qt loop needed) ────────────

class TestRealtimeWorkerLogic:
    @pytest.fixture
    def qapp(self, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
        except Exception:
            pytest.skip("PyQt6 not available")
        return QApplication.instance() or QApplication([])

    def test_ssf_off_processes_everything(self, qapp):
        from dataset_sorter.realtime.realtime_worker import RealtimeWorker
        w = RealtimeWorker(generate_worker=None)
        w._ssf_threshold = 0.0
        w._last_processed = Image.new("RGB", (64, 64), (10, 10, 10))
        assert w._should_skip(Image.new("RGB", (64, 64), (200, 200, 200))) is False

    def test_ssf_skips_identical_frame(self, qapp):
        from dataset_sorter.realtime.realtime_worker import RealtimeWorker
        w = RealtimeWorker(generate_worker=None)
        w._ssf_threshold = 0.05
        same = Image.new("RGB", (64, 64), (123, 123, 123))
        w._last_processed = same
        assert w._should_skip(same.copy()) is True

    def test_ssf_processes_very_different_frame(self, qapp):
        from dataset_sorter.realtime.realtime_worker import RealtimeWorker
        w = RealtimeWorker(generate_worker=None)
        w._ssf_threshold = 0.05
        w._last_processed = Image.new("RGB", (64, 64), (0, 0, 0))
        assert w._should_skip(Image.new("RGB", (64, 64), (255, 255, 255))) is False

    def test_run_without_model_emits_error(self, qapp):
        from dataset_sorter.realtime.realtime_worker import RealtimeWorker
        w = RealtimeWorker(generate_worker=None)
        errors = []
        w.error.connect(errors.append)
        w.run()  # no model → should emit and return, not raise
        assert errors and "model" in errors[0].lower()

    def test_configure_sets_state(self, qapp):
        from dataset_sorter.realtime.realtime_worker import RealtimeWorker
        from dataset_sorter.realtime.stream_engine import RealtimeParams
        w = RealtimeWorker(generate_worker=None)
        w.configure(
            camera_index=3, params=RealtimeParams(steps=1),
            positive="neon", negative="blurry",
            stream_batch=True, ssf_threshold=0.02,
        )
        assert w._camera_index == 3
        assert w._stream_batch is True
        assert w._prompt.positive == "neon"


# ── UI tab smoke ─────────────────────────────────────────────────────────────

class TestLiveFilterTab:
    @pytest.fixture
    def qapp(self, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
        except Exception:
            pytest.skip("PyQt6 not available")
        return QApplication.instance() or QApplication([])

    def test_tab_builds_and_disabled_without_model(self, qapp):
        from dataset_sorter.ui.live_filter_tab import LiveFilterTab
        tab = LiveFilterTab()
        assert tab._btn_start.isEnabled() is False  # no model yet

    def test_set_worker_enables_start(self, qapp):
        from dataset_sorter.ui.live_filter_tab import LiveFilterTab
        tab = LiveFilterTab()

        class FakeWorker:
            is_loaded = True
        tab.set_generate_worker(FakeWorker())
        assert tab._btn_start.isEnabled() is True
        assert "ready" in tab._model_hint.text().lower()

    def test_current_params_reflect_widgets(self, qapp):
        from dataset_sorter.ui.live_filter_tab import LiveFilterTab
        tab = LiveFilterTab()
        tab._strength_spin.setValue(0.6)
        tab._steps_spin.setValue(4)
        p = tab._current_params()
        assert abs(p.strength - 0.6) < 1e-9
        assert p.steps == 4

    def test_tiny_vae_on_by_default(self, qapp):
        from dataset_sorter.ui.live_filter_tab import LiveFilterTab
        tab = LiveFilterTab()
        assert tab._tiny_vae_check.isChecked() is True
        assert tab._current_params().tiny_vae is True
