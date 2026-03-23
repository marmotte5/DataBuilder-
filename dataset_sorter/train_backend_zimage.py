"""Z-Image training backend.

Architecture: ZImageTransformer2DModel (MMDiT variant).
Prediction: flow matching (rectified flow).
Resolution: 1024x1024 native.
Text encoder: Qwen3ForCausalLM (via Qwen2Tokenizer with chat template).

Z-Image uses a custom transformer architecture with Qwen3 as
the text encoder. This is fundamentally different from SD3 despite
both using flow matching.

Key differences from SD3:
- Qwen3 LLM text encoder (not CLIP/T5)
- Chat-template tokenization with thinking enabled
- ZImageTransformer2DModel (not SD3Transformer2DModel)
- Custom latent scaling with shift_factor
- Dynamic timestep shifting based on image dimensions
- Only transformer is trained; VAE and TE are always frozen
"""

import logging

import torch

from dataset_sorter.train_backend_base import TrainBackendBase
from dataset_sorter.utils import autocast_device_type

log = logging.getLogger(__name__)

# Max token length for Qwen3 text encoder.
_QWEN3_MAX_LENGTH = 512


def _apply_chat_template(tokenizer, caption: str) -> str:
    """Format a raw caption string through the Qwen3 chat template.

    Wraps the caption as a user message and applies the tokenizer's chat
    template with thinking enabled. This must be applied identically in
    both encode_text_batch (live) and the TE caching path to ensure
    embeddings match. Falls back to the raw caption if templating fails.
    """
    messages = [{"role": "user", "content": caption}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        # Fallback: use raw caption if chat template fails
        return caption


class ZImageBackend(TrainBackendBase):
    """Z-Image training backend (Qwen3 + ZImageTransformer2D)."""

    model_name = "zimage"
    default_resolution = 1024
    supports_dual_te = False
    prediction_type = "flow"

    def _get_te_load_kwargs(self) -> dict:
        """Build kwargs for quantized text encoder loading via bitsandbytes.

        Returns extra from_pretrained kwargs when quantize_text_encoder is
        'int8' or 'int4'. Saves 50-75% TE VRAM for the frozen Qwen3 encoder
        with zero quality impact (no gradients flow through it).
        """
        quant = self.config.quantize_text_encoder
        if quant == "int8":
            try:
                import bitsandbytes  # noqa: F401
                log.info("Z-Image TE: loading Qwen3 in INT8 (saves ~50%% TE VRAM)")
                return {"load_in_8bit": True, "device_map": "auto"}
            except ImportError:
                log.warning("bitsandbytes not installed; skipping INT8 TE quantization")
        elif quant == "int4":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                log.info("Z-Image TE: loading Qwen3 in INT4/NF4 (saves ~75%% TE VRAM)")
                return {"quantization_config": bnb_config, "device_map": "auto"}
            except ImportError:
                log.warning("bitsandbytes not installed; skipping INT4 TE quantization")
        return {}

    def _load_single_file_pipeline(self, model_path: str):
        """Try multiple methods to load a single .safetensors/.ckpt file.

        Attempts in order:
        1. DiffusionPipeline.from_single_file (diffusers >= 0.25)
        2. StableDiffusion3Pipeline.from_single_file (SD3-compatible)
        3. Manual state-dict extraction (split by component prefix)
        Returns a pipeline object, or None if manual loading was used
        (in which case components are set directly on self).
        """
        errors = []

        # Try DiffusionPipeline.from_single_file
        try:
            from diffusers import DiffusionPipeline
            if hasattr(DiffusionPipeline, 'from_single_file'):
                return DiffusionPipeline.from_single_file(
                    model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
            errors.append("DiffusionPipeline has no from_single_file method")
        except Exception as e:
            errors.append(f"DiffusionPipeline.from_single_file: {e}")

        # Try StableDiffusion3Pipeline.from_single_file (Z-Image is SD3-like)
        try:
            from diffusers import StableDiffusion3Pipeline
            if hasattr(StableDiffusion3Pipeline, 'from_single_file'):
                return StableDiffusion3Pipeline.from_single_file(
                    model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
            errors.append("StableDiffusion3Pipeline has no from_single_file method")
        except Exception as e:
            errors.append(f"StableDiffusion3Pipeline.from_single_file: {e}")

        # Manual loading: extract component state dicts from the single file
        log.info("Pipeline loaders failed; attempting manual single-file loading")
        try:
            self._load_single_file_manual(model_path)
            return None  # signals that components were loaded directly
        except Exception as e:
            errors.append(f"Manual single-file loading: {e}")

        raise RuntimeError(
            f"All single-file loading methods failed for '{model_path}':\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nYou can also convert the model to diffusers format and "
            "point to the directory instead."
        )

    # Official HuggingFace repos for Z-Image (Alibaba Tongyi Lab).
    _HF_BASE_REPO = "Tongyi-MAI/Z-Image"
    _HF_TURBO_REPO = "Tongyi-MAI/Z-Image-Turbo"

    def _load_single_file_manual(self, model_path: str):
        """Manually load a Z-Image single-file checkpoint by splitting the state dict.

        Reads the .safetensors (or .ckpt) file, partitions keys by component
        prefix (text_encoder, transformer/unet, vae), strips the prefix, and
        loads each component individually. This avoids needing diffusers'
        from_single_file to understand Z-Image's custom architecture.
        """
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

        # Load the raw state dict
        if model_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                full_sd = load_file(model_path, device="cpu")
            except ImportError:
                raise RuntimeError(
                    "safetensors package is required for single-file loading. "
                    "Install with: pip install safetensors"
                )
        else:
            full_sd = torch.load(model_path, map_location="cpu", weights_only=True)
            # .ckpt files may wrap the state dict under a key
            if "state_dict" in full_sd:
                full_sd = full_sd["state_dict"]

        # Identify component prefixes by scanning keys
        prefix_map = {
            "text_encoder.": "text_encoder",
            "transformer.": "transformer",
            "unet.": "transformer",  # alias
            "vae.": "vae",
        }
        component_sds: dict[str, dict] = {
            "text_encoder": {},
            "transformer": {},
            "vae": {},
        }
        unmatched_keys = []
        for key, tensor in full_sd.items():
            matched = False
            for prefix, component in prefix_map.items():
                if key.startswith(prefix):
                    stripped = key[len(prefix):]
                    component_sds[component][stripped] = tensor
                    matched = True
                    break
            if not matched:
                unmatched_keys.append(key)

        if unmatched_keys:
            log.debug(
                "Single-file load: %d keys did not match any component prefix "
                "(first 5: %s)", len(unmatched_keys), unmatched_keys[:5]
            )

        # If no prefixed transformer keys were found, check whether the
        # unmatched keys look like unprefixed Z-Image transformer weights.
        # Z-Image single-file checkpoints often store the transformer
        # without a "transformer." prefix — the keys start directly with
        # layer/embedder names like 'layers', 't_embedder', 'cap_embedder',
        # 'x_pad_token', 'noise_refiner', 'context_refiner', etc.
        _ZIMAGE_TRANSFORMER_PREFIXES = {
            "layers", "t_embedder", "x_embedder", "all_x_embedder",
            "cap_embedder", "cap_pad_token", "x_pad_token",
            "noise_refiner", "context_refiner", "all_final_layer",
            "final_layer", "pos_embed", "adaln", "norm_out",
        }
        if not component_sds["transformer"] and unmatched_keys:
            top_level = {k.split(".")[0] for k in unmatched_keys}
            if top_level & _ZIMAGE_TRANSFORMER_PREFIXES:
                log.info(
                    "Detected unprefixed Z-Image transformer keys (%s); "
                    "treating all unmatched keys as transformer weights.",
                    sorted(top_level & _ZIMAGE_TRANSFORMER_PREFIXES),
                )
                for key in unmatched_keys:
                    component_sds["transformer"][key] = full_sd[key]
                unmatched_keys = []

        # Sanity check: we need at least a transformer
        if not component_sds["transformer"]:
            raise RuntimeError(
                f"No transformer/unet keys found in '{model_path}'. "
                f"Key prefixes found: {sorted(set(k.split('.')[0] for k in full_sd))}"
            )

        te_kwargs = self._get_te_load_kwargs()

        # --- Text encoder (Qwen3) ---
        if component_sds["text_encoder"]:
            log.info("Loading Qwen3 text encoder from single-file checkpoint "
                     "(%d parameters)", len(component_sds["text_encoder"]))
            # Try to infer config from the state dict shape
            try:
                te_config = self._infer_qwen3_config(component_sds["text_encoder"])
                from transformers import Qwen3Config, Qwen3Model
                config_obj = Qwen3Config(**te_config)
                self.text_encoder = Qwen3Model(config_obj)
                self.text_encoder.load_state_dict(component_sds["text_encoder"], strict=False)
                self.text_encoder = self.text_encoder.to(dtype=self.dtype)
            except Exception as e:
                log.warning("Could not load text encoder from checkpoint: %s. "
                            "Will attempt to load from HuggingFace.", e)
                self.text_encoder = None
        else:
            log.info("No text_encoder keys in checkpoint; will load from HuggingFace.")
            self.text_encoder = None

        # If text encoder wasn't in the checkpoint or failed to load,
        # try loading from the official Z-Image repo or a standalone Qwen3 repo.
        if self.text_encoder is None:
            te_candidates = [
                (self._HF_BASE_REPO, "text_encoder"),
                ("Qwen/Qwen3-4B", None),
            ]
            for repo, subfolder in te_candidates:
                try:
                    sf_kwargs = {"subfolder": subfolder} if subfolder else {}
                    self.text_encoder = AutoModelForCausalLM.from_pretrained(
                        repo, torch_dtype=self.dtype,
                        trust_remote_code=True, **te_kwargs, **sf_kwargs,
                    )
                    log.info("Loaded Qwen3 text encoder from %s", repo)
                    break
                except Exception:
                    continue
            if self.text_encoder is None:
                log.warning("Could not load text encoder from HuggingFace. "
                            "Training will fail unless a text encoder is provided.")

        # --- Tokenizer: can't be extracted from a checkpoint, load from HF ---
        # Z-Image uses Qwen3-4B; try it first, then fall back to other sizes
        # and the official Z-Image repo (which bundles the tokenizer).
        tokenizer_candidates = [
            "Qwen/Qwen3-4B",
            self._HF_BASE_REPO,
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-0.6B",
        ]
        if self.tokenizer is None:
            for candidate in tokenizer_candidates:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate, trust_remote_code=True,
                    )
                    log.info("Loaded tokenizer from %s", candidate)
                    break
                except Exception:
                    continue
            if self.tokenizer is None:
                raise RuntimeError(
                    "Could not load Qwen3 tokenizer. Please ensure you have "
                    "internet access or specify a diffusers-format model directory."
                )

        # --- VAE ---
        if component_sds["vae"]:
            log.info("Loading VAE from single-file checkpoint (%d parameters)",
                     len(component_sds["vae"]))
            try:
                vae_config = self._infer_vae_config(component_sds["vae"])
                self.vae = AutoencoderKL(**vae_config)
                self.vae.load_state_dict(component_sds["vae"], strict=False)
                self.vae = self.vae.to(dtype=self.vae_dtype)
            except Exception as e:
                log.warning("Could not load VAE from checkpoint: %s", e)
                self.vae = None
        else:
            log.info("No vae keys in checkpoint; will load VAE from HuggingFace.")
            self.vae = None

        # If VAE wasn't in the checkpoint, load from the official Z-Image repo.
        if self.vae is None:
            try:
                self.vae = AutoencoderKL.from_pretrained(
                    self._HF_BASE_REPO, subfolder="vae",
                    torch_dtype=self.dtype,
                )
                log.info("Loaded VAE from %s", self._HF_BASE_REPO)
            except Exception as e:
                log.warning("Could not load VAE from %s: %s", self._HF_BASE_REPO, e)

        # --- Transformer ---
        log.info("Loading transformer from single-file checkpoint (%d parameters)",
                 len(component_sds["transformer"]))
        self.unet = self._load_transformer_weights(component_sds["transformer"])

        # --- Scheduler ---
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()
        log.info("Manual single-file loading complete for Z-Image model")

    def _load_transformer_weights(self, state_dict: dict):
        """Load Z-Image transformer weights, trying multiple model classes.

        Attempts in order:
        1. Load transformer config from the official HF Z-Image repo and
           instantiate + load weights (handles unprefixed checkpoint keys).
        2. Diffusers Transformer2DModel with inferred config.
        3. Direct state dict loading as final fallback.
        """
        errors = []

        # Strategy 1: Load config from official Z-Image HF repo, instantiate,
        # then load the state dict from the checkpoint.
        for repo in (self._HF_BASE_REPO, self._HF_TURBO_REPO):
            try:
                from diffusers.models import Transformer2DModel
                model = Transformer2DModel.from_pretrained(
                    repo, subfolder="transformer",
                    torch_dtype=self.dtype, trust_remote_code=True,
                    low_cpu_mem_usage=False,  # need full model to load_state_dict
                )
                result = model.load_state_dict(state_dict, strict=False)
                n_missing = len(result.missing_keys)
                n_unexpected = len(result.unexpected_keys)
                if n_missing > 0:
                    log.warning(
                        "Transformer loaded from %s config: %d missing keys, "
                        "%d unexpected keys", repo, n_missing, n_unexpected,
                    )
                else:
                    log.info("Transformer loaded via %s config (%d unexpected keys)",
                             repo, n_unexpected)
                return model.to(dtype=self.dtype)
            except Exception as e:
                errors.append(f"HF repo {repo}: {e}")
                continue

        # Strategy 2: Try any custom model class that might be registered
        # for Z-Image in newer diffusers versions.
        for cls_name in ("ZImageTransformer2DModel", "HunyuanDiT2DModel"):
            try:
                import diffusers.models
                cls = getattr(diffusers.models, cls_name, None)
                if cls is None:
                    continue
                config = self._infer_transformer_config(state_dict)
                model = cls(**config)
                model.load_state_dict(state_dict, strict=False)
                log.info("Transformer loaded via %s with inferred config", cls_name)
                return model.to(dtype=self.dtype)
            except Exception as e:
                errors.append(f"{cls_name}: {e}")

        # Strategy 3: Generic Transformer2DModel
        try:
            from diffusers.models import Transformer2DModel
            config = self._infer_transformer_config(state_dict)
            model = Transformer2DModel(**config)
            result = model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                log.warning("Transformer has %d missing keys after load",
                            len(result.missing_keys))
            log.info("Transformer loaded via generic Transformer2DModel")
            return model.to(dtype=self.dtype)
        except Exception as e:
            errors.append(f"Transformer2DModel: {e}")

        raise RuntimeError(
            "Could not load transformer from single-file checkpoint. "
            "Tried:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    @staticmethod
    def _infer_qwen3_config(state_dict: dict) -> dict:
        """Infer Qwen3 model config from state dict tensor shapes."""
        config = {}
        # Try to find embedding dimension
        for key, tensor in state_dict.items():
            if "embed_tokens" in key and "weight" in key:
                config["vocab_size"] = tensor.shape[0]
                config["hidden_size"] = tensor.shape[1]
                break
        # Count layers
        layer_indices = set()
        for key in state_dict:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
        if layer_indices:
            config["num_hidden_layers"] = max(layer_indices) + 1
        # Infer attention heads from q_proj
        for key, tensor in state_dict.items():
            if "q_proj" in key and "weight" in key:
                # q_proj weight shape: [num_heads * head_dim, hidden_size]
                hidden = config.get("hidden_size", tensor.shape[1])
                head_dim = 128  # common default for Qwen3
                if tensor.shape[0] % head_dim == 0:
                    config["num_attention_heads"] = tensor.shape[0] // head_dim
                break
        for key, tensor in state_dict.items():
            if "k_proj" in key and "weight" in key:
                head_dim = 128
                if tensor.shape[0] % head_dim == 0:
                    config["num_key_value_heads"] = tensor.shape[0] // head_dim
                break
        # Infer intermediate_size from gate_proj
        for key, tensor in state_dict.items():
            if "gate_proj" in key and "weight" in key:
                config["intermediate_size"] = tensor.shape[0]
                break
        return config

    @staticmethod
    def _infer_vae_config(state_dict: dict) -> dict:
        """Infer AutoencoderKL config from VAE state dict shapes."""
        config = {}
        # Find latent channels from quant_conv or post_quant_conv
        for key, tensor in state_dict.items():
            if "quant_conv" in key and "weight" in key and tensor.dim() == 4:
                config["latent_channels"] = tensor.shape[0] // 2  # quant_conv doubles channels
                break
        # Infer block_out_channels from encoder/decoder conv layers
        for key, tensor in state_dict.items():
            if "encoder.conv_in.weight" in key:
                config["in_channels"] = tensor.shape[1]
                break
        return config

    @staticmethod
    def _infer_transformer_config(state_dict: dict) -> dict:
        """Infer Transformer2DModel config from state dict shapes."""
        config = {}
        # Find sample_size and inner_dim from positional embeddings or projections
        for key, tensor in state_dict.items():
            if "pos_embed" in key and tensor.dim() >= 2:
                config["sample_size"] = int(tensor.shape[1] ** 0.5) if tensor.dim() == 3 else tensor.shape[-1]
                break
        # Count transformer blocks
        block_indices = set()
        for key in state_dict:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p in ("transformer_blocks", "blocks", "layers") and i + 1 < len(parts) and parts[i + 1].isdigit():
                    block_indices.add(int(parts[i + 1]))
        if block_indices:
            config["num_layers"] = max(block_indices) + 1
        return config

    def load_model(self, model_path: str):
        """Load all Z-Image model components from *model_path*.

        For single-file .safetensors/.ckpt:
          1. Try base class helper (download Z-Image pipeline from HF + swap weights)
          2. Fall back to _load_single_file_pipeline (from_single_file attempts)
          3. Fall back to _load_single_file_manual (component-by-component)

        For directories:
          1. Try DiffusionPipeline.from_pretrained
          2. Fall back to manual component loading

        Sets self.tokenizer, text_encoder, unet, vae, and noise_scheduler.
        The VAE is frozen and moved to device immediately; its shift_factor
        and scaling_factor are cached for use in prepare_latents().
        """
        is_single_file = self._is_single_file(model_path)
        te_kwargs = self._get_te_load_kwargs()
        use_quantized_te = bool(te_kwargs)
        pipe = None

        if is_single_file:
            errors = []

            # Strategy 1: Load full Z-Image pipeline from HF and swap
            # fine-tuned transformer weights from the checkpoint.
            # This is the most reliable method because it downloads all
            # components (Qwen3 TE, VAE, scheduler) from the official repo.
            if not use_quantized_te:
                for repo in (self._HF_BASE_REPO, self._HF_TURBO_REPO):
                    try:
                        log.info("Trying base pipeline from %s + weight swap", repo)
                        pipe = self._load_base_and_swap_weights(
                            model_path, repo,
                            trust_remote_code=True,
                        )
                        log.info("Z-Image loaded via base pipeline swap from %s", repo)
                        break
                    except Exception as e:
                        errors.append(f"Base swap ({repo}): {e}")
                        log.debug("Base swap from %s failed: %s", repo, e)

            # Strategy 2: Try from_single_file methods
            if pipe is None and not use_quantized_te:
                try:
                    pipe = self._load_single_file_pipeline(model_path)
                    if pipe is None:
                        # Manual loading was used; components already set on self
                        pass
                except Exception as e:
                    errors.append(f"Pipeline loading: {e}")
                    log.debug("_load_single_file_pipeline failed: %s", e)

            # Strategy 3: Manual single-file loading (supports quantized TE).
            # Strategies 1-2 are skipped when use_quantized_te is True, so
            # this is the only path that handles quantized TE + single file.
            if pipe is None and self.unet is None:
                try:
                    self._load_single_file_manual(model_path)
                    log.info("Z-Image loaded via manual single-file parsing")
                except Exception as e:
                    errors.append(f"Manual single-file: {e}")
                    log.debug("_load_single_file_manual failed: %s", e)

            # If all strategies failed, raise with combined errors
            if pipe is None and self.unet is None:
                raise RuntimeError(
                    f"All loading methods failed for '{model_path}':\n"
                    + "\n".join(f"  - {e}" for e in errors)
                    + "\n\nConvert to diffusers format and point to the "
                    "directory instead."
                )

            if pipe is not None:
                self.pipeline = pipe
                self.tokenizer = getattr(pipe, 'tokenizer', None)
                self.text_encoder = getattr(pipe, 'text_encoder', None)
                self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
                self.vae = getattr(pipe, 'vae', None)
                self.noise_scheduler = getattr(pipe, 'scheduler', None)
        else:
            # Directory path — try DiffusionPipeline first, then manual
            try:
                if use_quantized_te:
                    raise RuntimeError("Skip pipeline path for quantized TE")
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                self.pipeline = pipe
                self.tokenizer = pipe.tokenizer
                self.text_encoder = pipe.text_encoder
                self.unet = getattr(pipe, 'transformer', getattr(pipe, 'unet', None))
                self.vae = pipe.vae
                self.noise_scheduler = pipe.scheduler
            except Exception:
                # Manual component loading for diffusers-format directories
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, subfolder="tokenizer",
                    trust_remote_code=True,
                )
                self.text_encoder = AutoModelForCausalLM.from_pretrained(
                    model_path, subfolder="text_encoder",
                    torch_dtype=self.dtype, trust_remote_code=True,
                    **te_kwargs,
                )
                self.vae = AutoencoderKL.from_pretrained(
                    model_path, subfolder="vae",
                    torch_dtype=self.dtype,
                )
                try:
                    from diffusers.models import Transformer2DModel
                    self.unet = Transformer2DModel.from_pretrained(
                        model_path, subfolder="transformer",
                        torch_dtype=self.dtype, trust_remote_code=True,
                    )
                except Exception:
                    # Fallback: load full pipeline but only extract the
                    # transformer.  Delete the pipeline immediately to free
                    # the duplicate TE + VAE copies (which can waste ~10+ GB
                    # when quantization is enabled for the TE above).
                    from diffusers import DiffusionPipeline
                    pipe = DiffusionPipeline.from_pretrained(
                        model_path, torch_dtype=self.dtype,
                        trust_remote_code=True,
                    )
                    self.unet = getattr(pipe, 'transformer', pipe.unet)
                    del pipe
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    model_path, subfolder="scheduler",
                )
                if self.pipeline is None:
                    log.info("Z-Image loaded in manual mode (sample generation unavailable)")

        if self.vae is not None:
            self.vae.to(self.device, dtype=self.vae_dtype)
            self.vae.requires_grad_(False)

        # Store VAE scaling info
        self._vae_shift_factor = getattr(
            self.vae.config, 'shift_factor', 0.0
        ) if self.vae is not None else 0.0
        self._vae_scaling_factor = getattr(
            self.vae.config, 'scaling_factor', 0.18215
        ) if self.vae is not None else 0.18215

        log.info(f"Loaded Z-Image model from {model_path}")

    def _get_lora_target_modules(self) -> list[str]:
        """Return the list of attention/MLP module names to target with LoRA.

        Covers the joint-attention projections (to_q/k/v/out), MLP
        projections, and adaptive-norm linear layers specific to the
        ZImageTransformer2DModel architecture.
        """
        # ZImageTransformerBlock has:
        #   attention.{to_q, to_k, to_v, to_out.0}
        #   feed_forward.{w1, w2, w3}
        #   adaLN_modulation (adaptive layer norm)
        # Using short names for PEFT substring matching.
        return [
            "to_q", "to_k", "to_v", "to_out.0",
            "w1", "w2", "w3",
            "adaLN_modulation",
        ]

    def _format_caption(self, caption: str) -> str:
        """Apply the Qwen3 chat template to a single caption string.

        Public entry point wrapping the module-level _apply_chat_template().
        Exposed as an instance method so that external callers (e.g. the
        TE caching path in train_dataset.py) can call
        backend._format_caption() and get identical preprocessing to
        what encode_text_batch() uses internally.
        """
        return _apply_chat_template(self.tokenizer, caption)

    def encode_text_batch(self, captions: list[str]) -> tuple:
        """Encode with Qwen3 using chat template.

        Z-Image uses Qwen3ForCausalLM which requires chat-style
        tokenization. The second-to-last hidden state layer is used
        as text conditioning (matching the official pipeline).

        Returns (hidden_states, attention_mask) as padded tensors
        so the output is compatible with the TE caching system. The
        training_step strips padding per-sample before passing to
        the transformer (which expects variable-length sequences).
        """
        # Format all captions through chat template
        texts = [_apply_chat_template(self.tokenizer, c) for c in captions]

        # Batch tokenize all captions at once
        tokens = self.tokenizer(
            texts, padding="max_length",
            max_length=_QWEN3_MAX_LENGTH,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with self._te_no_grad():
            attention_mask = tokens["attention_mask"]
            out = self.text_encoder(
                **tokens,
                output_hidden_states=True,
            )
            # Use second-to-last hidden state (matches official pipeline)
            encoder_hidden = out.hidden_states[-2]
            del out  # Free all LLM hidden states from VRAM

        # Return (hidden_states, attention_mask) — mask is used by
        # training_step to strip padding before the transformer call.
        return (encoder_hidden, attention_mask)

    def prepare_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values into latent space using the frozen VAE.

        Applies Z-Image's custom latent scaling: (latents - shift_factor) * scaling_factor.
        The shift_factor centres the latent distribution, while the scaling_factor
        normalises its variance. Uses channels_last memory format for performance.
        """
        self.vae.eval()
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(
                    device=self.device, dtype=self.vae_dtype,
                    memory_format=torch.channels_last,
                )
            ).latent_dist.sample()
            # Apply Z-Image specific scaling: (latents - shift) * scale
            # This matches the diffusers convention for VAEs with shift_factor.
            latents = (latents - self._vae_shift_factor) * self._vae_scaling_factor
        return latents

    def _get_timestep_shift(self, latents: torch.Tensor) -> float:
        """Compute a resolution-dependent timestep shift factor.

        Higher-resolution images (more patches) are shifted closer to
        max_shift, biasing the noise schedule toward higher noise levels
        where large images benefit from more denoising steps. The shift
        is linearly interpolated between base_shift (0.5) and max_shift
        (1.15) based on the ratio of actual patches to 4096 (the count
        at 1024x1024 with patch_size=2).
        """
        h, w = latents.shape[-2:]
        # Patch size is typically 2 for the latent space
        num_patches = (h // 2) * (w // 2)
        base_shift = 0.5
        max_shift = 1.15
        # Linear interpolation based on number of patches
        shift = base_shift + (max_shift - base_shift) * min(num_patches / 4096, 1.0)
        return shift

    def training_step(
        self, latents: torch.Tensor, te_out: tuple, batch_size: int,
    ) -> torch.Tensor:
        """Z-Image training step with flow matching + dynamic timestep shift.

        This overrides the base flow_training_step because Z-Image needs
        resolution-dependent timestep shifting. All shared features
        (noise_offset, EMA sampling, SpeeD weighting, token weighting,
        autocast, min_snr warning) are included for parity.

        Z-exclusive optimizations (when enabled):
        - Logit-Normal timestep sampling (focuses on informative mid-range)
        - Straight-path velocity weighting (emphasizes where flow changes most)
        """
        config = self.config

        # Adaptive timestep sampling (shared with other flow models)
        if getattr(self, '_timestep_bandit', None) is not None:
            # Z-exclusive: Thompson Sampling bandit (most advanced option)
            u = self._timestep_bandit.sample_timesteps(batch_size)
        elif self._timestep_ema_sampler is not None:
            discrete_ts = self._timestep_ema_sampler.sample_timesteps(batch_size)
            u = discrete_ts.float() / 1000.0
        elif getattr(self, '_logit_normal_sampler', None) is not None:
            # Z-exclusive: Logit-Normal sampling for straight-path training
            u = self._logit_normal_sampler.sample(batch_size)
        else:
            # Falls through to _sample_flow_timesteps which handles SpeeD
            # (timestep_sampling="speed" or speed_asymmetric), logit-normal,
            # sigmoid, or uniform sampling via config
            u = self._sample_flow_timesteps(batch_size)

        # Apply dynamic timestep shift (Z-Image specific)
        shift = self._get_timestep_shift(latents)
        t = shift * u / (1 + (shift - 1) * u)

        noise = torch.randn_like(latents)
        # Apply noise_offset (consistent with base flow_training_step)
        if config.noise_offset > 0:
            noise += config.noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1,
                device=latents.device, dtype=latents.dtype,
            )
        noisy_latents = self._flow_interpolate(latents, noise, t)

        # Build per-sample caption features for the transformer.
        # ZImageTransformer2DModel expects cap_feats as a list of 2D
        # tensors [seq_len_i, dim] with padding REMOVED (variable length).
        #
        # te_out format:
        #   Live:   (hidden_states [B, seq, dim], attention_mask [B, seq])
        #   Cached: (hidden_states [B, seq, dim], None_or_mask)
        #
        # When an attention_mask is available (live encoding or if cache
        # stored it), use it to strip padding. Otherwise pass as-is
        # (the transformer handles padded sequences, just less efficiently).
        cap_feats_raw = te_out[0]
        attn_mask = te_out[1] if len(te_out) > 1 else None

        # Normalise to 3D [B, seq, dim]
        if isinstance(cap_feats_raw, torch.Tensor):
            while cap_feats_raw.dim() > 3:
                cap_feats_raw = cap_feats_raw.squeeze(1)
            if cap_feats_raw.dim() == 2:
                cap_feats_raw = cap_feats_raw.unsqueeze(0)

        # Strip padding using attention mask (produces variable-length list)
        if attn_mask is not None and isinstance(attn_mask, torch.Tensor):
            cap_feats = []
            for i in range(cap_feats_raw.shape[0]):
                mask = attn_mask[i].bool()
                cap_feats.append(cap_feats_raw[i][mask])
        else:
            # No mask available (cached without mask) — pass padded
            cap_feats = list(cap_feats_raw.unbind(dim=0))

        # Forward pass with autocast for mixed precision speed
        # ZImageTransformer2DModel.forward() expects:
        #   x: list of 4-D tensors [C, 1, H, W] (frame dim required)
        #   t: timestep tensor [batch_size]
        #   cap_feats: list of 2-D tensors [seq_len, dim] (no padding)
        _act = autocast_device_type()
        with torch.autocast(device_type=_act, dtype=self.dtype, enabled=self.device.type != "cpu"):
            # Add frame dimension: [B, C, H, W] -> [B, C, 1, H, W] -> unbind
            x_list = list(noisy_latents.unsqueeze(2).unbind(dim=0))
            noise_pred = self.unet(
                x=x_list,
                t=t,
                cap_feats=cap_feats,
                return_dict=False,
            )[0]
            # Reconstruct batch tensor from list of per-sample outputs
            if isinstance(noise_pred, (list, tuple)):
                noise_pred = torch.stack(noise_pred)
            # Remove frame dimension: [B, C, 1, H, W] -> [B, C, H, W]
            if noise_pred.dim() == 5:
                noise_pred = noise_pred.squeeze(2)

        loss = self._compute_flow_loss(noise_pred, noise, latents)

        # Z-exclusive: straight-path velocity weighting
        # Up-weights mid-range timesteps where the flow velocity changes most
        if getattr(self, '_velocity_weighting', False):
            vel_weights = 1.0 / (t * (1 - t) + 1e-5)
            vel_weights = vel_weights / vel_weights.mean()  # Normalize to mean=1
            loss = loss * vel_weights.view(-1, *([1] * (loss.dim() - 1)))

        if config.debiased_estimation:
            # Clamp denominator to prevent extreme weights when t → 1.0
            weight = 1.0 / torch.clamp(1.0 - t.float() + 1e-6, min=0.01)
            loss = loss * weight

        # min_snr_gamma: not applicable to flow matching (requires alphas_cumprod)
        if config.min_snr_gamma > 0 and not getattr(self, '_warned_snr_flow', False):
            log.warning("min_snr_gamma is not supported for flow matching models "
                        "(Z-Image, Flux, SD3, etc.) and will be ignored.")
            self._warned_snr_flow = True

        # SpeeD change-aware loss weighting (CVPR 2025)
        if config.speed_change_aware and self._speed_sampler is not None:
            timesteps_int = (t * 1000).long()
            speed_weights = self._speed_sampler.compute_weights(timesteps_int, loss.detach())
            loss = loss * speed_weights

        # Per-timestep EMA: update tracker and apply adaptive weighting
        if self._timestep_ema_sampler is not None:
            per_sample_loss = loss.detach()
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.flatten(1).mean(1)
            timesteps = (t * 1000).long()
            self._timestep_ema_sampler.update(timesteps, per_sample_loss)
            ema_weights = self._timestep_ema_sampler.compute_loss_weights(timesteps)
            loss = loss * ema_weights.view(-1, *([1] * (loss.dim() - 1)))

        # Token weighting (consistent with base flow_training_step)
        if getattr(self, '_token_weight_mask', None) is not None:
            mask = self._token_weight_mask
            if mask.device != loss.device:
                mask = mask.to(loss.device)
            if mask.dim() >= 1 and mask.shape[0] == loss.shape[0]:
                non_zero = mask > 0
                sample_weight = mask.sum(dim=-1) / non_zero.sum(dim=-1).clamp(min=1)
                while sample_weight.dim() < loss.dim():
                    sample_weight = sample_weight.unsqueeze(-1)
                loss = loss * sample_weight
            self._token_weight_mask = None

        # Adaptive per-sample weights (set by trainer's tag weighter)
        if getattr(self, '_adaptive_sample_weights', None) is not None:
            weights = self._adaptive_sample_weights
            if loss.dim() > 0 and loss.shape[0] == weights.shape[0]:
                loss = loss * weights
            self._adaptive_sample_weights = None

        # Store per-sample loss for adaptive tag weighting (before .mean())
        self._per_sample_loss = loss.detach()

        # Z-exclusive: update timestep bandit with observed losses
        if getattr(self, '_timestep_bandit', None) is not None:
            per_sample = loss.detach()
            if per_sample.dim() > 1:
                per_sample = per_sample.flatten(1).mean(1)
            self._timestep_bandit.update(t.detach(), per_sample)

        return loss.mean()
