"""Config build/apply/save/load methods for TrainingTab.

Extracted from training_tab.py to keep serialisation logic separate
from the UI layout and training control code.
"""

import json
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog

from dataset_sorter.constants import MODEL_TYPE_KEYS, VRAM_TIERS
from dataset_sorter.models import TrainingConfig
from dataset_sorter.ui.toast import show_toast


class TrainingConfigIOMixin:
    """Mixin providing build_config / apply_config / save / load for TrainingTab."""

    def build_config(self) -> TrainingConfig:
        """Collect values from every UI widget and return a populated TrainingConfig.

        Reads model, network, optimizer, dataset, advanced, checkpointing,
        sampling, and RLHF settings from their respective spin boxes, combo
        boxes, and check boxes.
        """
        config = TrainingConfig()

        # Model
        idx = self.train_model_combo.currentIndex()
        config.model_type = MODEL_TYPE_KEYS[idx] if 0 <= idx < len(MODEL_TYPE_KEYS) else "sdxl_lora"
        vidx = self.train_vram_combo.currentIndex()
        config.vram_gb = VRAM_TIERS[vidx] if 0 <= vidx < len(VRAM_TIERS) else 24
        config.resolution = self.resolution_spin.value()
        config.clip_skip = self.clip_skip_spin.value()

        # Network
        config.network_type = self.train_network_combo.currentData() or "lora"
        config.lora_rank = self.rank_spin.value()
        config.lora_alpha = self.alpha_spin.value()
        config.conv_rank = self.conv_rank_spin.value()
        config.conv_alpha = self.conv_rank_spin.value() // 2 if self.conv_rank_spin.value() > 0 else 0
        config.use_dora = self.dora_check.isChecked()
        config.use_rslora = self.rslora_check.isChecked()
        config.lora_init = self.lora_init_combo.currentData() or "default"

        # EMA
        config.use_ema = self.ema_check.isChecked()
        config.ema_cpu_offload = self.ema_cpu_check.isChecked()
        config.ema_decay = self.ema_decay_spin.value()

        # Optimizer
        config.optimizer = self.train_optimizer_combo.currentData() or "Adafactor"
        config.learning_rate = self.lr_spin.value()
        config.text_encoder_lr = self.te_lr_spin.value()
        config.weight_decay = self.wd_spin.value()
        config.max_grad_norm = self.grad_norm_spin.value()
        config.lr_scheduler = self.scheduler_combo.currentData() or "cosine"
        config.warmup_steps = self.warmup_spin.value()

        # Batch
        config.batch_size = self.batch_spin.value()
        config.gradient_accumulation = self.grad_accum_spin.value()
        config.effective_batch_size = config.batch_size * config.gradient_accumulation
        config.epochs = self.epochs_spin.value()
        config.max_train_steps = self.max_steps_spin.value()

        # Text encoder
        config.train_text_encoder = self.train_te_check.isChecked()
        config.train_text_encoder_2 = self.train_te2_check.isChecked()

        # Dataset
        config.cache_latents = self.cache_latents_check.isChecked()
        config.cache_latents_to_disk = self.cache_disk_check.isChecked()
        config.cache_text_encoder = self.cache_te_check.isChecked()
        config.fast_image_decoder = self.fast_decoder_check.isChecked()
        config.safetensors_cache = self.safetensors_cache_check.isChecked()
        config.fp16_latent_cache = self.fp16_cache_check.isChecked()
        config.cache_to_ram_disk = self.ram_disk_check.isChecked()
        config.lmdb_cache = self.lmdb_cache_check.isChecked()
        config.tag_shuffle = self.tag_shuffle_check.isChecked()
        config.keep_first_n_tags = self.keep_tags_spin.value()
        config.caption_dropout_rate = self.caption_dropout_spin.value()
        config.random_crop = self.random_crop_check.isChecked()
        config.flip_augmentation = self.flip_aug_check.isChecked()
        config.enable_bucket = self.bucket_check.isChecked()
        config.bucket_reso_steps = self.bucket_step_spin.value()
        config.resolution_min = self.bucket_min_spin.value()
        config.resolution_max = self.bucket_max_spin.value()

        # Advanced
        config.noise_offset = self.noise_offset_spin.value()
        config.min_snr_gamma = self.snr_gamma_spin.value()
        config.ip_noise_gamma = self.ip_noise_spin.value()
        config.debiased_estimation = self.debiased_check.isChecked()
        config.speed_asymmetric = self.speed_asymmetric_check.isChecked()
        config.speed_change_aware = self.speed_change_check.isChecked()
        config.mebp_enabled = self.mebp_check.isChecked()
        config.approx_vjp = self.vjp_check.isChecked()
        config.async_dataload = self.async_data_check.isChecked()
        config.cuda_graph_training = self.cuda_graph_check.isChecked()
        config.async_optimizer_step = self.async_opt_check.isChecked()
        config.curriculum_learning = self.curriculum_check.isChecked()
        config.curriculum_temperature = self.curriculum_temp_spin.value()
        config.curriculum_warmup_epochs = self.curriculum_warmup_spin.value()
        config.timestep_ema_sampling = self.timestep_ema_check.isChecked()
        config.timestep_ema_skip_threshold = self.timestep_ema_threshold_spin.value()
        config.timestep_ema_num_buckets = self.timestep_ema_buckets_spin.value()
        config.concept_probe_enabled = self.concept_probe_check.isChecked()
        config.concept_probe_steps = self.concept_probe_steps_spin.value()
        config.concept_probe_images = self.concept_probe_images_spin.value()
        config.concept_probe_threshold = self.concept_probe_threshold_spin.value()
        config.adaptive_tag_weighting = self.adaptive_tag_check.isChecked()
        config.adaptive_tag_warmup = self.adaptive_tag_warmup_spin.value()
        config.adaptive_tag_rate = self.adaptive_tag_rate_spin.value()
        config.adaptive_tag_max_weight = self.adaptive_tag_max_spin.value()
        config.attention_rebalancing = self.attention_rebalance_check.isChecked()
        config.attention_rebalance_threshold = self.attention_rebalance_threshold_spin.value()
        config.attention_rebalance_boost = self.attention_rebalance_boost_spin.value()
        config.timestep_sampling = self.timestep_combo.currentData() or "uniform"
        config.model_prediction_type = self.prediction_combo.currentData() or "epsilon"
        config.gradient_checkpointing = self.grad_ckpt_check.isChecked()
        config.fused_backward_pass = self.fused_backward_check.isChecked()
        config.fp8_base_model = self.fp8_check.isChecked()
        config.quantize_text_encoder = self.te_quant_combo.currentData() or "none"
        config.cudnn_benchmark = self.cudnn_check.isChecked()
        config.mixed_precision = self.precision_combo.currentText()

        attn = self.attention_combo.currentData() or "sdpa"
        config.sdpa = attn == "sdpa"
        config.xformers = attn == "xformers"
        config.flash_attention = attn == "flash_attention"

        # Checkpointing
        config.save_every_n_steps = self.save_steps_spin.value()
        config.save_every_n_epochs = self.save_epochs_spin.value()
        config.save_last_n_checkpoints = self.keep_ckpt_spin.value()
        config.save_precision = self.save_prec_combo.currentData() or "bf16"

        # Sampling
        config.sample_every_n_steps = self.sample_steps_spin.value()
        config.sample_sampler = self.sampler_combo.currentData() or "euler_a"
        config.sample_steps = self.sample_inf_steps_spin.value()
        config.sample_cfg_scale = self.cfg_spin.value()
        config.sample_seed = self.seed_spin.value()
        config.num_sample_images = self.num_samples_spin.value()

        # Prompts
        prompts_text = self.prompts_input.toPlainText().strip()
        if prompts_text:
            config.sample_prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]

        # Token Weighting
        config.token_weighting_enabled = self.token_weight_check.isChecked()
        config.token_default_weight = self.token_default_weight_spin.value()
        config.token_trigger_weight = self.token_trigger_weight_spin.value()

        # Attention Debug
        config.attention_debug_enabled = self.attn_debug_check.isChecked()
        config.attention_debug_every_n_steps = self.attn_debug_steps_spin.value()
        config.attention_debug_top_k = self.attn_debug_topk_spin.value()

        # Smart Resume
        config.smart_resume = self.smart_resume_check.isChecked()
        config.smart_resume_auto_apply = self.smart_resume_auto_apply_check.isChecked()

        # RLHF
        config.rlhf_enabled = self.rlhf_enable_check.isChecked()
        config.rlhf_pairs_per_round = self.rlhf_pairs_spin.value()
        config.rlhf_collect_every_n_steps = self.rlhf_interval_spin.value()
        config.dpo_beta = self.rlhf_dpo_beta_spin.value()
        config.dpo_loss_type = self.rlhf_loss_combo.currentData() or "sigmoid"

        return config

    def apply_config(self, config: TrainingConfig):
        """Populate every UI widget from the given TrainingConfig.

        Sets combo box indices, spin box values, and check box states to
        match the supplied config. Unknown combo values are silently skipped.
        """
        # Model
        try:
            idx = MODEL_TYPE_KEYS.index(config.model_type)
            self.train_model_combo.setCurrentIndex(idx)
        except ValueError:
            pass
        try:
            vidx = VRAM_TIERS.index(config.vram_gb)
            self.train_vram_combo.setCurrentIndex(vidx)
        except ValueError:
            pass
        self.resolution_spin.setValue(config.resolution)
        self.clip_skip_spin.setValue(config.clip_skip)

        # Network
        for i in range(self.train_network_combo.count()):
            if self.train_network_combo.itemData(i) == config.network_type:
                self.train_network_combo.setCurrentIndex(i)
                break
        self.rank_spin.setValue(config.lora_rank)
        self.alpha_spin.setValue(config.lora_alpha)
        self.conv_rank_spin.setValue(config.conv_rank)
        self.dora_check.setChecked(config.use_dora)
        self.rslora_check.setChecked(config.use_rslora)
        for i in range(self.lora_init_combo.count()):
            if self.lora_init_combo.itemData(i) == config.lora_init:
                self.lora_init_combo.setCurrentIndex(i)
                break

        # EMA
        self.ema_check.setChecked(config.use_ema)
        self.ema_cpu_check.setChecked(config.ema_cpu_offload)
        self.ema_decay_spin.setValue(config.ema_decay)

        # Optimizer
        for i in range(self.train_optimizer_combo.count()):
            if self.train_optimizer_combo.itemData(i) == config.optimizer:
                self.train_optimizer_combo.setCurrentIndex(i)
                break
        self.lr_spin.setValue(config.learning_rate)
        self.te_lr_spin.setValue(config.text_encoder_lr)
        self.wd_spin.setValue(config.weight_decay)
        self.grad_norm_spin.setValue(config.max_grad_norm)
        for i in range(self.scheduler_combo.count()):
            if self.scheduler_combo.itemData(i) == config.lr_scheduler:
                self.scheduler_combo.setCurrentIndex(i)
                break
        self.warmup_spin.setValue(config.warmup_steps)

        # Batch
        self.batch_spin.setValue(config.batch_size)
        self.grad_accum_spin.setValue(config.gradient_accumulation)
        self.epochs_spin.setValue(config.epochs)
        self.max_steps_spin.setValue(config.max_train_steps)

        # TE
        self.train_te_check.setChecked(config.train_text_encoder)
        self.train_te2_check.setChecked(config.train_text_encoder_2)

        # Dataset
        self.cache_latents_check.setChecked(config.cache_latents)
        self.cache_disk_check.setChecked(config.cache_latents_to_disk)
        self.cache_te_check.setChecked(config.cache_text_encoder)
        self.fast_decoder_check.setChecked(config.fast_image_decoder)
        self.safetensors_cache_check.setChecked(config.safetensors_cache)
        self.fp16_cache_check.setChecked(config.fp16_latent_cache)
        self.ram_disk_check.setChecked(config.cache_to_ram_disk)
        self.lmdb_cache_check.setChecked(config.lmdb_cache)
        self.tag_shuffle_check.setChecked(config.tag_shuffle)
        self.keep_tags_spin.setValue(config.keep_first_n_tags)
        self.caption_dropout_spin.setValue(config.caption_dropout_rate)
        self.random_crop_check.setChecked(config.random_crop)
        self.flip_aug_check.setChecked(config.flip_augmentation)
        self.bucket_check.setChecked(config.enable_bucket)
        self.bucket_step_spin.setValue(config.bucket_reso_steps)
        self.bucket_min_spin.setValue(config.resolution_min)
        self.bucket_max_spin.setValue(config.resolution_max)

        # Advanced
        self.noise_offset_spin.setValue(config.noise_offset)
        self.snr_gamma_spin.setValue(config.min_snr_gamma)
        self.ip_noise_spin.setValue(config.ip_noise_gamma)
        self.debiased_check.setChecked(config.debiased_estimation)
        self.speed_asymmetric_check.setChecked(config.speed_asymmetric)
        self.speed_change_check.setChecked(config.speed_change_aware)
        self.mebp_check.setChecked(config.mebp_enabled)
        self.vjp_check.setChecked(config.approx_vjp)
        self.async_data_check.setChecked(config.async_dataload)
        self.cuda_graph_check.setChecked(config.cuda_graph_training)
        self.async_opt_check.setChecked(config.async_optimizer_step)
        self.curriculum_check.setChecked(config.curriculum_learning)
        self.curriculum_temp_spin.setValue(config.curriculum_temperature)
        self.curriculum_warmup_spin.setValue(config.curriculum_warmup_epochs)
        self.timestep_ema_check.setChecked(config.timestep_ema_sampling)
        self.timestep_ema_threshold_spin.setValue(config.timestep_ema_skip_threshold)
        self.timestep_ema_buckets_spin.setValue(config.timestep_ema_num_buckets)
        self.concept_probe_check.setChecked(config.concept_probe_enabled)
        self.concept_probe_steps_spin.setValue(config.concept_probe_steps)
        self.concept_probe_images_spin.setValue(config.concept_probe_images)
        self.concept_probe_threshold_spin.setValue(config.concept_probe_threshold)
        self.adaptive_tag_check.setChecked(config.adaptive_tag_weighting)
        self.adaptive_tag_warmup_spin.setValue(config.adaptive_tag_warmup)
        self.adaptive_tag_rate_spin.setValue(config.adaptive_tag_rate)
        self.adaptive_tag_max_spin.setValue(config.adaptive_tag_max_weight)
        self.attention_rebalance_check.setChecked(config.attention_rebalancing)
        self.attention_rebalance_threshold_spin.setValue(config.attention_rebalance_threshold)
        self.attention_rebalance_boost_spin.setValue(config.attention_rebalance_boost)
        for i in range(self.timestep_combo.count()):
            if self.timestep_combo.itemData(i) == config.timestep_sampling:
                self.timestep_combo.setCurrentIndex(i)
                break
        for i in range(self.prediction_combo.count()):
            if self.prediction_combo.itemData(i) == config.model_prediction_type:
                self.prediction_combo.setCurrentIndex(i)
                break
        self.grad_ckpt_check.setChecked(config.gradient_checkpointing)
        self.fused_backward_check.setChecked(config.fused_backward_pass)
        self.fp8_check.setChecked(config.fp8_base_model)
        for i in range(self.te_quant_combo.count()):
            if self.te_quant_combo.itemData(i) == config.quantize_text_encoder:
                self.te_quant_combo.setCurrentIndex(i)
                break
        self.cudnn_check.setChecked(config.cudnn_benchmark)
        pidx = ["bf16", "fp16", "fp32"].index(config.mixed_precision) if config.mixed_precision in ["bf16", "fp16", "fp32"] else 0
        self.precision_combo.setCurrentIndex(pidx)

        if config.sdpa:
            self.attention_combo.setCurrentIndex(0)
        elif config.xformers:
            self.attention_combo.setCurrentIndex(1)
        elif config.flash_attention:
            self.attention_combo.setCurrentIndex(2)

        # Checkpointing
        self.save_steps_spin.setValue(config.save_every_n_steps)
        self.save_epochs_spin.setValue(config.save_every_n_epochs)
        self.keep_ckpt_spin.setValue(config.save_last_n_checkpoints)
        for i in range(self.save_prec_combo.count()):
            if self.save_prec_combo.itemData(i) == config.save_precision:
                self.save_prec_combo.setCurrentIndex(i)
                break

        # Sampling
        self.sample_steps_spin.setValue(config.sample_every_n_steps)
        for i in range(self.sampler_combo.count()):
            if self.sampler_combo.itemData(i) == config.sample_sampler:
                self.sampler_combo.setCurrentIndex(i)
                break
        self.sample_inf_steps_spin.setValue(config.sample_steps)
        self.cfg_spin.setValue(config.sample_cfg_scale)
        self.seed_spin.setValue(config.sample_seed)
        self.num_samples_spin.setValue(config.num_sample_images)

        if config.sample_prompts:
            self.prompts_input.setPlainText("\n".join(config.sample_prompts))

        # Token Weighting
        self.token_weight_check.setChecked(config.token_weighting_enabled)
        self.token_default_weight_spin.setValue(config.token_default_weight)
        self.token_trigger_weight_spin.setValue(config.token_trigger_weight)

        # Attention Debug
        self.attn_debug_check.setChecked(config.attention_debug_enabled)
        self.attn_debug_steps_spin.setValue(config.attention_debug_every_n_steps)
        self.attn_debug_topk_spin.setValue(config.attention_debug_top_k)

        # Smart Resume
        self.smart_resume_check.setChecked(config.smart_resume)
        self.smart_resume_auto_apply_check.setChecked(config.smart_resume_auto_apply)

        # RLHF
        self.rlhf_enable_check.setChecked(config.rlhf_enabled)
        self.rlhf_pairs_spin.setValue(config.rlhf_pairs_per_round)
        self.rlhf_interval_spin.setValue(config.rlhf_collect_every_n_steps)
        self.rlhf_dpo_beta_spin.setValue(config.dpo_beta)
        for i in range(self.rlhf_loss_combo.count()):
            if self.rlhf_loss_combo.itemData(i) == config.dpo_loss_type:
                self.rlhf_loss_combo.setCurrentIndex(i)
                break

    def _save_training_config(self):
        """Open a Save dialog and write the current training config to a JSON file.

        Builds the config from UI controls, serialises it via dataclasses.asdict,
        and writes pretty-printed JSON. Logs success or failure to the training log.
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Config", "training_config.json", "JSON (*.json)",
        )
        if not path:
            return
        config = self.build_config()
        data = asdict(config)
        try:
            Path(path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            self._log(f"Training config saved: {path}")
            show_toast(self, "Training config saved", "success")
        except OSError as e:
            self._log(f"ERROR saving config: {e}")
            show_toast(self, "Config save failed", "error")

    def _load_training_config(self):
        """Open a Load dialog and restore training config from a JSON file.

        Parses the JSON into a TrainingConfig, coercing value types to match
        dataclass field types. Null values and type mismatches are skipped to
        preserve safe defaults. Applies the result to all UI widgets.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Config", "", "JSON (*.json)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self._log(f"ERROR loading config: {e}")
            return

        config = TrainingConfig()
        for key, value in data.items():
            if hasattr(config, key):
                # Skip JSON null values — keep the dataclass default instead
                # of corrupting numeric/bool fields with None
                if value is None:
                    continue
                try:
                    # Handle list fields
                    current = getattr(config, key)
                    if isinstance(current, list) and isinstance(value, list):
                        setattr(config, key, value)
                    elif isinstance(current, bool):
                        # bool must be checked before int (bool is subclass of int).
                        # Prevent bool("false") = True corruption from JSON strings.
                        if isinstance(value, bool):
                            setattr(config, key, value)
                        elif isinstance(value, (int, float)):
                            setattr(config, key, bool(value))
                        # else: skip string/other types for bool fields
                    else:
                        setattr(config, key, type(current)(value))
                except (ValueError, TypeError):
                    pass  # Keep default rather than setting an incompatible value
        self.apply_config(config)
        self._log(f"Training config loaded: {path}")
        show_toast(self, "Training config loaded", "success")
