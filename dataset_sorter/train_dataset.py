"""Training dataset with latent caching, TE caching, tag shuffle.

Handles:
- Pre-computing and caching VAE latents (RAM or disk)
- Pre-computing and caching text encoder outputs (RAM or disk)
- Random tag shuffling with keep_first_n
- Caption dropout
- Multi-aspect ratio bucketing
- Center/random crop
- Horizontal flip augmentation

Speed optimizations:
- Fast image decoder (turbojpeg → cv2 → PIL fallback)
- Safetensors cache format (2-5x faster I/O)
- FP16 latent compression (50% RAM/disk savings)
- Batched VAE encoding (amortize GPU overhead)
- Deduplication-aware caching (skip duplicate images)
- Parallel disk cache writes
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger(__name__)


class CachedTrainDataset(Dataset):
    """Training dataset with latent/TE caching for efficient multi-epoch training.

    Cache strategy:
    - First epoch: compute latents + TE outputs, store in RAM (or disk)
    - Subsequent epochs: load from cache, skip VAE/TE forward pass
    - Saves ~60-70% of per-epoch training time
    """

    def __init__(
        self,
        image_paths: list[Path],
        captions: list[str],
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = False,
        tag_shuffle: bool = True,
        keep_first_n_tags: int = 1,
        caption_dropout_rate: float = 0.0,
        cache_dir: Optional[Path] = None,
        bucket_assignments: Optional[list[tuple[int, int]]] = None,
    ):
        self.image_paths = image_paths
        self.captions = captions
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tag_shuffle = tag_shuffle
        self.keep_first_n_tags = keep_first_n_tags
        self.caption_dropout_rate = caption_dropout_rate
        self.cache_dir = cache_dir

        # Per-image bucket resolutions (None = all use self.resolution)
        self._bucket_assignments = bucket_assignments

        # Caches (populated by cache_latents / cache_text_encoder)
        self._latent_cache: dict[int, torch.Tensor] = {}
        self._te_cache: dict[int, tuple] = {}
        self._token_id_cache: dict[int, tuple] = {}  # Pre-tokenized caption IDs
        self._latents_cached = False
        self._te_cached = False
        self._tokens_cached = False

        # Image transforms — BILINEAR is 2-3x faster than LANCZOS with negligible
        # quality difference for training (latents are compressed by 8x anyway)
        self._transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _get_transforms_for_resolution(self, width: int, height: int):
        """Build transforms for a specific (w, h) bucket resolution."""
        crop_cls = transforms.CenterCrop if self.center_crop else transforms.RandomCrop
        return transforms.Compose([
            transforms.Resize(
                min(width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            crop_cls((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, idx):
        result = {}

        # Determine target resolution for this image
        if self._bucket_assignments is not None and idx < len(self._bucket_assignments):
            target_w, target_h = self._bucket_assignments[idx]
            result["bucket_width"] = target_w
            result["bucket_height"] = target_h
        else:
            target_w, target_h = self.resolution, self.resolution

        # --- Image / Latent ---
        if self._latents_cached and idx in self._latent_cache:
            latent = self._latent_cache[idx]
            # Apply random horizontal flip to cached latents (flip the width dim)
            if self.random_flip and random.random() < 0.5:
                latent = torch.flip(latent, [-1])
            result["latent"] = latent
        else:
            try:
                from dataset_sorter.io_speed import fast_decode_image
                img = fast_decode_image(self.image_paths[idx])
            except Exception as e:
                log.warning(f"Failed to open {self.image_paths[idx]}: {e}, using blank image")
                img = Image.new("RGB", (target_w, target_h))
            if self.random_flip and random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Use bucket-specific transforms when bucketing is active
            if self._bucket_assignments is not None:
                t = self._get_transforms_for_resolution(target_w, target_h)
                pixel_values = t(img)
            else:
                pixel_values = self._transforms(img)
            result["pixel_values"] = pixel_values

        # --- Caption with tag shuffle ---
        caption = self._process_caption(self.captions[idx])
        result["caption"] = caption
        # Keep raw caption with weight markers for token weighting
        result["raw_caption"] = self.captions[idx]

        # --- Text encoder cache (skip if caption was modified by shuffle/dropout) ---
        if self._te_cached and idx in self._te_cache and caption == self.captions[idx]:
            result["te_cache"] = self._te_cache[idx]

        # --- Pre-tokenized caption IDs (skip tokenizer calls during training) ---
        if self._tokens_cached and idx in self._token_id_cache:
            result["token_ids"] = self._token_id_cache[idx]

        result["index"] = idx
        return result

    def _process_caption(self, caption: str) -> str:
        """Apply tag shuffle and caption dropout."""
        # Caption dropout: return empty string
        if self.caption_dropout_rate > 0 and random.random() < self.caption_dropout_rate:
            return ""

        if not self.tag_shuffle:
            return caption

        tags = [t.strip() for t in caption.split(",") if t.strip()]
        if len(tags) <= self.keep_first_n_tags:
            return caption

        # Keep first N tags fixed, shuffle the rest
        fixed = tags[:self.keep_first_n_tags]
        rest = tags[self.keep_first_n_tags:]
        random.shuffle(rest)
        return ", ".join(fixed + rest)

    def cache_latents_from_vae(self, vae, device, dtype, to_disk=False, progress_fn=None):
        """Pre-compute and cache VAE latents for all images.

        When bucket_assignments are set, each image is resized to its
        assigned bucket resolution before encoding. This means latents
        have different spatial dimensions per bucket.

        Speed optimizations:
        - Fast image decoder (turbojpeg → cv2 → PIL)
        - Safetensors disk format (2-5x faster I/O)
        - FP16 latent compression (50% RAM/disk savings)
        - Content-based deduplication (skip identical images)
        - Parallel disk cache writes
        """
        from dataset_sorter.io_speed import (
            fast_decode_image, save_tensor_fast, load_tensor_fast,
            compress_latent_fp16, decompress_latent_fp16,
            compute_file_hashes, save_tensors_parallel,
        )

        vae.eval()
        vae.requires_grad_(False)

        if to_disk and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Deduplication: group identical images by content hash
        hash_groups = compute_file_hashes(self.image_paths)
        # Invert: idx → representative idx (first in its group)
        idx_to_rep: dict[int, int] = {}
        for group_indices in hash_groups.values():
            rep = group_indices[0]
            for idx in group_indices:
                idx_to_rep[idx] = rep

        pending_saves: list[tuple] = []  # (path, tensor) for parallel save

        # First pass: load disk cache and dedup, collect indices needing VAE encode
        to_encode: list[int] = []
        for idx in range(len(self)):
            # Check disk cache first (try safetensors, then .pt)
            if to_disk and self.cache_dir:
                cache_path = self._latent_disk_path(idx)
                try:
                    latent = load_tensor_fast(cache_path, use_safetensors=True)
                    self._latent_cache[idx] = decompress_latent_fp16(latent, dtype)
                    if progress_fn:
                        progress_fn(idx + 1, len(self))
                    continue
                except FileNotFoundError:
                    pass

            # Deduplication: reuse already-encoded latent for identical images
            rep = idx_to_rep.get(idx, idx)
            if rep != idx and rep in self._latent_cache:
                self._latent_cache[idx] = self._latent_cache[rep]
                if to_disk and self.cache_dir:
                    pending_saves.append(
                        (self._latent_disk_path(idx), compress_latent_fp16(self._latent_cache[idx]))
                    )
                if progress_fn:
                    progress_fn(idx + 1, len(self))
                continue

            to_encode.append(idx)

        # Build inverted dedup index: representative idx → list of duplicate idxs
        rep_to_dups: dict[int, list[int]] = {}
        for idx in to_encode:
            rep = idx_to_rep.get(idx, idx)
            if rep != idx:
                rep_to_dups.setdefault(rep, []).append(idx)

        # Filter to_encode: only encode representatives (skip duplicates that will be filled later)
        to_encode_unique = [idx for idx in to_encode if idx_to_rep.get(idx, idx) == idx]

        # Second pass: batch VAE encode, grouped by resolution
        # Group by resolution since batching requires same-size tensors
        resolution_groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
        for idx in to_encode_unique:
            if self._bucket_assignments is not None and idx < len(self._bucket_assignments):
                target_w, target_h = self._bucket_assignments[idx]
            else:
                target_w, target_h = self.resolution, self.resolution

            try:
                img = fast_decode_image(self.image_paths[idx])
            except Exception as e:
                log.warning(f"Failed to open {self.image_paths[idx]}: {e}, using blank image")
                img = Image.new("RGB", (target_w, target_h))

            if self._bucket_assignments is not None:
                t = self._get_transforms_for_resolution(target_w, target_h)
                pixel_values = t(img).unsqueeze(0)
            else:
                pixel_values = self._transforms(img).unsqueeze(0)

            resolution_groups.setdefault((target_w, target_h), []).append((idx, pixel_values))

        # Encode each resolution group in batches
        vae_batch_size = 4
        encoded_count = len(self) - len(to_encode)  # already cached
        for (_w, _h), group in resolution_groups.items():
            for batch_start in range(0, len(group), vae_batch_size):
                batch_items = group[batch_start:batch_start + vae_batch_size]
                batch_tensor = torch.cat(
                    [pv for _, pv in batch_items], dim=0
                ).to(device, dtype=dtype, memory_format=torch.channels_last)

                with torch.no_grad():
                    encoded = vae.encode(batch_tensor).latent_dist.sample()
                    encoded = encoded * vae.config.scaling_factor

                for bi, (idx, _) in enumerate(batch_items):
                    latent = encoded[bi].cpu()
                    self._latent_cache[idx] = latent

                    if to_disk and self.cache_dir:
                        pending_saves.append(
                            (self._latent_disk_path(idx), compress_latent_fp16(latent))
                        )

                    # Fill dedup targets via O(1) lookup
                    for dup_idx in rep_to_dups.get(idx, []):
                        self._latent_cache[dup_idx] = latent
                        if to_disk and self.cache_dir:
                            pending_saves.append(
                                (self._latent_disk_path(dup_idx), compress_latent_fp16(latent))
                            )
                        encoded_count += 1

                    encoded_count += 1
                    if progress_fn:
                        progress_fn(encoded_count, len(self))

        # Flush pending disk saves in parallel
        if pending_saves:
            save_tensors_parallel(pending_saves, num_workers=4, use_safetensors=True)
            log.info(f"Saved {len(pending_saves)} latent cache files in parallel")

        self._latents_cached = True

    def cache_text_encoder_outputs(
        self, tokenizer, text_encoder, device, dtype,
        tokenizer_2=None, text_encoder_2=None,
        tokenizer_3=None, text_encoder_3=None,
        to_disk=False, progress_fn=None,
        clip_skip: int = 0,
        caption_preprocessor=None,
        max_token_length: int = 0,
        max_token_length_2: int = 0,
    ):
        """Pre-compute and cache text encoder outputs.

        Args:
            caption_preprocessor: Optional callable(str) -> str applied to each
                caption before tokenization. Used by Z-Image to apply Qwen3
                chat template so cached embeddings match live encoding.
            max_token_length: Override tokenizer.model_max_length if > 0.
                Z-Image uses 512 while Qwen3's default is 32768.
            max_token_length_2: Override tokenizer_2.model_max_length if > 0.
                Hunyuan uses 256 for mT5 while model default may be larger.

        Speed optimizations:
        - Safetensors disk format (2-5x faster I/O)
        - Caption deduplication (already present, encode each unique caption once)
        """
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        if text_encoder_2 is not None:
            text_encoder_2.eval()
            text_encoder_2.requires_grad_(False)
        if text_encoder_3 is not None:
            text_encoder_3.eval()
            text_encoder_3.requires_grad_(False)

        if to_disk and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache unique captions to avoid recomputation
        unique_captions = {}
        for idx in range(len(self)):
            caption = self.captions[idx]
            if caption not in unique_captions:
                unique_captions[caption] = []
            unique_captions[caption].append(idx)

        processed = 0
        for caption, indices in unique_captions.items():
            # Check disk cache (try safetensors first, then .pt)
            cache_key = hashlib.md5(caption.encode()).hexdigest()
            if to_disk and self.cache_dir:
                cache_path = self.cache_dir / f"te_{cache_key}.pt"
                sf_path = self.cache_dir / f"te_{cache_key}.safetensors"
                loaded = None
                if sf_path.exists():
                    try:
                        from safetensors.torch import load_file
                        data = load_file(str(sf_path))
                        # Reconstruct tuple preserving None positions from key indices
                        # Keys are t00, t01, ... — missing indices were None at save time
                        keys = sorted(data.keys())
                        if not keys:
                            raise ValueError(f"Empty safetensors cache: {sf_path}")
                        max_idx = max(int(k[1:]) for k in keys) + 1
                        loaded = tuple(data.get(f"t{i:02d}") for i in range(max_idx))
                    except (ImportError, Exception):
                        pass
                if loaded is None and cache_path.exists():
                    loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
                if loaded is not None:
                    for idx in indices:
                        self._te_cache[idx] = loaded
                    processed += len(indices)
                    if progress_fn:
                        progress_fn(processed, len(self))
                    continue

            with torch.no_grad():
                # Preprocess caption (e.g. Qwen3 chat template for Z-Image)
                tok_input = caption_preprocessor(caption) if caption_preprocessor else caption
                _max_len = max_token_length if max_token_length > 0 else tokenizer.model_max_length

                # Tokenize
                _tok_out = tokenizer(
                    tok_input, padding="max_length",
                    max_length=_max_len,
                    truncation=True, return_tensors="pt",
                )

                # LLM-based text encoders (e.g. Qwen3 for Z-Image) need
                # attention_mask alongside input_ids. CLIP models accept
                # positional input_ids only.
                if caption_preprocessor is not None:
                    # LLM path: pass full tokenizer output as kwargs
                    _tok_device = {k: v.to(device) for k, v in _tok_out.items()}
                    encoder_output = text_encoder(
                        **_tok_device, output_hidden_states=True,
                    )
                else:
                    tokens = _tok_out.input_ids.to(device)
                    encoder_output = text_encoder(tokens, output_hidden_states=True)

                # LLM text encoders (Qwen3) use the last hidden state [-1].
                # CLIP models use the penultimate layer [-2] by default,
                # adjusted by clip_skip.
                if caption_preprocessor is not None:
                    # LLM path: use last hidden state (matches encode_text_batch)
                    hidden_states = encoder_output.hidden_states[-1].cpu()
                    _skip = 1  # default for LLM path (used by TE2 below)
                    tokens = _tok_out.input_ids.to(device)  # needed for token_id_result
                else:
                    _skip = max(clip_skip, 1)
                    _skip = min(_skip, len(encoder_output.hidden_states) - 2)
                    hidden_states = encoder_output.hidden_states[-(_skip + 1)].cpu()
                # pooler_output contains the pooled [CLS] embedding; [0] is last_hidden_state
                pooled = (
                    encoder_output.pooler_output.cpu()
                    if hasattr(encoder_output, "pooler_output") and encoder_output.pooler_output is not None
                    else None
                )

                te_result = (hidden_states, pooled)
                hidden_states_2 = None
                pooled_2 = None

                # Cache tokenized IDs (avoids re-tokenization every step)
                token_id_result = (_tok_out.input_ids.cpu(),)

                # Second text encoder (SDXL)
                if tokenizer_2 is not None and text_encoder_2 is not None:
                    _max_len2 = max_token_length_2 if max_token_length_2 > 0 else tokenizer_2.model_max_length
                    tokens_2 = tokenizer_2(
                        caption, padding="max_length",
                        max_length=_max_len2,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)

                    encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
                    _skip2 = min(_skip, len(encoder_output_2.hidden_states) - 2)
                    hidden_states_2 = encoder_output_2.hidden_states[-(_skip2 + 1)].cpu()
                    pooled_2 = (
                        encoder_output_2.pooler_output.cpu()
                        if hasattr(encoder_output_2, "pooler_output") and encoder_output_2.pooler_output is not None
                        else None
                    )

                    te_result = (hidden_states, pooled, hidden_states_2, pooled_2)
                    token_id_result = (tokens.cpu(), tokens_2.cpu())

                # Third text encoder (SD3 / SD 3.5 — T5-XXL)
                if tokenizer_3 is not None and text_encoder_3 is not None:
                    tokens_3 = tokenizer_3(
                        caption, padding="max_length",
                        max_length=tokenizer_3.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)

                    encoder_output_3 = text_encoder_3(tokens_3)
                    hidden_states_3 = encoder_output_3.last_hidden_state.cpu()

                    te_result = (hidden_states, pooled, hidden_states_2, pooled_2, hidden_states_3)
                    token_id_result = (tokens.cpu(), tokens_2.cpu() if tokenizer_2 else None, tokens_3.cpu())

            for idx in indices:
                self._te_cache[idx] = te_result
                self._token_id_cache[idx] = token_id_result

            if to_disk and self.cache_dir:
                # Save as safetensors (faster) with .pt fallback
                try:
                    from safetensors.torch import save_file
                    sf_path = self.cache_dir / f"te_{cache_key}.safetensors"
                    # Convert tuple to dict for safetensors
                    te_dict = {}
                    for ti, t in enumerate(te_result):
                        if t is not None:
                            te_dict[f"t{ti:02d}"] = t
                    if te_dict:
                        save_file(te_dict, str(sf_path))
                    else:
                        cache_path = self.cache_dir / f"te_{cache_key}.pt"
                        torch.save(te_result, cache_path)
                except (ImportError, Exception):
                    cache_path = self.cache_dir / f"te_{cache_key}.pt"
                    torch.save(te_result, cache_path)

            processed += len(indices)
            if progress_fn:
                progress_fn(processed, len(self))

        self._te_cached = True
        self._tokens_cached = True
        log.info(f"Cached tokenized IDs for {len(unique_captions)} unique captions")

    def _latent_disk_path(self, idx: int) -> Path:
        """Generate disk cache path for a latent."""
        img_hash = hashlib.md5(str(self.image_paths[idx]).encode()).hexdigest()
        return self.cache_dir / f"latent_{img_hash}.pt"

    def clear_caches(self):
        """Free cached tensors from RAM."""
        self._latent_cache.clear()
        self._te_cache.clear()
        self._token_id_cache.clear()
        self._latents_cached = False
        self._te_cached = False
        self._tokens_cached = False
