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


def training_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable keys and None values in tuples.

    The dataset returns dicts with different keys depending on caching state
    (e.g. 'latent' vs 'pixel_values', optional 'te_cache'/'token_ids').
    PyTorch's default_collate fails on mismatched keys and None in tuples.
    This collate only batches keys present in ALL samples and handles None.
    """
    if not batch:
        return {}

    # Only collate keys present in every sample
    common_keys = set(batch[0].keys())
    for sample in batch[1:]:
        common_keys &= set(sample.keys())

    result = {}
    for key in common_keys:
        values = [sample[key] for sample in batch]
        elem = values[0]

        if isinstance(elem, torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(elem, (int, float)):
            result[key] = torch.tensor(values)
        elif isinstance(elem, str):
            result[key] = values  # keep as list
        elif isinstance(elem, tuple):
            # Recursively collate each position, preserving None.
            # Use min length across samples to avoid IndexError when
            # tuple lengths vary (e.g. some cached with mask, some without).
            lengths = [len(v) for v in values]
            min_len = min(lengths)
            if min_len != max(lengths):
                log.warning(
                    "TE cache tuple length mismatch in batch (lengths: %s). "
                    "This may indicate cache corruption or mixed model types. "
                    "Truncating to min length %d.", lengths, min_len,
                )
            collated_tuple = []
            for pos in range(min_len):
                pos_values = [v[pos] for v in values]
                if all(pv is None for pv in pos_values):
                    collated_tuple.append(None)
                elif all(isinstance(pv, torch.Tensor) for pv in pos_values):
                    collated_tuple.append(torch.stack(pos_values))
                elif any(pv is None for pv in pos_values):
                    # Mixed None/tensor — keep None (can't stack)
                    collated_tuple.append(None)
                else:
                    collated_tuple.append(pos_values)
            result[key] = tuple(collated_tuple)
        else:
            result[key] = values  # fallback: keep as list

    return result


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
                max(width, height),
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

        # --- Text encoder cache ---
        # When TE outputs are cached, always use the cached embedding regardless
        # of tag shuffle. CLIP/T5 text encoders are largely order-invariant for
        # comma-separated tags, and the cached embedding from the original caption
        # is a valid approximation. This prevents a critical failure where the
        # shuffled caption doesn't match the cache key, causing a fallback to
        # live encoding on an already-offloaded text encoder.
        # Caption dropout (empty string) also uses the cached embedding — the
        # dropout effect is achieved by the training loop, not the TE output.
        if self._te_cached and idx in self._te_cache:
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
            fast_decode_image, load_tensor_fast,
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
                    latent = decompress_latent_fp16(latent, dtype)
                    if torch.cuda.is_available():
                        latent = latent.pin_memory()
                    self._latent_cache[idx] = latent
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

        # Encode each resolution group in batches with dynamic batch sizing.
        # Start with a larger batch and reduce on OOM for maximum throughput.
        vae_batch_size = self._estimate_vae_batch_size(device)
        # Read shift_factor once before the loop so it's available in OOM
        # fallback paths (where vae.encode() may fail before assigning it).
        shift = getattr(vae.config, 'shift_factor', 0.0)
        encoded_count = len(self) - len(to_encode)  # already cached
        for (_w, _h), group in resolution_groups.items():
            for batch_start in range(0, len(group), vae_batch_size):
                batch_items = group[batch_start:batch_start + vae_batch_size]
                batch_tensor = torch.cat(
                    [pv for _, pv in batch_items], dim=0
                ).to(device, dtype=dtype, memory_format=torch.channels_last)

                try:
                    with torch.no_grad():
                        encoded = vae.encode(batch_tensor).latent_dist.sample()
                        if shift:
                            encoded = (encoded - shift) * vae.config.scaling_factor
                        else:
                            encoded = encoded * vae.config.scaling_factor
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and vae_batch_size > 1:
                        # OOM: reduce batch size and retry this batch one-by-one
                        vae_batch_size = max(1, vae_batch_size // 2)
                        log.warning(f"VAE OOM, reducing batch size to {vae_batch_size}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Re-encode items individually as fallback
                        for idx_item, pv_item in batch_items:
                            single = pv_item.to(device, dtype=dtype, memory_format=torch.channels_last)
                            with torch.no_grad():
                                enc = vae.encode(single).latent_dist.sample()
                                if shift:
                                    enc = (enc - shift) * vae.config.scaling_factor
                                else:
                                    enc = enc * vae.config.scaling_factor
                            latent = enc[0].cpu()
                            if torch.cuda.is_available():
                                latent = latent.pin_memory()
                            self._latent_cache[idx_item] = latent
                            if to_disk and self.cache_dir:
                                pending_saves.append(
                                    (self._latent_disk_path(idx_item), compress_latent_fp16(latent))
                                )
                            for dup_idx in rep_to_dups.get(idx_item, []):
                                self._latent_cache[dup_idx] = latent
                                if to_disk and self.cache_dir:
                                    pending_saves.append(
                                        (self._latent_disk_path(dup_idx), compress_latent_fp16(latent))
                                    )
                                encoded_count += 1
                            encoded_count += 1
                            if progress_fn:
                                progress_fn(encoded_count, len(self))
                        continue
                    else:
                        raise

                for bi, (idx, _) in enumerate(batch_items):
                    latent = encoded[bi].cpu()
                    # Pin memory for faster async CPU→GPU transfers (15-25% speedup)
                    if torch.cuda.is_available():
                        latent = latent.pin_memory()
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
        encode_fn=None,
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
            encode_fn: Optional callable(list[str]) -> tuple that replaces the
                generic per-encoder caching logic. Used by backends with custom
                encoding (Flux2 multi-layer extraction, HiDream 4-encoder
                concatenation) that can't be reproduced by the generic path.
                Must return a tuple of tensors on CPU.

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
            # Check disk cache (try safetensors first, then .pt).
            # Include encoding params in key so changing clip_skip/max_length
            # invalidates stale cached embeddings from a different config.
            _cache_str = (
                f"{caption}|cs={clip_skip}|ml={max_token_length}"
                f"|ml2={max_token_length_2}|pp={caption_preprocessor is not None}"
            )
            cache_key = hashlib.md5(_cache_str.encode()).hexdigest()
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
                # Backend-provided encode function for models with custom
                # encoding that the generic path can't reproduce (Flux2
                # multi-layer extraction, HiDream 4-encoder concatenation).
                if encode_fn is not None:
                    te_result = tuple(
                        t.cpu() if isinstance(t, torch.Tensor) else t
                        for t in encode_fn([caption])
                    )
                    # Squeeze batch dim since we encode one caption at a time
                    te_result = tuple(
                        t.squeeze(0) if isinstance(t, torch.Tensor) and t.dim() > 1 else t
                        for t in te_result
                    )
                    token_id_result = ()  # token IDs not available via encode_fn

                    for idx in indices:
                        self._te_cache[idx] = te_result
                        self._token_id_cache[idx] = token_id_result

                    if to_disk and self.cache_dir:
                        try:
                            from safetensors.torch import save_file
                            sf_path = self.cache_dir / f"te_{cache_key}.safetensors"
                            te_dict = {}
                            for ti, t in enumerate(te_result):
                                if t is not None and isinstance(t, torch.Tensor):
                                    te_dict[f"t{ti:02d}"] = t
                            if te_dict:
                                save_file(te_dict, str(sf_path))
                        except (ImportError, Exception):
                            cache_path = self.cache_dir / f"te_{cache_key}.pt"
                            torch.save(te_result, cache_path)

                    processed += len(indices)
                    if progress_fn:
                        progress_fn(processed, len(self))
                    continue

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

                # LLM text encoders (Qwen3 for Z-Image) use second-to-last
                # hidden state [-2], matching the official pipeline.
                # CLIP models use the penultimate layer [-2] by default,
                # adjusted by clip_skip.
                if caption_preprocessor is not None:
                    # LLM path: second-to-last hidden state (matches pipeline).
                    # Squeeze batch dim since caching runs with batch=1.
                    hidden_states = encoder_output.hidden_states[-2].squeeze(0).cpu()
                    _skip = 1  # default for LLM path (used by TE2 below)
                    tokens = _tok_out.input_ids.to(device)  # needed for token_id_result
                else:
                    _skip = max(clip_skip, 1)
                    _skip = min(_skip, len(encoder_output.hidden_states) - 2)
                    hidden_states = encoder_output.hidden_states[-(_skip + 1)].squeeze(0).cpu()
                # pooler_output contains the pooled [CLS] embedding; [0] is last_hidden_state.
                # For LLM text encoders (e.g. Qwen3), store the attention mask
                # instead — the transformer needs it to strip padding tokens.
                if caption_preprocessor is not None:
                    # LLM path: store attention mask (used to strip padding
                    # in training_step before passing to transformer).
                    # Squeeze to [seq_len] since caching runs with batch=1.
                    pooled = _tok_out["attention_mask"].squeeze(0).cpu()
                else:
                    pooled = (
                        encoder_output.pooler_output.squeeze(0).cpu()
                        if hasattr(encoder_output, "pooler_output") and encoder_output.pooler_output is not None
                        else None
                    )

                te_result = (hidden_states, pooled)
                hidden_states_2 = None
                pooled_2 = None
                tokens_2 = None

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
                    hidden_states_2 = encoder_output_2.hidden_states[-(_skip2 + 1)].squeeze(0).cpu()
                    pooled_2 = (
                        encoder_output_2.pooler_output.squeeze(0).cpu()
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
                    token_id_result = (tokens.cpu(), tokens_2.cpu() if tokens_2 is not None else None, tokens_3.cpu())

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

    @staticmethod
    def _estimate_vae_batch_size(device) -> int:
        """Estimate optimal VAE batch size based on available VRAM.

        Starts large and allows dynamic reduction on OOM. This maximizes
        throughput by using as much VRAM as available for batched encoding.
        """
        try:
            import torch
            if device.type == "cuda" and torch.cuda.is_available():
                free_gb = (
                    torch.cuda.get_device_properties(device).total_memory
                    - torch.cuda.memory_allocated(device)
                ) / (1024 ** 3)
                # Each image at 1024x1024 bf16 needs ~0.5 GB for VAE forward
                # Use ~60% of free memory for batching, leave headroom
                batch = max(1, int(free_gb * 0.6 / 0.5))
                return min(batch, 16)  # Cap at 16 for memory safety
        except (ImportError, RuntimeError):
            pass
        return 4  # Conservative default

    def _latent_disk_path(self, idx: int) -> Path:
        """Generate disk cache path for a latent.

        Includes bucket resolution in the hash so latents are invalidated
        when bucket assignments change between runs.
        """
        key = str(self.image_paths[idx])
        if self._bucket_assignments is not None and idx < len(self._bucket_assignments):
            w, h = self._bucket_assignments[idx]
            key += f"_{w}x{h}"
        img_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"latent_{img_hash}.pt"

    def clear_caches(self):
        """Free cached tensors from RAM."""
        self._latent_cache.clear()
        self._te_cache.clear()
        self._token_id_cache.clear()
        self._latents_cached = False
        self._te_cached = False
        self._tokens_cached = False
