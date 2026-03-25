"""Test script: LoRA SD 1.5 training on M1 8GB — 10 steps minimum."""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("test_lora_sd15_m1")

DATASET_DIR = Path("/Users/loudelestre/DataBuilder-/test_dataset")
OUTPUT_DIR = Path("/Users/loudelestre/DataBuilder-/test_output_lora_sd15")
# Use local cache if already downloaded, else download from HF
MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def load_dataset():
    image_paths = sorted(DATASET_DIR.glob("*.png"))
    captions = []
    for img in image_paths:
        txt = img.with_suffix(".txt")
        captions.append(txt.read_text().strip() if txt.exists() else "")
    log.info(f"Dataset: {len(image_paths)} images")
    return image_paths, captions


def main():
    from dataset_sorter.models import TrainingConfig
    from dataset_sorter.trainer import Trainer, get_gpu_info

    # Report device
    gpu = get_gpu_info()
    log.info(f"Device: {gpu['device']} ({gpu['backend']}) — torch {gpu['torch_version']}")

    config = TrainingConfig(
        model_type="sd15_lora",
        resolution=256,            # Léger pour M1 8GB
        batch_size=1,
        gradient_accumulation=1,
        mixed_precision="bf16",    # MPS supporte bf16
        lora_rank=4,
        lora_alpha=4,
        max_train_steps=10,
        warmup_steps=1,
        save_every_n_steps=9999,   # Ne pas sauvegarder pendant le test
        sample_every_n_steps=9999, # Ne pas générer de samples
        gradient_checkpointing=True,
        cache_latents=True,
        cache_text_encoder=True,
        train_text_encoder=False,  # Économise la VRAM
        train_text_encoder_2=False,
        xformers=False,
        torch_compile=False,
        flash_attention=False,
        sdpa=True,
        async_dataload=False,      # MPS: pas d'async GPU prefetch
        cuda_graph_training=False,
        async_optimizer_step=False,
        pipeline_integration=False, # Skip validation pipeline
        enable_bucket=False,        # Dataset simple, pas besoin de bucketing
        optimizer="Adafactor",
        adafactor_relative_step=False,
        adafactor_scale_parameter=False,
        learning_rate=1e-4,
        fp16_latent_cache=False,    # Évite les problèmes de cast sur MPS
    )

    image_paths, captions = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(config)
    log.info(f"Training dtype: {trainer.dtype}")

    steps_done = []
    losses = []
    t_start = None

    def on_progress(current, total, msg):
        log.info(f"[{current}/{total}] {msg}")

    def on_loss(step, loss, lr):
        nonlocal t_start
        if t_start is None:
            t_start = time.time()
        steps_done.append(step)
        losses.append(loss)
        elapsed = time.time() - t_start if t_start else 0
        it_s = len(steps_done) / elapsed if elapsed > 0 else 0
        log.info(f"  step={step:4d}  loss={loss:.4f}  lr={lr:.2e}  speed={it_s:.3f} it/s")

    try:
        log.info("=== Setup (chargement modèle, cache latents) ===")
        t0 = time.time()
        trainer.setup(
            model_path=MODEL_PATH,
            image_paths=image_paths,
            captions=captions,
            output_dir=OUTPUT_DIR,
            progress_fn=on_progress,
        )
        log.info(f"Setup terminé en {time.time()-t0:.1f}s")

        log.info("=== Training (10 steps) ===")
        t_start = time.time()
        trainer.train(
            progress_fn=on_progress,
            loss_fn=on_loss,
        )

        total_time = time.time() - t_start
        log.info("=== Résultat ===")
        log.info(f"Steps complétés : {len(steps_done)}/{config.max_train_steps}")
        if losses:
            log.info(f"Loss final      : {losses[-1]:.4f}")
            log.info(f"Loss moyen      : {sum(losses)/len(losses):.4f}")
        if total_time > 0 and steps_done:
            log.info(f"Vitesse         : {len(steps_done)/total_time:.3f} it/s")
            log.info(f"Temps total     : {total_time:.1f}s")
        log.info("SUCCESS: LoRA SD 1.5 training fonctionne sur M1!")

    except MemoryError as e:
        log.error(f"OOM (MemoryError): {e}")
        log.error("Essaie resolution=128 ou gradient_checkpointing=True")
        sys.exit(1)
    except Exception as e:
        log.error(f"ERREUR: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            trainer.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    # Fix macOS multiprocessing (déjà appliqué mais par sécurité)
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
