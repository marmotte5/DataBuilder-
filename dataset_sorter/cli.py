"""
Module: cli.py
===============
Command-line interface for DataBuilder.

Enables AI agents (Claude, ChatGPT, etc.) and automation scripts to use
DataBuilder without the GUI. Every GUI feature has a corresponding CLI command
so an agent can orchestrate the full training pipeline from a single shell
session.

Commands:
    databuilder train          Train a LoRA or fine-tune a model
    databuilder generate       Generate images with optional LoRA
    databuilder tag            Auto-tag images in a folder
    databuilder scan-library   Scan and index model/LoRA files
    databuilder analyze-dataset Analyze dataset quality

Usage examples:
    databuilder tag --folder ./photos --model wd-eva02-v3 --trigger-word sks
    databuilder train --model sd15 --dataset ./photos --trigger-word sks --steps 500
    databuilder generate --model sd15 --prompt "a sks cat" --lora ./output/sks.safetensors
    databuilder scan-library --folder ./models
    databuilder analyze-dataset --folder ./photos --trigger-word sks

All commands output JSON to stdout so they can be piped / parsed by scripts.
Progress messages and logs go to stderr.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


# ── Argument parser ───────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="databuilder",
        description="DataBuilder — AI-powered model training and generation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── train ─────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train a LoRA or fine-tune a model")
    tr.add_argument("--model", required=True,
                    help="Model type: sd15, sdxl, flux, sd3, sd35, pony, etc.")
    tr.add_argument("--dataset", required=True,
                    help="Path to dataset folder (images + .txt captions)")
    tr.add_argument("--output", default="./output",
                    help="Output directory for the trained weights (default: ./output)")
    tr.add_argument("--trigger-word", default="",
                    help="Trigger word / activation token for the LoRA")
    tr.add_argument("--steps", type=int, default=500,
                    help="Training steps (default: 500)")
    tr.add_argument("--lora-rank", type=int, default=8,
                    help="LoRA decomposition rank (default: 8)")
    tr.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha (default: lora_rank // 2)")
    tr.add_argument("--optimizer", default="Adafactor",
                    help="Optimizer name (default: Adafactor)")
    tr.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate (default: 1e-4)")
    tr.add_argument("--resolution", type=int, default=512,
                    help="Training resolution in pixels (default: 512)")
    tr.add_argument("--batch-size", type=int, default=1,
                    help="Images per training step (default: 1)")
    tr.add_argument("--network-type", default="lora",
                    choices=["lora", "dora", "full"],
                    help="Network type (default: lora)")
    tr.add_argument("--json-config",
                    help="Path to a JSON file whose keys override all other args. "
                         "Keys match TrainingConfig field names.")

    # ── generate ──────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Generate images")
    gen.add_argument("--model", required=True,
                     help="Model type: sd15, sdxl, flux, etc.")
    gen.add_argument("--prompt", required=True,
                     help="Text prompt describing the image to generate")
    gen.add_argument("--negative-prompt", default="",
                     help="Negative prompt (what to avoid)")
    gen.add_argument("--model-path",
                     help="Path to local .safetensors or diffusers directory. "
                          "Omit to use the HuggingFace default for the model type.")
    gen.add_argument("--lora",
                     help="Path to a .safetensors LoRA file to apply")
    gen.add_argument("--lora-weight", type=float, default=1.0,
                     help="LoRA activation strength (default: 1.0)")
    gen.add_argument("--output", default="./generated",
                     help="Output directory for PNG images (default: ./generated)")
    gen.add_argument("--width", type=int, default=512,
                     help="Image width in pixels (default: 512)")
    gen.add_argument("--height", type=int, default=512,
                     help="Image height in pixels (default: 512)")
    gen.add_argument("--steps", type=int, default=28,
                     help="Diffusion steps (default: 28)")
    gen.add_argument("--cfg", type=float, default=7.0,
                     help="Classifier-free guidance scale (default: 7.0)")
    gen.add_argument("--count", type=int, default=1,
                     help="Number of images to generate (default: 1)")
    gen.add_argument("--seed", type=int, default=-1,
                     help="Random seed (-1 = random, default: -1)")

    # ── tag ───────────────────────────────────────────────────────────
    tg = sub.add_parser("tag", help="Auto-tag images in a folder")
    tg.add_argument("--folder", required=True,
                    help="Path to the image folder")
    tg.add_argument("--model", default="wd-eva02-v3",
                    help="Tagger model key (default: wd-eva02-v3). "
                         "Run 'databuilder list-taggers' to see all options.")
    tg.add_argument("--threshold", type=float, default=0.35,
                    help="General tag confidence threshold (default: 0.35)")
    tg.add_argument("--char-threshold", type=float, default=0.85,
                    help="Character tag confidence threshold (default: 0.85)")
    tg.add_argument("--trigger-word", default="",
                    help="Prepend this word to every generated caption")
    tg.add_argument("--overwrite", action="store_true",
                    help="Re-tag images that already have a .txt file")
    tg.add_argument("--include-rating", action="store_true",
                    help="Include the top WD rating tag in output (WD models only)")
    tg.add_argument("--format", dest="output_format", default="booru",
                    choices=["booru", "natural"],
                    help="Output format: 'booru' (comma tags) or 'natural' (sentence)")

    # ── scan-library ──────────────────────────────────────────────────
    sl = sub.add_parser("scan-library", help="Scan and index model files")
    sl.add_argument("--folder", required=True,
                    help="Root folder to scan for model files")
    sl.add_argument("--no-recursive", action="store_true",
                    help="Do not recurse into subdirectories")

    # ── analyze-dataset ───────────────────────────────────────────────
    ad = sub.add_parser("analyze-dataset", help="Analyze dataset quality and statistics")
    ad.add_argument("--folder", required=True,
                    help="Path to the dataset folder")
    ad.add_argument("--trigger-word", default="",
                    help="Check whether captions contain this trigger word")

    # ── list-taggers ──────────────────────────────────────────────────
    sub.add_parser("list-taggers", help="List all available tagger models")

    # ── list-models ───────────────────────────────────────────────────
    sub.add_parser("list-models", help="List all supported model architecture types")

    # ── serve-mcp ─────────────────────────────────────────────────────
    sub.add_parser("serve-mcp", help="Start the stdio MCP server for AI agent integration")

    return parser


# ── Command handlers ──────────────────────────────────────────────────

def _handle_train(args) -> dict:
    """Execute the train command and return a result dict."""
    from dataset_sorter.api import train_lora

    extra_config: dict = {}
    if args.json_config:
        cfg_path = Path(args.json_config)
        if not cfg_path.exists():
            _die(f"JSON config not found: {args.json_config}")
        with open(cfg_path) as f:
            extra_config = json.load(f)
        log.info("Loaded extra config from %s: %d keys", args.json_config, len(extra_config))

    def _progress(step, total, msg):
        print(
            json.dumps({"event": "progress", "step": step, "total": total, "message": msg}),
            file=sys.stderr,
            flush=True,
        )

    # Explicit CLI args serve as defaults; json-config overrides them.
    # progress_callback is always injected last so it cannot be overridden.
    kwargs: dict = {
        "model_type":    args.model,
        "dataset":       args.dataset,
        "output":        args.output,
        "trigger_word":  args.trigger_word,
        "steps":         args.steps,
        "lora_rank":     args.lora_rank,
        "lora_alpha":    args.lora_alpha,
        "optimizer":     args.optimizer,
        "learning_rate": args.lr,
        "resolution":    args.resolution,
        "batch_size":    args.batch_size,
        "network_type":  args.network_type,
    }
    kwargs.update(extra_config)  # JSON config wins over CLI defaults
    kwargs["progress_callback"] = _progress
    return train_lora(**kwargs)


def _handle_generate(args) -> dict:
    """Execute the generate command and return a result dict."""
    from dataset_sorter.api import generate

    return generate(
        model_type=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_path=args.model_path,
        lora=args.lora,
        lora_weight=args.lora_weight,
        output=args.output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg,
        count=args.count,
        seed=args.seed,
    )


def _handle_tag(args) -> dict:
    """Execute the tag command and return a result dict."""
    from dataset_sorter.api import tag_folder

    total_processed = [0]

    def _progress(current, total, filename):
        if filename != "done":
            total_processed[0] = current
        print(
            json.dumps({"event": "progress", "current": current, "total": total, "file": filename}),
            file=sys.stderr,
            flush=True,
        )

    return tag_folder(
        folder=args.folder,
        model=args.model,
        overwrite=args.overwrite,
        threshold_general=args.threshold,
        threshold_character=args.char_threshold,
        include_rating=args.include_rating,
        trigger_word=args.trigger_word,
        output_format=args.output_format,
        progress_callback=_progress,
    )


def _handle_scan_library(args) -> list:
    """Execute the scan-library command and return a result list."""
    from dataset_sorter.api import scan_models

    return scan_models(args.folder, recursive=not args.no_recursive)


def _handle_analyze_dataset(args) -> dict:
    """Execute the analyze-dataset command and return a result dict."""
    from dataset_sorter.api import analyze_dataset

    return analyze_dataset(args.folder, trigger_word=args.trigger_word)


def _handle_list_taggers(_args) -> list:
    """Return available tagger models as a list of dicts."""
    from dataset_sorter.api import list_tagger_models

    return list_tagger_models()


def _handle_list_models(_args) -> list:
    """Return supported model architecture keys."""
    from dataset_sorter.api import list_model_types

    return [{"key": k} for k in list_model_types()]


# ── Utility helpers ───────────────────────────────────────────────────

def _die(message: str, code: int = 1) -> None:
    """Print an error message to stderr and exit with a non-zero status."""
    print(json.dumps({"error": message}), file=sys.stderr)
    sys.exit(code)


def _output(data) -> None:
    """Serialize result to stdout as pretty-printed JSON."""
    print(json.dumps(data, indent=2, default=str))


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Console script entry point (registered as 'databuilder' in pyproject.toml)."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging early so library code uses the right level
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.command:
        parser.print_help()
        sys.exit(0)

    def _handle_serve_mcp(_args):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        from dataset_sorter.mcp_server import run_server
        run_server()

    dispatch = {
        "train":            _handle_train,
        "generate":         _handle_generate,
        "tag":              _handle_tag,
        "scan-library":     _handle_scan_library,
        "analyze-dataset":  _handle_analyze_dataset,
        "list-taggers":     _handle_list_taggers,
        "list-models":      _handle_list_models,
        "serve-mcp":        _handle_serve_mcp,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        _die(f"Unknown command: {args.command!r}")

    try:
        result = handler(args)
        _output(result)
    except KeyboardInterrupt:
        _die("Interrupted by user", code=130)
    except Exception as exc:
        log.exception("Command '%s' failed", args.command)
        _die(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
