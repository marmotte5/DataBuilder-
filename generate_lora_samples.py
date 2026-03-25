"""Generate sample images using trained SD 1.5 LoRA."""
import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

LORA_PATH = Path.home() / "DataBuilder-/test_output_lora_sd15/checkpoints/final"
OUTPUT_DIR = Path.home() / "DataBuilder-/test_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("a red square on white background", "red_square"),
    ("a blue circle on dark background", "blue_circle"),
    ("a green triangle on white background", "green_triangle"),
    ("colorful geometric shapes", "colorful_shapes"),
]

print("Loading SD 1.5 pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = pipe.to("mps")

print(f"Loading LoRA from {LORA_PATH}...")
pipe.load_lora_weights(str(LORA_PATH))
print("LoRA loaded.")

generator = torch.Generator(device="cpu").manual_seed(42)

for prompt, name in PROMPTS:
    print(f"Generating: {prompt}")
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=256,
                width=256,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator,
            )
        img = result.images[0]
        out_path = OUTPUT_DIR / f"{name}.png"
        img.save(out_path)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("Done.")
