"""Test FP8 Klein 4B model inference with CPU offloading.

Usage:
    python scripts/test_fp8_gen.py [--prompt "your prompt"] [--steps 4] [--width 1024] [--height 1024]

Requires:
    - Model: models/flux-2-kelin-4b-fp8/flux-2-klein-4b-fp8.safetensors
    - Text encoder: models/Qwen3-4B-FP8/ (local Qwen3-4B in FP8)
    - VAE: models/vae/ae.safetensors
"""
import argparse
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_OFFLINE"] = "1"

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["AE_MODEL_PATH"] = os.path.join(BASE_DIR, "models", "vae", "ae.safetensors")
os.environ["KLEIN_4B_MODEL_PATH"] = os.path.join(
    BASE_DIR, "models", "flux-2-kelin-4b-fp8", "flux-2-klein-4b-fp8.safetensors"
)

import torch
from einops import rearrange
from PIL import Image

from src.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    get_schedule,
    scatter_ids,
)
from src.flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


def main():
    parser = argparse.ArgumentParser(description="Test FP8 Klein 4B inference")
    parser.add_argument("--prompt", type=str, default="a photo of a cat sitting on a table, high quality, detailed")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "outputs", "fp8_test.png"))
    args = parser.parse_args()

    model_name = "flux.2-klein-4b"
    model_info = FLUX2_MODEL_INFO[model_name]
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ── Step 1: Load model to GPU (fast FP8 dequant), then move to CPU ──
    print(f"Loading models for '{model_name}'...")
    model = load_flow_model(model_name, device=device)
    model.eval()
    model = model.cpu()
    torch.cuda.empty_cache()
    print(f"  Model: loaded + moved to CPU")

    # ── Step 2: Load AE + text encoder on GPU ──
    ae = load_ae(model_name, device=device)
    text_encoder = load_text_encoder(model_name, device=device)
    ae.eval()
    text_encoder.eval()

    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB (model on CPU, AE+TE on GPU)")

    # ── Inference ──
    print(f"\nPrompt:   '{args.prompt}'")
    print(f"Size:     {args.width}x{args.height}")
    print(f"Steps:    {args.steps}, guidance: {args.guidance}, seed: {args.seed}\n")

    with torch.no_grad():
        # Text encoding (text_encoder on GPU)
        ctx = text_encoder.forward([args.prompt]).to(dtype)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Offload text_encoder → CPU, move model → GPU
        text_encoder = text_encoder.cpu()
        torch.cuda.empty_cache()
        model = model.to(device)
        print(f"VRAM after swap (te→cpu, model→gpu): {torch.cuda.memory_allocated()/1e9:.2f}GB")

        # Noise
        shape = (1, 128, args.height // 16, args.width // 16)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        randn = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        x, x_ids = batched_prc_img(randn)

        # Timesteps
        timesteps = get_schedule(args.steps, x.shape[1])
        print(f"Timesteps: {len(timesteps)-1} steps")

        # Denoise
        print("Denoising...")
        x = denoise(
            model=model, img=x, img_ids=x_ids, txt=ctx, txt_ids=ctx_ids,
            timesteps=timesteps, guidance=args.guidance,
        )

        # Decode (AE already on GPU)
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    # ── Save ──
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    img.save(args.output)
    print(f"Saved to {args.output}")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak VRAM: {peak_vram:.2f}GB")
    print("Done.")


if __name__ == "__main__":
    main()
