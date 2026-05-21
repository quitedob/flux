"""Test NVFP4 model generation — matches CLI flow exactly."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["AE_MODEL_PATH"] = "F:/python/flux/models/ae.safetensors"
# NVFP4 model uses this env var
os.environ["KLEIN_4B_BASE_MODEL_PATH"] = "F:/python/flux/models/flux-2-klein-base-4b-nvfp4.safetensors"

import torch
from einops import rearrange
from PIL import Image

from src.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise_cfg,
    get_schedule,
    scatter_ids,
)
from src.flux2.util import load_ae, load_flow_model, load_text_encoder

device = torch.device("cuda")
dtype = torch.bfloat16

# ── Load models ──
print("Loading models...")
model = load_flow_model("flux.2-klein-base-4b-nvfp4", device=device)
ae = load_ae("flux.2-klein-base-4b-nvfp4", device=device)
text_encoder = load_text_encoder("flux.2-klein-base-4b-nvfp4", device=device)
model.eval()
ae.eval()
text_encoder.eval()
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# ── Config ──
prompt = "a photo of a cat sitting on a table, high quality, detailed"
seed = 42
height, width = 512, 512
num_steps = 50
guidance = 4.0

print(f"Prompt: '{prompt}'")
print(f"Size: {width}x{height}, steps: {num_steps}, guidance: {guidance}, seed: {seed}")

with torch.no_grad():
    # ── Text encoding (follows CLI exactly) ──
    ctx_empty = text_encoder.forward([""]).to(dtype)
    ctx_prompt = text_encoder.forward([prompt]).to(dtype)
    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
    ctx, ctx_ids = batched_prc_txt(ctx)
    print(f"ctx: {ctx.shape}, ctx_ids: {ctx_ids.shape}")

    # ── Noise (follows CLI exactly) ──
    shape = (1, 128, height // 16, width // 16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    randn = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    x, x_ids = batched_prc_img(randn)
    print(f"x: {x.shape}, x_ids: {x_ids.shape}")

    # ── Timesteps ──
    timesteps = get_schedule(num_steps, x.shape[1])
    print(f"Timesteps: {len(timesteps)-1} steps, range [{timesteps[0]:.4f}, {timesteps[-1]:.4f}]")

    # ── Denoise (CFG, no image refs) ──
    print("Denoising...")
    x = denoise_cfg(
        model=model,
        img=x,
        img_ids=x_ids,
        txt=ctx,
        txt_ids=ctx_ids,
        timesteps=timesteps,
        guidance=guidance,
    )
    print(f"Post-denoise x: {x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")

    # ── Decode (follows CLI exactly) ──
    x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
    print(f"Post-scatter: {x.shape}")
    x = ae.decode(x).float()
    print(f"Post-decode: {x.shape}")

# ── Save ──
x = x.clamp(-1, 1)
x = rearrange(x[0], "c h w -> h w c")
img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

out_path = "F:/python/flux/outputs/nvfp4_test.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
img.save(out_path)
print(f"Saved to {out_path}")

# Cleanup
del model, ae, text_encoder
torch.cuda.empty_cache()
print("Done.")
