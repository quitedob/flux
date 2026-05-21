"""Find exact point of divergence between BF16 and NVFP4 models."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["AE_MODEL_PATH"] = "F:/python/flux/models/ae.safetensors"
os.environ["KLEIN_4B_BASE_MODEL_PATH"] = "F:/python/flux/models/flux-2-klein-base-4b.safetensors"
os.environ["KLEIN_4B_MODEL_PATH"] = "F:/python/flux/models/flux-2-klein-base-4b.safetensors"

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_sft

from src.flux2.model import Flux2, Klein4BParams
from src.flux2.util import load_nvfp4_model, _unpack_fp4_e2m1, _FP4_E2M1_LOOKUP

device = torch.device("cuda")

# ── Step 1: Load both models and compare ALL state dict tensors ──────────
print("=" * 70)
print("STEP 1: Verify all state dict tensors match")
print("=" * 70)

# Load BF16 model
bf16_sd = load_sft("F:/python/flux/models/flux-2-klein-base-4b.safetensors", device="cuda")

# Load NVFP4 model via custom loader
nvfp4_model = load_nvfp4_model("flux.2-klein-base-4b-nvfp4", device="cuda")
nvfp4_model.eval()

# Extract NVFP4 state dict (already dequantized)
nvfp4_sd = {k: v for k, v in nvfp4_model.state_dict().items()}

# Compare every key
keys_only_bf16 = set(bf16_sd.keys()) - set(nvfp4_sd.keys())
keys_only_nvfp4 = set(nvfp4_sd.keys()) - set(bf16_sd.keys())
common_keys = set(bf16_sd.keys()) & set(nvfp4_sd.keys())

print(f"Common keys: {len(common_keys)}")
print(f"Only in BF16: {keys_only_bf16}")
print(f"Only in NVFP4: {keys_only_nvfp4}")

mismatches = []
for k in sorted(common_keys):
    bf16_val = bf16_sd[k]
    nv_val = nvfp4_sd[k]
    if bf16_val.shape != nv_val.shape:
        mismatches.append((k, f"SHAPE: {bf16_val.shape} vs {nv_val.shape}"))
    elif not torch.allclose(bf16_val.float(), nv_val.float(), rtol=1e-3, atol=1e-2):
        max_diff = (bf16_val.float() - nv_val.float()).abs().max().item()
        corr = torch.corrcoef(
            torch.stack([bf16_val.float().flatten(), nv_val.float().flatten()])
        )[0, 1].item()
        mismatches.append((k, f"max_diff={max_diff:.6f}, corr={corr:.6f}"))

if mismatches:
    print(f"\nMISMATCHED WEIGHTS ({len(mismatches)}):")
    for k, info in mismatches:
        print(f"  {k}: {info}")
else:
    print("\nAll weights match between BF16 and NVFP4!")

# ── Step 2: Setup identical dummy inputs ──────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Forward pass with identical inputs")
print("=" * 70)

params = Klein4BParams()
B = 1
img_seq = 256  # 16x16 latent
txt_seq = 128
hidden = params.hidden_size  # 3072
ctx_dim = params.context_in_dim  # 7680
in_ch = params.in_channels  # 128

# Create consistent random inputs
torch.manual_seed(42)
x = torch.randn(B, img_seq, in_ch, device=device, dtype=torch.bfloat16)
ctx = torch.randn(B, txt_seq, ctx_dim, device=device, dtype=torch.bfloat16)
x_ids = torch.randint(0, 32, (B, img_seq, 4), device=device)
ctx_ids = torch.randint(0, 32, (B, txt_seq, 4), device=device)
timesteps = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
guidance = torch.tensor([4.0], device=device, dtype=torch.bfloat16)

# Load BF16 model
bf16_model = Flux2(params).to(torch.bfloat16).to(device)
bf16_model.load_state_dict(bf16_sd, strict=True)
bf16_model.eval()

# ── Step 3: Register hooks on all nn.Linear layers ────────────────────────
print("Registering hooks on all nn.Linear layers...")

bf16_outputs = {}
nvfp4_outputs = {}

def make_hook(storage_dict, prefix):
    def hook(module, input, output):
        storage_dict[prefix] = {
            "input": input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone(),
            "output": output.detach().clone(),
            "weight": module.weight.detach().clone() if hasattr(module, 'weight') and module.weight is not None else None,
        }
    return hook

# Register on NVFP4 model
for name, module in nvfp4_model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(make_hook(nvfp4_outputs, f"nvfp4/{name}"))

# Register on BF16 model
for name, module in bf16_model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(make_hook(bf16_outputs, f"bf16/{name}"))

# ── Step 4: Run forward pass on both models ───────────────────────────────
print("Running BF16 model...")
with torch.no_grad():
    bf16_out = bf16_model(
        x=x, x_ids=x_ids, timesteps=timesteps,
        ctx=ctx, ctx_ids=ctx_ids, guidance=guidance,
    )

print("Running NVFP4 model...")
with torch.no_grad():
    nvfp4_out = nvfp4_model(
        x=x.clone(), x_ids=x_ids.clone(), timesteps=timesteps.clone(),
        ctx=ctx.clone(), ctx_ids=ctx_ids.clone(), guidance=guidance.clone(),
    )

# ── Step 5: Compare final output ──────────────────────────────────────────
print("\n--- FINAL OUTPUT ---")
final_max_diff = (bf16_out - nvfp4_out).abs().max().item()
print(f"  max_diff: {final_max_diff:.6f}")
print(f"  BF16:   [{bf16_out.min().item():.4f}, {bf16_out.max().item():.4f}] std={bf16_out.std().item():.4f}")
print(f"  NVFP4:  [{nvfp4_out.min().item():.4f}, {nvfp4_out.max().item():.4f}] std={nvfp4_out.std().item():.4f}")

# ── Step 6: Compare all hooked layers sequentially ────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Sequential comparison of all Linear layers")
print("=" * 70)

# Get execution order from BF16 hooks (they fire in order)
bf16_order = list(bf16_outputs.keys())

first_diverged = False
for bf16_key in bf16_order:
    layer_name = bf16_key.replace("bf16/", "")
    nv_key = f"nvfp4/{layer_name}"

    bf16_data = bf16_outputs.get(bf16_key)
    nv_data = nvfp4_outputs.get(nv_key)

    if bf16_data is None or nv_data is None:
        print(f"  {layer_name}: MISSING from one model!")
        continue

    # Compare inputs
    inp_bf = bf16_data["input"]
    inp_nv = nv_data["input"]
    inp_max_diff = (inp_bf.float() - inp_nv.float()).abs().max().item()

    # Compare outputs
    out_bf = bf16_data["output"]
    out_nv = nv_data["output"]
    out_max_diff = (out_bf.float() - out_nv.float()).abs().max().item()

    # Compare weights
    w_bf = bf16_data["weight"]
    w_nv = nv_data["weight"]
    w_max_diff = 0.0
    if w_bf is not None and w_nv is not None:
        w_max_diff = (w_bf.float() - w_nv.float()).abs().max().item()

    status = "OK"
    if inp_max_diff > 1e-3 or out_max_diff > 1e-3:
        status = "DIVERGED"
        if not first_diverged:
            first_diverged = True
            print(f"\n  *** FIRST DIVERGENCE at {layer_name} ***")
            print(f"      input  max_diff: {inp_max_diff:.6f}")
            print(f"      weight max_diff: {w_max_diff:.6f}")
            print(f"      output max_diff: {out_max_diff:.6f}")
            print(f"      BF16  out: [{out_bf.min().item():.4f}, {out_bf.max().item():.4f}]")
            print(f"      NVFP4 out: [{out_nv.min().item():.4f}, {out_nv.max().item():.4f}]")
            # Deeper analysis: recompute manually
            with torch.no_grad():
                manual_bf16 = inp_bf @ w_bf.T
                manual_nvfp4 = inp_nv @ w_nv.T
                manual_diff = (manual_bf16.float() - manual_nvfp4.float()).abs().max().item()
                print(f"      Manual recompute (inp @ w.T) max_diff: {manual_diff:.6f}")
                hook_vs_manual_bf16 = (out_bf.float() - manual_bf16.float()).abs().max().item()
                hook_vs_manual_nvfp4 = (out_nv.float() - manual_nvfp4.float()).abs().max().item()
                print(f"      Hook vs manual BF16:  {hook_vs_manual_bf16:.6f}")
                print(f"      Hook vs manual NVFP4: {hook_vs_manual_nvfp4:.6f}")

    if not first_diverged:
        print(f"  {layer_name}: inp_diff={inp_max_diff:.2e} out_diff={out_max_diff:.2e} w_diff={w_max_diff:.2e}")

del bf16_model
torch.cuda.empty_cache()
print("\nDone.")
