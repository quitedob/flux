"""Benchmark FP8 vs BF16 Klein 4B model on generation speed and PSNR quality.

Compares the FP8-quantized model (models/flux-2-kelin-4b-fp8/) against the
original BF16 model (models/flux-2-klein-4b/) using identical prompts and seeds.

Metrics:
    - Load time: safetensors read + dequantize + model creation
    - Inference time: per-step GPU time for denoising, plus total wall-clock
    - Peak VRAM: maximum GPU memory allocated during inference
    - PSNR: Peak Signal-to-Noise Ratio between FP8 and BF16 outputs
    - SSIM: Structural Similarity Index between FP8 and BF16 outputs

Usage:
    python scripts/benchmark_fp8_vs_bf16.py [--steps 4] [--width 1024] [--height 1024]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    get_schedule,
    scatter_ids,
)
from src.flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
AE_PATH = BASE_DIR / "models" / "vae" / "ae.safetensors"
BF16_MODEL_PATH = BASE_DIR / "models" / "flux-2-klein-4b" / "flux-2-klein-4b.safetensors"
FP8_MODEL_PATH = BASE_DIR / "models" / "flux-2-kelin-4b-fp8" / "flux-2-klein-4b-fp8.safetensors"
OUTPUT_BASE = BASE_DIR / "outputs" / "benchmark_fp8_vs_bf16"

# ── Test prompts ───────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    "a photo of a cat sitting on a table, high quality, detailed",
    "a serene mountain landscape at sunset, golden hour lighting",
    "a modern minimalist living room with large windows",
    "a close-up macro photo of a butterfly on a flower",
    "a cyberpunk city street at night with neon lights",
]

SEEDS = [42, 123, 456]


def format_time(seconds: float) -> str:
    """Human-readable time string."""
    if seconds >= 60:
        return f"{seconds:.1f}s ({seconds/60:.1f}m)"
    elif seconds >= 1:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds*1000:.0f}ms"


def compute_quality_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Compute PSNR and SSIM between two RGB uint8 images [H, W, 3].

    Args:
        img1, img2: numpy arrays of shape (H, W, 3), dtype uint8.

    Returns:
        Dict with 'psnr_db' and 'ssim' keys.
    """
    psnr = float(peak_signal_noise_ratio(img1, img2, data_range=255))
    # SSIM with channel_axis=2 for HWC format
    ssim = float(structural_similarity(img1, img2, channel_axis=2, data_range=255))
    return {"psnr_db": round(psnr, 3), "ssim": round(ssim, 5)}


def run_one_model(
    model_path: Path,
    model_name: str,
    label: str,
    output_dir: Path,
    prompts: list[str],
    seeds: list[int],
    steps: int,
    width: int,
    height: int,
    guidance: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Run full inference pipeline for one model variant.

    Returns:
        dict with keys: label, load_time_s, runs (list of per-prompt/seed results),
        peak_vram_gb, model_size_mb.
    """
    # Set env vars for model loading
    os.environ["AE_MODEL_PATH"] = str(AE_PATH)
    os.environ["KLEIN_4B_MODEL_PATH"] = str(model_path)

    results = {"label": label, "model_path": str(model_path), "runs": []}

    # ── Measure model load time ──
    print(f"\n{'='*70}")
    print(f"  Loading [{label}] model: {model_path.name}")
    print(f"{'='*70}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = load_flow_model(model_name, device=device)
    model.eval()
    # Offload to CPU
    model = model.cpu()
    torch.cuda.empty_cache()
    load_time = time.perf_counter() - t0
    results["load_time_s"] = round(load_time, 2)
    print(f"  Model load + CPU offload: {format_time(load_time)}")

    # Model file size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    results["model_size_mb"] = round(model_size_mb, 1)
    print(f"  Model file size: {model_size_mb:.1f} MB")

    # ── Load AE + Text Encoder (shared, but reloaded for clean measurement) ──
    ae = load_ae(model_name, device=device)
    ae.eval()
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()

    vram_after_load = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM after loading AE+TE: {vram_after_load:.2f}GB / {vram_total:.2f}GB")

    # ── Run inference for each prompt × seed ──
    total_runs = len(prompts) * len(seeds)
    run_idx = 0

    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(seeds):
            run_idx += 1
            print(f"\n  [{label}] Run {run_idx}/{total_runs}: "
                  f"prompt={pi+1}/{len(prompts)}, seed={seed}")

            torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()

            with torch.no_grad():
                # ── Text encoding ──
                ctx = text_encoder.forward([prompt]).to(dtype)
                ctx, ctx_ids = batched_prc_txt(ctx)

                # ── Swap: TE → CPU, model → GPU ──
                te_device = next(text_encoder.parameters()).device
                text_encoder = text_encoder.cpu()
                torch.cuda.empty_cache()
                model = model.to(device)

                swap_after_vram = torch.cuda.memory_allocated() / 1e9
                print(f"    VRAM after swap (model on GPU): {swap_after_vram:.2f}GB")

                # ── Noise ──
                shape = (1, 128, height // 16, width // 16)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                randn = torch.randn(shape, generator=generator, dtype=dtype, device=device)
                x, x_ids = batched_prc_img(randn)

                # ── Timesteps ──
                timesteps = get_schedule(steps, x.shape[1])

                # ── Denoise (with GPU timing) ──
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                x = denoise(
                    model=model,
                    img=x,
                    img_ids=x_ids,
                    txt=ctx,
                    txt_ids=ctx_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                )
                end_ev.record()
                torch.cuda.synchronize()
                denoise_time = start_ev.elapsed_time(end_ev) / 1000.0  # ms → s

                # ── Decode ──
                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = ae.decode(x).float()

                # ── Swap back: model → CPU, TE → GPU ──
                model = model.cpu()
                torch.cuda.empty_cache()
                text_encoder = text_encoder.to(te_device)

            total_time = time.perf_counter() - t_start
            peak_vram_run = torch.cuda.max_memory_allocated() / 1e9

            # ── Convert to numpy image ──
            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img_np = (127.5 * (x.cpu() + 1.0)).byte().numpy()

            # ── Save image ──
            img_out_dir = output_dir / label
            img_out_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_out_dir / f"prompt{pi:02d}_seed{seed:03d}.png"
            Image.fromarray(img_np).save(img_path)

            run_result = {
                "prompt_idx": pi,
                "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "seed": seed,
                "denoise_gpu_time_s": round(denoise_time, 3),
                "denoise_per_step_ms": round(denoise_time / steps * 1000, 1),
                "total_wall_time_s": round(total_time, 3),
                "peak_vram_gb": round(peak_vram_run, 2),
                "image_path": str(img_path),
            }
            results["runs"].append(run_result)

            print(f"    Denoise (GPU): {denoise_time:.3f}s ({denoise_time/steps*1000:.0f}ms/step)")
            print(f"    Total wall:    {total_time:.3f}s")
            print(f"    Peak VRAM:     {peak_vram_run:.2f}GB")
            print(f"    Saved:         {img_path}")

    # Clean up large objects
    del model, ae, text_encoder
    torch.cuda.empty_cache()

    # Summary stats
    denoise_times = [r["denoise_gpu_time_s"] for r in results["runs"]]
    wall_times = [r["total_wall_time_s"] for r in results["runs"]]
    results["summary"] = {
        "num_runs": len(results["runs"]),
        "denoise_mean_s": round(np.mean(denoise_times), 3),
        "denoise_std_s": round(np.std(denoise_times), 3),
        "denoise_per_step_mean_ms": round(np.mean(denoise_times) / steps * 1000, 1),
        "wall_time_mean_s": round(np.mean(wall_times), 3),
        "wall_time_std_s": round(np.std(wall_times), 3),
    }
    print(f"\n  [{label}] Summary: denoise={results['summary']['denoise_mean_s']:.3f}s ± "
          f"{results['summary']['denoise_std_s']:.3f}s, "
          f"wall={results['summary']['wall_time_mean_s']:.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 vs BF16 Klein 4B model speed and quality"
    )
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--prompts", type=int, default=3,
                        help="Number of test prompts to use (1-5)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per prompt")
    parser.add_argument("--skip-bf16", action="store_true", help="Skip BF16 model")
    parser.add_argument("--skip-fp8", action="store_true", help="Skip FP8 model")
    args = parser.parse_args()

    # Validate
    prompts = TEST_PROMPTS[: args.prompts]
    seeds = SEEDS[: args.seeds]
    assert len(prompts) > 0, "Need at least 1 prompt"

    model_name = "flux.2-klein-4b"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("  FLUX.2 Klein 4B — FP8 vs BF16 Benchmark")
    print("=" * 70)
    print(f"  Prompts:  {len(prompts)}")
    print(f"  Seeds:    {seeds}")
    print(f"  Steps:    {args.steps}")
    print(f"  Size:     {args.width}×{args.height}")
    print(f"  Guidance: {args.guidance}")
    print(f"  Device:   {device}  |  dtype: {dtype}")
    print(f"  Total:    {len(prompts) * len(seeds)} images per model")
    print()

    # Create output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}\n")

    # Check model files exist
    for path, label in [(BF16_MODEL_PATH, "BF16"), (FP8_MODEL_PATH, "FP8")]:
        if not path.exists():
            print(f"ERROR: {label} model not found at {path}")
            sys.exit(1)
    if not AE_PATH.exists():
        print(f"ERROR: AE model not found at {AE_PATH}")
        sys.exit(1)

    all_results = {"config": vars(args), "timestamp": ts, "models": {}}

    # ── Run BF16 model ──
    if not args.skip_bf16:
        print("\n" + "█" * 70)
        print("█  PHASE 1: Original BF16 Model")
        print("█" * 70)
        bf16_results = run_one_model(
            model_path=BF16_MODEL_PATH,
            model_name=model_name,
            label="bf16",
            output_dir=output_dir,
            prompts=prompts,
            seeds=seeds,
            steps=args.steps,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            device=device,
            dtype=dtype,
        )
        all_results["models"]["bf16"] = bf16_results

    # ── Run FP8 model ──
    if not args.skip_fp8:
        print("\n" + "█" * 70)
        print("█  PHASE 2: FP8 Quantized Model")
        print("█" * 70)
        fp8_results = run_one_model(
            model_path=FP8_MODEL_PATH,
            model_name=model_name,
            label="fp8",
            output_dir=output_dir,
            prompts=prompts,
            seeds=seeds,
            steps=args.steps,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            device=device,
            dtype=dtype,
        )
        all_results["models"]["fp8"] = fp8_results

    # ── Quality comparison (PSNR / SSIM) ──
    if not args.skip_bf16 and not args.skip_fp8:
        print("\n" + "█" * 70)
        print("█  PHASE 3: Quality Comparison (PSNR + SSIM)")
        print("█" * 70)

        bf16_runs = all_results["models"]["bf16"]["runs"]
        fp8_runs = all_results["models"]["fp8"]["runs"]

        quality_results = []
        for bf16_run, fp8_run in zip(bf16_runs, fp8_runs):
            assert bf16_run["seed"] == fp8_run["seed"]
            assert bf16_run["prompt_idx"] == fp8_run["prompt_idx"]

            bf16_img = np.array(Image.open(bf16_run["image_path"]))
            fp8_img = np.array(Image.open(fp8_run["image_path"]))
            metrics = compute_quality_metrics(bf16_img, fp8_img)

            qr = {
                "prompt_idx": bf16_run["prompt_idx"],
                "seed": bf16_run["seed"],
                **metrics,
            }
            quality_results.append(qr)
            print(f"  prompt={bf16_run['prompt_idx']} seed={bf16_run['seed']}: "
                  f"PSNR={metrics['psnr_db']:.2f}dB, SSIM={metrics['ssim']:.5f}")

        # Aggregate quality
        psnr_values = [q["psnr_db"] for q in quality_results]
        ssim_values = [q["ssim"] for q in quality_results]
        all_results["quality"] = {
            "per_pair": quality_results,
            "summary": {
                "psnr_mean_db": round(np.mean(psnr_values), 3),
                "psnr_min_db": round(np.min(psnr_values), 3),
                "psnr_max_db": round(np.max(psnr_values), 3),
                "psnr_std_db": round(np.std(psnr_values), 3),
                "ssim_mean": round(np.mean(ssim_values), 5),
                "ssim_min": round(np.min(ssim_values), 5),
                "ssim_max": round(np.max(ssim_values), 5),
            },
        }
        print(f"\n  Quality Summary:")
        print(f"    PSNR: mean={all_results['quality']['summary']['psnr_mean_db']:.2f}dB, "
              f"min={all_results['quality']['summary']['psnr_min_db']:.2f}dB, "
              f"max={all_results['quality']['summary']['psnr_max_db']:.2f}dB")
        print(f"    SSIM: mean={all_results['quality']['summary']['ssim_mean']:.5f}, "
              f"min={all_results['quality']['summary']['ssim_min']:.5f}, "
              f"max={all_results['quality']['summary']['ssim_max']:.5f}")

    # ── Speed comparison ──
    if not args.skip_bf16 and not args.skip_fp8:
        print("\n" + "█" * 70)
        print("█  PHASE 4: Speed Comparison")
        print("█" * 70)
        bf16_summary = all_results["models"]["bf16"]["summary"]
        fp8_summary = all_results["models"]["fp8"]["summary"]

        denoise_diff = bf16_summary["denoise_mean_s"] - fp8_summary["denoise_mean_s"]
        wall_diff = bf16_summary["wall_time_mean_s"] - fp8_summary["wall_time_mean_s"]
        load_diff = (all_results["models"]["bf16"]["load_time_s"] -
                     all_results["models"]["fp8"]["load_time_s"])

        size_bf16 = all_results["models"]["bf16"]["model_size_mb"]
        size_fp8 = all_results["models"]["fp8"]["model_size_mb"]
        size_ratio = size_bf16 / size_fp8 if size_fp8 > 0 else 1.0

        speed_comparison = {
            "bf16_load_time_s": all_results["models"]["bf16"]["load_time_s"],
            "fp8_load_time_s": all_results["models"]["fp8"]["load_time_s"],
            "load_time_diff_s": round(load_diff, 2),
            "bf16_denoise_mean_s": bf16_summary["denoise_mean_s"],
            "fp8_denoise_mean_s": fp8_summary["denoise_mean_s"],
            "denoise_diff_s": round(denoise_diff, 3),
            "bf16_wall_mean_s": bf16_summary["wall_time_mean_s"],
            "fp8_wall_mean_s": fp8_summary["wall_time_mean_s"],
            "wall_diff_s": round(wall_diff, 3),
            "bf16_size_mb": size_bf16,
            "fp8_size_mb": size_fp8,
            "size_ratio": round(size_ratio, 2),
        }
        all_results["speed_comparison"] = speed_comparison

        print(f"  Model file size:        BF16={size_bf16:.0f}MB  FP8={size_fp8:.0f}MB  "
              f"(FP8 is {size_ratio:.1f}× smaller)")
        print(f"  Load time:              BF16={all_results['models']['bf16']['load_time_s']:.1f}s  "
              f"FP8={all_results['models']['fp8']['load_time_s']:.1f}s  "
              f"(diff={load_diff:+.1f}s)")
        print(f"  Denoise (GPU):          BF16={bf16_summary['denoise_mean_s']:.3f}s  "
              f"FP8={fp8_summary['denoise_mean_s']:.3f}s  "
              f"(diff={denoise_diff:+.3f}s)")
        print(f"  Wall time:              BF16={bf16_summary['wall_time_mean_s']:.3f}s  "
              f"FP8={fp8_summary['wall_time_mean_s']:.3f}s  "
              f"(diff={wall_diff:+.3f}s)")
        if abs(denoise_diff) < 0.01:
            print(f"  → Inference speed is essentially identical (<10ms difference)")
        elif denoise_diff > 0:
            print(f"  → FP8 is {denoise_diff*1000:.0f}ms faster in denoising")
        else:
            print(f"  → BF16 is {abs(denoise_diff)*1000:.0f}ms faster in denoising")

    # ── Save results JSON ──
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"  Results saved to: {results_path}")
    print(f"  Images saved to:  {output_dir}")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
