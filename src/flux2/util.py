import base64
import io
import os

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import load_mistral_small_embedder, load_qwen3_embedder

# FP4 E2M1 lookup table: maps 4-bit nibble → float32 value
# Format: 1 sign | 2 exponent | 1 mantissa, values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
_FP4_E2M1_LOOKUP = torch.zeros(16, dtype=torch.float32)
for _nibble in range(16):
    _sign = -1.0 if (_nibble & 0x8) else 1.0
    _exp = (_nibble >> 1) & 0x3
    _mant = _nibble & 0x1
    if _exp == 0:
        _val = 0.0 if _mant == 0 else 0.5
    else:
        _val = (1.0 + _mant * 0.5) * (2.0 ** (_exp - 1))
    _FP4_E2M1_LOOKUP[_nibble] = _sign * _val


def _unpack_fp4_e2m1(uint8_weight: torch.Tensor) -> torch.Tensor:
    """Unpack NVFP4 packed uint8 [M, K] → float32 [M, 2*K].

    Each uint8 byte stores two FP4 E2M1 values (low/high nibbles).
    Block size is 16 elements (8 packed bytes) per scale group.
    """
    M, K = uint8_weight.shape
    low = uint8_weight & 0x0F
    high = (uint8_weight >> 4) & 0x0F
    lookup = _FP4_E2M1_LOOKUP.to(uint8_weight.device)
    result = torch.empty(M, 2 * K, dtype=torch.float32, device=uint8_weight.device)
    result[:, 0::2] = lookup[high.to(torch.int64)]
    result[:, 1::2] = lookup[low.to(torch.int64)]
    return result


def _patch_fp8_linears(model: torch.nn.Module) -> None:
    """Patch nn.Linear layers with FP8 weights to use torch._scaled_mm (true FP8 matmul).

    Keeps weights as FP8 in GPU memory. During forward, quantizes input activations
    to FP8 and uses _scaled_mm which runs on FP8 Tensor Cores (RTX 40xx+).
    """
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                ws = getattr(module, "weight_scale", None)
                if ws is None:
                    continue

                def make_fp8_forward(lin, w_scale):
                    # Cache the "one" scale tensor — same device as the weight
                    _one = torch.tensor([[1.0]], device=lin.weight.device, dtype=torch.float32)

                    @torch.no_grad()
                    def fp8_forward(inp):
                        # Flatten batch dims for matmul
                        orig_shape = inp.shape
                        inp_2d = inp.reshape(-1, orig_shape[-1])  # (M, K)

                        # Per-row quantization: scale per row for better precision
                        inp_amax = inp_2d.abs().max(dim=-1, keepdim=True)[0].float()
                        inp_scale = (inp_amax / 240.0).clamp(min=1e-12)  # (M, 1)
                        inp_fp8 = (inp_2d.float() / inp_scale).clamp(-240, 240).to(torch.float8_e4m3fn)

                        # FP8 matmul via Tensor Cores (scale=1.0, per-row compensation after)
                        # weight is (out, in) row-major → .t() gives (in, out) col-major
                        out_2d = torch._scaled_mm(
                            inp_fp8, lin.weight.t(),
                            _one, _one,
                            out_dtype=torch.float32,
                        )
                        # Per-row compensation: multiply back the row + column scales
                        out_2d = out_2d * inp_scale * w_scale.unsqueeze(0)
                        out_2d = out_2d.to(torch.bfloat16)

                        out = out_2d.reshape(*orig_shape[:-1], -1)
                        if lin.bias is not None:
                            out = out + lin.bias
                        return out
                    return fp8_forward

                module.forward = make_fp8_forward(module, ws)


def load_nvfp4_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    """Load an NVFP4-quantized FLUX.2 model with on-the-fly dequantization."""
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if config["model_path"] in os.environ:
            weight_path = os.environ[config["model_path"]]
            assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
        else:
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                raise RuntimeError(
                    f"Failed to access the model repository: {config['repo_id']}. "
                    "Please check your internet connection and make sure you have access."
                )

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(config["params"]).to(torch.bfloat16)

        print(f"Loading NVFP4 quantized model from {weight_path}")
        raw_sd = load_sft(weight_path, device=str(device))

        sd = {}
        for k, v in raw_sd.items():
            if k.endswith((".input_scale", ".weight_scale", ".weight_scale_2")):
                continue
            if v.dtype == torch.uint8:
                # Unpack FP4 E2M1 packed weight, then apply block-wise + global scales
                prefix = k.rsplit(".", 1)[0]
                ws = raw_sd[f"{prefix}.weight_scale"].float()
                ws2 = raw_sd.get(f"{prefix}.weight_scale_2", torch.tensor(1.0, device=device)).float()
                blk_size = (v.shape[1] * 2) // ws.shape[1]  # should be 16
                ws_expanded = ws.repeat_interleave(blk_size, dim=1)
                unpacked = _unpack_fp4_e2m1(v)
                sd[k] = (unpacked * ws_expanded * ws2).to(torch.bfloat16)
            elif v.dtype == torch.float8_e4m3fn:
                sd[k] = v.float().to(torch.bfloat16)
            else:
                sd[k] = v

        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(config["params"]).to(torch.bfloat16)


FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b-fp8": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9b-fp8",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b-fp8.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_FP8_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b-kv": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B-kv",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b-kv.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_KV_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
        "use_kv_cache": True,
    },
    "flux.2-klein-base-4b-nvfp4": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-4b-nvfp4.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
        "load_fn": load_nvfp4_model,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    # Route to custom loader if specified (e.g., NVFP4)
    if "load_fn" in config:
        return config["load_fn"](model_name, debug_mode=debug_mode, device=device)

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if config["model_path"] in os.environ:
            weight_path = os.environ[config["model_path"]]
            assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
        else:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                raise RuntimeError(
                    f"Failed to access the model repository: {config['repo_id']}. "
                    "Please check your internet connection and make sure you have access."
                )

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(config["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        raw_sd = load_sft(weight_path, device=str(device))

        # Separate FP8 weights and scales from regular params
        scale_keys = set()
        fp8_keys = set()
        for k, v in raw_sd.items():
            if k.endswith(".input_scale") or k.endswith(".weight_scale"):
                scale_keys.add(k)
            elif v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                fp8_keys.add(k)

        if fp8_keys:
            # FP8 model: keep weights as FP8 in GPU, register scales, patch Linears
            sd = {k: v for k, v in raw_sd.items() if k not in scale_keys}
            model.load_state_dict(sd, strict=True, assign=True)

            # Register weight_scale buffers on parent modules for runtime dequantization
            for fp8_k in fp8_keys:
                prefix = fp8_k.rsplit(".", 1)[0]
                ws_key = f"{prefix}.weight_scale"
                if ws_key in raw_sd:
                    ws_val = raw_sd[ws_key].float().to(device)
                    *path, _leaf = fp8_k.split(".")
                    mod = model
                    for part in path:
                        mod = getattr(mod, part)
                    mod.register_buffer("weight_scale", ws_val, persistent=False)

            # Patch nn.Linear layers to dequantize FP8 weights on-the-fly
            _patch_fp8_linears(model)
        else:
            # BF16 model: just filter scale keys and load
            sd = {k: v for k, v in raw_sd.items() if k not in scale_keys}
            model.load_state_dict(sd, strict=True, assign=True)

        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda"):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device)


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if "AE_MODEL_PATH" in os.environ:
        weight_path = os.environ["AE_MODEL_PATH"]
        assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
    else:
        # download from huggingface
        try:
            ae_repo = config.get("ae_repo_id", config["repo_id"])
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=ae_repo,
                filename=config["filename_ae"],
                repo_type="model",
            )
        except huggingface_hub.errors.RepositoryNotFoundError:
            print(
                f"Failed to access the model repository. Please check your internet "
                f"connection and make sure you've access to {config['repo_id']}."
                "Stopping."
            )
            raise RuntimeError(
                f"Failed to access the model repository: {config['repo_id']}. "
                "Please check your internet connection and make sure you have access."
            )

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)

    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
