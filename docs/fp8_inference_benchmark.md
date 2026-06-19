# FLUX.2 Klein — FP8 推理方案

## 环境

- GPU: RTX 4080 (SM89, 16GB VRAM)
- CUDA: 13.0, PyTorch: 2.12.1+cu130
- triton-windows: 3.7.0.post26

## 模型清单

| 模型 | 文件大小 | 权重精度 | 112 FP8 Linear | 类型 |
|------|---------|---------|:---:|------|
| `flux.2-klein-4b-fp8` | 3.8 GB | FP8 E4M3 | ✅ | 蒸馏 (4步, guidance=1) |
| `flux.2-klein-9b-fp8` | 8.8 GB | FP8 E4M3 | ✅ | 蒸馏 (4步, guidance=1) |
| `flux.2-klein-base-9b` | 18.2 GB | BF16 | ❌ | Base (50步, guidance=4) |

## 推理管线

### CPU Offload 策略（手动实现）

RTX 4080 16GB 同时装不下 AE + TE + Model，按阶段手动交换：

| 阶段 | GPU | CPU | VRAM |
|------|-----|-----|------|
| 加载 Model | — | Model (FP8) | 0 GB |
| 加载 AE+TE | AE + TE | Model | ~8–12 GB |
| 文本编码 | AE + TE | Model | ~8–12 GB |
| 编码完成 | AE + Model | **TE** | ~4–10 GB |
| 去噪 | AE + Model | TE | ~4–10 GB |
| 解码 | AE | TE + Model | ~0.3 GB |

关键代码：
```python
# 编码后立即卸 TE 到 CPU，腾出空间给 Model
text_encoder = text_encoder.cpu()
torch.cuda.empty_cache()
model = model.to(device)
```

### FP8 Transformer 推理（方案 3b：per-row）

```python
# _patch_fp8_linears() 替换 nn.Linear.forward:
# 1. Per-row 动态量化输入
sa = inp.abs().max(dim=-1)[0] / 240  # (M, 1)
inp_fp8 = (inp / sa).to(float8_e4m3fn)

# 2. FP8 Tensor Core matmul（scale=1.0）
out = torch._scaled_mm(inp_fp8, weight.t(), one, one, out_dtype=float32)

# 3. 行列 scale 补偿
out = out * sa * w_scale.unsqueeze(0)
```

权重以 FP8 保留在 GPU（不反量化），matmul 走 FP8 Tensor Core。

## 4B vs 9B 实测对比

相同 prompt、seed、1024×1024、4 步去噪：

| 指标 | Klein 4B FP8 | Klein 9B FP8 | 倍率 |
|------|:-----------:|:-----------:|:----:|
| 模型权重文件 | 3.8 GB | 8.8 GB | 2.3× |
| VRAM (AE+TE) | 8.38 GB | 12.27 GB | 0.68× |
| VRAM (AE+Model) | 4.43 GB | 9.79 GB | 0.45× |
| 去噪时间 (4步) | 2.79s (697ms/步) | 4.85s (1213ms/步) | 0.57× |
| 总耗时 | 13.6s | 33.8s | 0.41× |
| **峰值 VRAM** | **8.73 GB** | **14.09 GB** | 0.62× |

### 为什么 4B 编码器比 9B 小？

| | Qwen3-4B-FP8 | Qwen3-8B-FP8 |
|--|:--:|:--:|
| 参数量 | 4B | 8B |
| hidden_size | 2560 | 4096 |
| 加载精度 | BF16 | FP8 (原生量化器) |
| GPU 占用 | ~8 GB | ~12 GB |

4B 用 BF16（8GB），8B 用原生 FP8 量化器（12GB）。8B 虽然参数翻倍，但 FP8 存储 + 原生量化器更高效，实际只多 50%。

## Bug 修复记录

### Bug 1: `input_scale` 误乘（已修复）

```python
# Bug: input_scale 是激活量化 scale，权重反量化不需要
v = (v.float() * input_scale * weight_scale).to(torch.bfloat16)
# 正确: 只乘 weight_scale
v = (v.float() * weight_scale).to(torch.bfloat16)
```

多乘导致 PSNR 仅 10.46 dB。修复后走方案 3b（per-row scaled_mm），彻底不再用 input_scale。

### Bug 2: Qwen3-4B-FP8 编码器加载异常（已修复）

**根因**: `models/Qwen3-4B-FP8/config.json` 中 `"quantization_config": "none"` 是无效字符串值，导致 transformers 以非量化方式加载，4B 参数模型占了 **16.44 GB**。

**修复**:
1. `config.json` — 删除无效字段 `quantization_config`
2. `text_encoder.py` — `load_qwen3_embedder` 按 variant 区分 dtype：
   ```python
   load_dtype = torch.bfloat16 if variant == "4B" else None  # 8B 走原生 FP8
   ```

| | 修复前 | 修复后 |
|--|:--:|:--:|
| 4B TE 显存 | 16.44 GB | **8.38 GB** |
| 4B 峰值显存 | 16.99 GB | **8.73 GB** |
| 4B 总耗时 | 26.9s | **13.6s** |

### Bug 3: util.py 代码质量修复

| 问题 | 修复 |
|------|------|
| `sys.exit(1)` 3 处 | → `raise RuntimeError(...)` |
| 重复 config 查找 | → 用局部变量 `config` |
| 未使用变量 `_name` | → `_` |
| 缺少 `@torch.no_grad()` | → 添加装饰器 |
| 每次 forward 创建 `one` 张量 | → 闭包内缓存 |

## 测试脚本

```bash
# Klein 4B FP8
PYTHONPATH=src python scripts/test_fp8_gen.py --prompt "a cat on a table" --steps 4

# Klein 9B FP8
PYTHONPATH=src python scripts/test_fp8_9b_gen.py --prompt "a cat on a table" --steps 4
```

环境变量：
```bash
export AE_MODEL_PATH=models/vae/ae.safetensors
export KLEIN_4B_MODEL_PATH=models/flux-2-kelin-4b-fp8/flux-2-klein-4b-fp8.safetensors
export KLEIN_9B_FP8_MODEL_PATH=models/flux-2-kelin-9b-fp8/flux-2-klein-9b-fp8.safetensors
```

## 已知限制

1. **SM89 不支持原生 per-row scaling**: PyTorch 未实现 SM89 per-row，需手动补偿
2. **9B 峰值 14.09 GB**: 16GB 显卡刚好够，8B 编码器 + 9B Transformer 在小分辨率下可行
3. **numpy 2.x 兼容**: 升级 PyTorch cu130 后部分库（transformers/sklearn）有警告但不影响推理
4. **Windows triton**: 需用 `triton-windows` 替代 `triton`

## 参考

- [FLUX.2 Klein 4B FP8 on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)
- [FLUX.2 Klein 9B FP8 on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)
- [BFL flux2 GitHub](https://github.com/black-forest-labs/flux2)
- [PyTorch scaled_mm issue #130359](https://github.com/pytorch/pytorch/issues/130359)
