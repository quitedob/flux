# NVFP4 模型支持开发记录

## 项目环境

- **Python 环境**: 使用项目根目录下的 `.venv`（不是 conda！）
- **激活命令**: `F:/python/flux/.venv/Scripts/python` 或 `source .venv/Scripts/activate`
- **硬件**: RTX 5060 Ti (Blackwell, 16GB VRAM)
- **系统**: Windows 11

## 模型下载记录

所有模型文件位于 `F:/python/flux/models/`：

| 文件 | 大小 | SHA256 | 来源 |
|------|------|--------|------|
| `flux-2-klein-base-4b.safetensors` | 7.75 GB | - | HuggingFace: `black-forest-labs/FLUX.2-klein-base-4B` |
| `flux-2-klein-base-4b-nvfp4.safetensors` | 2.49 GB | `f66faefe...81ab` | ModelScope（魔搭社区） |
| `ae.safetensors` | 336 MB | - | HuggingFace: `black-forest-labs/FLUX.2-dev` |

## NVFP4 文件结构

- **总 keys**: 383（含 scale 和 metadata）
- **量化层 (uint8)**: 78 个（FP4 E2M1 packed）
- **未量化层 (BF16)**: 71 个（与 BF16 原版完全一致）
- **Block scale (float8_e4m3fn)**: 78 个（每层 1 个）
- **Global scale (float32)**: 78 个（`weight_scale_2`）
- **Input scale (float32)**: 78 个（`input_scale`，用于激活量化，未使用）
- **量化格式**: NVFP4 format_version "1.0"
- **Block size**: 16 elements
- **FP4 格式**: E2M1（1 sign + 2 exponent + 1 mantissa）

### 量化模式（哪些层被量化）

- `double_blocks.0`: qkv + mlp.0 + mlp.2（6层），proj 未量化
- `double_blocks.1-4`: qkv + proj + mlp.0 + mlp.2（每 block 8层）
- `single_blocks.0-19`: linear1 + linear2（每 block 2层）
- 总计: 6 + 4×8 + 20×2 = 78 量化层

### 未量化的层（与 BF16 完全一致）

- `img_in`, `txt_in`, `time_in` 等输入层
- `double_blocks.0` 的 img_attn.proj 和 txt_attn.proj
- 所有 RMSNorm/QKNorm scales
- 所有 Modulation 层（`double_stream_modulation_img/txt`, `single_stream_modulation`）
- `final_layer`（`adaLN_modulation` + `linear`）

## 已完成的工作

1. **NVFP4 模型文件获取**: 从 ModelScope 下载了 `flux-2-klein-base-4b-nvfp4.safetensors` (2.49GB)
2. **FP4 E2M1 格式研究**: 查明了 NVFP4 的位布局和查找表
3. **反量化实现**: 在 `src/flux2/util.py` 中实现了完整的 NVFP4 加载：
   - `_FP4_E2M1_LOOKUP`: 16 项 FP4 值查找表
   - `_unpack_fp4_e2m1()`: uint8 解包为 float32（高 nibble→偶数位，低 nibble→奇数位）
   - `load_nvfp4_model()`: 完整加载流程（解包 + 块级缩放 + 全局缩放）
   - `load_flow_model()`: 添加了 `load_fn` 分发逻辑
4. **权重形状验证**: 所有 149 个权重 key 形状匹配，`strict=True` 加载成功
5. **反量化正确性验证**:
   - 单层 Linear 手动计算 vs 模型 forward: max_diff=0.0 ✓
   - 非量化权重与 BF16 原版: max_diff=0.0 ✓
   - 量化权重与 BF16 原版平均相关系数: **0.9567**（范围 0.79-0.99）
6. **环境变量正确配置**: NVFP4 模型加载使用 `KLEIN_4B_BASE_MODEL_PATH` 指向正确的 NVFP4 文件
7. **文本编码器本地化**: Qwen3-4B-FP8 使用 `local_files_only=True`，禁止联网

## 当前状态：NVFP4 生成失败

### 测试结果

NVFP4 反量化模型产出**纯色纹理（橄榄绿/棕色）**，而非可识别图像：

| 步数 | guidance | 输出 |
|------|----------|------|
| 50 | 4.0 | 纯棕色纹理 |
| 50 | 2.0 | 纯棕色纹理 |
| 50 | 1.5 | 纯棕色纹理 |
| 50 | 1.0 | 纯棕色纹理 |
| 25 | 4.0 | 纯橄榄绿纹理 |
| 10 | 4.0 | 纯橄榄绿纹理 |
| 4 | 4.0 | 纯橄榄绿纹理 |

BF16 原版模型（相同 prompt/seed/参数）产出**正常猫照片**。

### 根因分析

NVFP4 checkpoint **设计用于 W4A4/W4A8 专用推理引擎**（如 Nunchaku SVDQuant 或 TensorRT-LLM visual_gen），而非简单的"反量化到 BF16，跑标准推理"。

专用引擎的关键特性（我们缺少的）：
1. **激活量化 (FP8)**: 运行时将激活量化到 FP8，与 FP4 权重配合使用
2. **SVDQuant 低秩补偿**: 用量化误差的低秩分解来补偿精度损失
3. **自定义 CUDA kernel**: W4A4/W4A8 矩阵乘法 kernel，直接在量化域计算
4. **SmoothQuant**: 激活平滑因子

我们目前的 W4A16（反量化到 BF16）方案丢失了激活量化带来的精度优势，FP4 量化误差通过 78 层累积传播，导致输出完全退化。

### 为什么权重相关性 0.96 仍不够

- 78 个量化层 × 每层 0.85-0.99 相关性
- 注意力机制中的 softmax 放大 Q/K 投影误差
- CFG (guidance=4.0) 进一步放大条件/无条件预测的差异
- 扩散模型对预测精度高度敏感

## 可能的解决方向

1. **使用 Nunchaku + ComfyUI**: 官方的 NVFP4 推理路径，支持 RTX 5060 Ti
2. **使用 BF16 原版模型**: 7.75GB VRAM，已验证可正常生成
3. **等待官方 FLUX.2 推理代码更新**: Black Forest Labs 可能在后续版本中添加 NVFP4 原生支持

## 错误记录与经验教训

### 错误 1: 使用了 conda 而非 .venv
- **表现**: 尝试用 `conda run -n flux` 运行 Python
- **教训**: **首先检查项目中的 `.venv`**

### 错误 2: 模型下载到 C 盘
- **教训**: 设置环境变量让模型下载到项目本地

### 错误 3: FP4 nibble 高低位顺序写反
- **表现**: 反量化权重与 BF16 相关性仅 0.001
- **修复**: 高 nibble→偶数位，低 nibble→奇数位
- **验证**: 相关性从 0.001 提升到 0.9567

### 错误 4: Python 脚本忘记 import os
- **教训**: inline Python 脚本注意导入所有用到的模块

### 错误 5: NVFP4 模型加载了 BF16 权重（本次发现）
- **表现**: `KLEIN_4B_BASE_MODEL_PATH` 指向 BF16 文件而非 NVFP4 文件
- **修复**: 运行时设置 `os.environ['KLEIN_4B_BASE_MODEL_PATH'] = '...nvfp4.safetensors'`
- **教训**: NVFP4 和 BF16 模型共用同一个环境变量名

## 关键代码位置

- `src/flux2/util.py`: NVFP4 加载逻辑（第 15-100 行）
- `src/flux2/model.py`: 模型架构定义（`Klein4BParams` 等）
- `scripts/cli.py`: CLI 生成入口
- `scripts/test_nvfp4_gen.py`: NVFP4 生成测试脚本
- `scripts/diagnose_divergence.py`: 反量化诊断脚本（OOM 风险）
- `models/flux-2-klein-base-4b-nvfp4.safetensors`: NVFP4 量化权重 (SHA256: f66faefe...81ab)
- `models/flux-2-klein-base-4b.safetensors`: BF16 原版权重
