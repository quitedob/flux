# FLUX.2 模型训练深度研究报告

## 一、模型版本与显存需求

### 1.1 各版本对比

| 模型版本 | 参数量 | License | 推理显存 | LoRA 训练显存 | 文本编码器 |
|----------|--------|---------|---------|-------------|-----------|
| FLUX.2 [dev] | 32B | Apache 2.0 | ~20-24GB (bf16) | 40GB+ | Mistral Small 3.1 (24B) |
| FLUX.2 Klein 9B Base | 9B | FLUX NCL | ~21.7GB (bf16) | 22-28GB | Qwen3-8B |
| FLUX.2 Klein 4B Base | 4B | Apache 2.0 | ~9.2GB (bf16) | 12-18GB | Qwen3-4B |

### 1.2 蒸馏 vs 基础模型

- **Distilled (蒸馏版)**: `flux.2-klein-4b`, `flux.2-klein-9b` — 只需 4 步推理 (guidance=1.0)，但**不能用于训练**
- **Base (基础版)**: `flux.2-klein-base-4b`, `flux.2-klein-base-9b` — 需要 ~50 步推理 (guidance≈4)，**适合 LoRA 训练**

### 1.3 RTX 4080 16GB 可行性

| 任务 | 可行性 | 说明 |
|------|--------|------|
| FLUX.2 [dev] 推理 | ✅ | 使用 4-bit 量化 (~18GB) 或 remote text encoder |
| FLUX.2 [dev] 训练 | ❌ | 需要 40GB+ |
| Klein 9B 推理 | ✅ | FP8 可降至 ~14GB |
| Klein 9B 训练 | ⚠️ | 需要重度量化 + CPU offload |
| Klein 4B 推理 | ✅ | FP8 可降至 ~6-8GB |
| Klein 4B 训练 | ✅ | 推荐方案，int8/FP8 后 ~13-15GB |

---

## 二、训练框架支持情况

### 2.1 框架总览

| 框架 | FLUX.2 [dev] | Klein 4B | Klein 9B | 16GB 显存优化 | 入门难度 |
|------|-------------|----------|----------|-------------|---------|
| [AI-Toolkit](https://github.com/ostris/ai-toolkit) | ✅ | ✅ | ✅ | int8 + low_vram | ⭐⭐ |
| [SimpleTuner](https://github.com/bghira/SimpleTuner) | ✅ | ✅ | ✅ | int8-quanto + cpu | ⭐⭐⭐ |
| [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | ✅ | ✅ | ✅ | Adafactor + fp8 | ⭐⭐ |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | ✅ | ✅ | ✅ | Split Training + disk | ⭐⭐ |
| [diffusers](https://github.com/huggingface/diffusers) | ✅ | ✅ | ✅ | 标准化量化 | ⭐ |

### 2.2 推荐选择路径

- **新手入门**: AI-Toolkit（配置简单，迭代快 20-30%）
- **生产级质量**: SimpleTuner（最稳定，配置选项最多）
- **中文生态**: DiffSynth-Studio（ModelScope 出品，磁盘 offload）
- **GUI 用户**: musubi-tuner（kohya-ss 系，Adafactor 优化器）
- **标准流程**: diffusers（HuggingFace 官方）

---

## 三、RTX 4080 16GB 训练方案

### 3.1 Klein 4B Base — SimpleTuner 配置

```bash
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto --quantize_via=cpu"
export OPTIMIZER="adamw8bit"
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export USE_GRADIENT_CHECKPOINTING=true
export RESOLUTION=1024
# 预期显存: ~13-15GB, 训练时间: 2-4 小时
```

### 3.2 Klein 4B Base — AI-Toolkit 配置

```yaml
model:
  arch: "flux2_klein_4b"
  quantize: true
  qtype: "qfloat8"
  low_vram: true

network:
  type: "lora"
  linear: 16
  linear_alpha: 16

train:
  batch_size: 1
  lr: 0.0001
  optimizer: "adamw8bit"
  timestep_type: "sigmoid"
  steps: 2000

sample:
  guidance_scale: 4
  sample_steps: 50  # Base 模型必须~50步
```

### 3.3 16GB 显存优化技巧

| 技巧 | 显存节省 | 代价 |
|------|---------|------|
| FP8/int8 量化 | 40-50% | 质量轻微下降 |
| Gradient Checkpointing | 30-40% | 训练慢 20-30% |
| CPU Quantize (SimpleTuner) | 避免加载时 OOM | 初始化多 60 秒 |
| Adafactor 优化器 | 10-15% | 收敛稍慢 |
| 低 Rank (8 vs 16) | 5-10% | 表达能力下降 |
| 512px 起步训练 | 30-40% | 细节损失 |

---

## 四、云端 GPU 训练成本

### 4.1 GPU 租赁价格 (2026年5月)

| GPU | VRAM | Vast.ai | RunPod | AutoDL |
|-----|------|---------|--------|--------|
| RTX 4090 | 24GB | $0.40-0.80/h | $0.49-0.79/h | ¥3-5/h |
| A100 40GB | 40GB | $3-5/h | $1.64/h | ¥15/h |
| A100 80GB | 80GB | ~$5/h | $1.89/h | ¥18-22/h |

### 4.2 fal.ai 训练服务

| Trainer | 价格/步 | 1000步 | 3000步 |
|---------|---------|--------|--------|
| Klein 4B Base | $0.005 | $5.00 | $15.00 |
| FLUX.2 [dev] | $0.008 | $8.00 | $24.00 |
| FLUX.2 [dev] V2 | $0.0255 | $25.50 | $76.50 |

### 4.3 性价比分析

| 方案 | 预估成本 | 推荐度 |
|------|---------|--------|
| 本地 RTX 4080 + Klein 4B | 电费 ~$1 | ⭐⭐⭐⭐⭐ |
| Vast.ai RTX 4090 + Klein 9B | $1.50-2.50 | ⭐⭐⭐⭐ |
| fal.ai Klein 4B Trainer | $10-15 | ⭐⭐⭐⭐ |
| Vast.ai A100 + FLUX.2 [dev] | $10-20 | ⭐⭐⭐ |
| fal.ai FLUX.2 [dev] Trainer | $24 | ⭐⭐⭐ |

---

## 五、数据集最佳实践

- **数量**: 20-50 张高质量图（质量 > 数量）
- **分辨率**: 1024x1024 起步
- **Caption**: VLM 自动标注 + 人工审核
- **Trigger word**: 独特触发词，自然嵌入 caption
- **多样性**: 不同角度、光照、背景

### 训练步骤参考

| LoRA 类型 | 推荐步数 | 学习率 |
|-----------|---------|--------|
| 风格 LoRA | 1500-2500 | 8e-5 ~ 1e-4 |
| 角色 LoRA | 1500-3000 | 1e-4 |
| 概念 LoRA | 2000-4000 | 1e-4 ~ 5e-5 |

---

## 六、关键注意事项

1. **必须使用 Base 模型训练**，蒸馏版只适合推理
2. **Klein 4B Base 是 Apache 2.0**，比 FLUX.1 更开放
3. **4B 和 9B 的 LoRA 互不兼容**，不可跨尺寸使用
4. **9B 训练可能崩坏**，建议用更保守的 LR (5e-5) 和较低的 Rank (16)
5. **FP8 量化对质量影响最小**，是推理和训练的首选精度
6. **SimpleTuner 16GB 关键**: 必须用 `--quantize_via=cpu`，否则量化前加载完整 bf16 模型会 OOM
