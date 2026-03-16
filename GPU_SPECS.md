# GPU Specifications & Experiment Setups

This document covers the hardware requirements for reproducing all Parallax experiments, from lightweight development to full paper-grade evaluation.

---

## 1. Model VRAM Footprint

### 1.1 Base Video Model

| Model | Parameters | dtype | VRAM (weights only) | VRAM (inference) | VRAM (+ backprop) |
|-------|-----------|-------|---------------------|------------------|-------------------|
| Wan2.1-I2V-14B-480P | 14B | bf16 | ~28 GB | ~35 GB | ~55 GB |
| Wan2.1-I2V-14B-480P | 14B | fp8 (quantized) | ~14 GB | ~20 GB | ~35 GB |
| Wan2.1-I2V-1.3B-480P | 1.3B | bf16 | ~2.6 GB | ~5 GB | ~10 GB |

> **Note:** The 14B model is the target for paper experiments. The 1.3B model can be used for rapid prototyping and ablation sweeps.

### 1.2 Guidance Models

| Model | Parameters | VRAM (bf16) | VRAM (fp32) | Notes |
|-------|-----------|-------------|-------------|-------|
| Depth-Anything V2 Small | 24M | ~48 MB | ~96 MB | Negligible |
| Depth-Anything V2 Large | 335M | ~670 MB | ~1.3 GB | Higher quality, optional |
| DINOv2-Small | 22M | ~44 MB | ~88 MB | Negligible |
| DINOv2-Large | 304M | ~608 MB | ~1.2 GB | Optional upgrade |
| SAM2-Hiera-Small | 38M | ~76 MB | ~152 MB | Negligible |
| RAFT-Small | 1M | ~2 MB | ~4 MB | Negligible |
| RAFT-Large | 5M | ~10 MB | ~20 MB | Optional |

### 1.3 Wan2.1 3D VAE (AutoencoderKLWan)

| Component | VRAM (fp32) | Notes |
|-----------|-------------|-------|
| VAE encoder | ~1.2 GB | Used once for conditioning image |
| VAE decoder | ~1.2 GB | Called differentiably every guided step |
| VAE decode (full 33 frames) | ~8-12 GB | Peak activation memory |
| VAE decode (subsampled, every 4th) | ~2-4 GB | Our default configuration |

### 1.4 Total Peak VRAM (Guidance Active)

The peak occurs during `_apply_guidance_step()` when we hold:
- Base DiT weights + KV cache
- VAE decoder weights + activations (for backprop)
- Guidance model weights + activations (for backprop)
- Latent tensor + gradient
- Decoded frames tensor (subsampled)

---

## 2. Experiment Tiers

### Tier A: Development & Debugging

**Purpose:** Rapid iteration, test new guidance modules, verify gradient flow.

| Spec | Requirement |
|------|-------------|
| GPU | 1x NVIDIA RTX 4090 (24 GB) or A5000 (24 GB) |
| Model | Wan2.1-I2V-**1.3B**-480P (bf16) |
| Resolution | 480x320 |
| Frames | 17 |
| Guidance | 1 module at a time |
| Subsample rate | 4 (decode every 4th frame) |
| Steps | 20 |

**VRAM breakdown:**
```
Base model (1.3B, bf16)      ~2.6 GB
VAE decoder (fp32)           ~1.2 GB
Guidance model (small)       ~0.1 GB
Activations + grads          ~4-6 GB
Latents + decoded frames     ~2-3 GB
─────────────────────────────────────
Total peak                   ~10-14 GB   ✓ fits 24 GB
```

**Estimated time per video:** 2-4 minutes

---

### Tier B: Single-Signal Experiments

**Purpose:** Full-quality single-guidance ablations (depth, normal, semantic, flow individually).

| Spec | Requirement |
|------|-------------|
| GPU | 1x NVIDIA A100 (80 GB) or H100 (80 GB) |
| Model | Wan2.1-I2V-**14B**-480P (bf16) |
| Resolution | 480x320 (480P) |
| Frames | 33 |
| Guidance | 1 module |
| Subsample rate | 4 |
| Steps | 30 |

**VRAM breakdown:**
```
Base model (14B, bf16)       ~28 GB
VAE decoder (fp32)           ~1.2 GB
Guidance model (small)       ~0.1 GB
Activations + grads          ~15-20 GB
Latents + decoded frames     ~5-8 GB
─────────────────────────────────────
Total peak                   ~50-58 GB   ✓ fits 80 GB
```

**Estimated time per video:** 8-15 minutes

---

### Tier C: Multi-Signal & Adaptive Composition

**Purpose:** Composite guidance experiments (depth+normal, depth+semantic+flow, full 5-signal).

| Spec | Requirement |
|------|-------------|
| GPU | 1x NVIDIA H100 (80 GB) or 2x A100 (80 GB each) |
| Model | Wan2.1-I2V-14B-480P (bf16) |
| Resolution | 480x320 |
| Frames | 33 |
| Guidance | 2-5 modules simultaneously |
| Subsample rate | 4-8 (increase for more modules) |
| Steps | 30 |

**VRAM breakdown (3 modules: depth + normal + semantic):**
```
Base model (14B, bf16)       ~28 GB
VAE decoder (fp32)           ~1.2 GB
3x guidance models           ~0.3 GB
Activations + grads (3x)     ~25-35 GB
Latents + decoded frames     ~5-8 GB
─────────────────────────────────────
Total peak                   ~60-73 GB   ✓ fits 80 GB (tight)
```

**With 5 modules (all signals):**
- Increase `frame_subsample_rate` to 8
- Use gradient checkpointing on VAE decoder
- Sequential (not parallel) guidance module evaluation
- **Peak:** ~65-78 GB on single H100

**Estimated time per video:** 15-30 minutes

---

### Tier D: Paper-Grade Evaluation Suite

**Purpose:** Full VBench++ I2V evaluation, ablation grids, human eval dataset generation.

| Spec | Requirement |
|------|-------------|
| GPU | 4-8x NVIDIA H100 (80 GB) or equivalent |
| Model | Wan2.1-I2V-14B-480P (bf16) |
| Resolution | 480x320 |
| Frames | 33 |
| Guidance | All configurations |
| Parallelism | Data-parallel across GPUs |

**Experiment matrix:**

| Experiment | # Videos | Est. GPU-hours |
|-----------|----------|----------------|
| Single-signal ablation (5 signals x 100 prompts) | 500 | ~80 |
| Pairwise composition (10 pairs x 50 prompts) | 500 | ~120 |
| Full composition (50 prompts) | 50 | ~25 |
| Guidance strength sweep (5 values x 3 signals x 30 prompts) | 450 | ~75 |
| Temporal scheduling ablation (4 schedules x 3 signals x 30 prompts) | 360 | ~60 |
| Unguided baselines (100 prompts) | 100 | ~8 |
| VBench++ I2V suite (~900 prompts) | 900 | ~75 |
| **Total** | **~2860** | **~443** |

**Cluster recommendation:** 8x H100 for ~55 hours, or 4x H100 for ~111 hours.

---

## 3. Memory Optimization Strategies

These strategies reduce VRAM at the cost of speed. Apply as needed:

### 3.1 Frame Subsampling (Default: ON)

```python
GuidanceConfig(frame_subsample_rate=4)  # decode every 4th latent frame
# 33 frames → 8-9 decoded frames → ~4x memory reduction on VAE decode
```

| Subsample rate | Frames decoded (of 33) | VAE memory | Quality impact |
|---------------|----------------------|------------|---------------|
| 1 | 33 | 100% | Best |
| 2 | 17 | ~50% | Minimal |
| 4 (default) | 9 | ~25% | Acceptable |
| 8 | 5 | ~12% | Noticeable (use for 5-signal) |

### 3.2 Gradient Checkpointing

Enable gradient checkpointing on the VAE decoder to trade compute for memory:

```python
pipe.vae.enable_gradient_checkpointing()
# ~40% activation memory reduction, ~30% slower
```

### 3.3 Model Offloading

For single-GPU setups with limited VRAM, offload inactive guidance models to CPU:

```python
# diffusers sequential CPU offload for base model
pipe.enable_sequential_cpu_offload()
# ~50% VRAM reduction on base model, ~2x slower
```

### 3.4 Quantization (Base Model)

Quantize the 14B DiT to reduce weight memory:

```python
from diffusers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# 14B model: 28 GB → 14 GB weights
```

### 3.5 Mixed Precision Guidance

Run guidance models in fp16 while keeping VAE decode in fp32:

```python
GuidanceConfig(decode_dtype=torch.float32)  # VAE in fp32 for stability
# Guidance models auto-cast to bf16 via load_model(dtype=torch.bfloat16)
```

### 3.6 Sequential Module Evaluation

For multi-guidance, evaluate one module at a time and accumulate gradients:

```python
# AdaptiveCompositeGuidance already does this by default:
# each module's gradient is computed independently via torch.autograd.grad()
# only one module's activations are alive at a time
```

---

## 4. Recommended Cloud Setups

### 4.1 Development (Tier A)

| Provider | Instance | GPU | VRAM | $/hr |
|----------|----------|-----|------|------|
| Lambda Labs | gpu_1x_rtx4090 | 1x RTX 4090 | 24 GB | ~$0.50 |
| RunPod | RTX 4090 | 1x RTX 4090 | 24 GB | ~$0.44 |
| Vast.ai | RTX 4090 | 1x RTX 4090 | 24 GB | ~$0.30 |

**Cost for dev cycle:** ~$5-15/day

### 4.2 Experiments (Tier B-C)

| Provider | Instance | GPU | VRAM | $/hr |
|----------|----------|-----|------|------|
| Lambda Labs | gpu_1x_a100_sxm4 | 1x A100 80GB | 80 GB | ~$1.10 |
| RunPod | A100 80GB | 1x A100 80GB | 80 GB | ~$1.64 |
| GCP | a2-highgpu-1g | 1x A100 80GB | 80 GB | ~$3.67 |
| AWS | p4d.24xlarge | 8x A100 40GB | 320 GB | ~$32.77 |

**Cost for single-signal ablation (80 GPU-hrs):** ~$90-290

### 4.3 Full Paper Eval (Tier D)

| Provider | Instance | GPU | VRAM | $/hr |
|----------|----------|-----|------|------|
| Lambda Labs | gpu_8x_h100_sxm5 | 8x H100 80GB | 640 GB | ~$23.92 |
| GCP | a3-highgpu-8g | 8x H100 80GB | 640 GB | ~$31.22 |
| CoreWeave | h100_sxm5_8 | 8x H100 80GB | 640 GB | ~$27.28 |

**Cost for full eval suite (443 GPU-hrs):** ~$1,200-1,700

---

## 5. Minimum Requirements Summary

| Experiment | Min GPU | Min VRAM | Base Model | Est. Cost |
|-----------|---------|----------|------------|-----------|
| Dev / debug | 1x RTX 4090 | 24 GB | 1.3B | ~$50 total |
| Single-signal (paper) | 1x A100 80GB | 80 GB | 14B | ~$200 |
| Multi-signal (paper) | 1x H100 80GB | 80 GB | 14B | ~$400 |
| Full eval suite | 4-8x H100 80GB | 320-640 GB | 14B | ~$1,500 |
| **Total for paper** | — | — | — | **~$2,000-2,500** |

---

## 6. Software Stack

```
Python          >= 3.10
PyTorch         >= 2.1.0 (CUDA 12.1+)
torchvision     >= 0.16.0
diffusers       >= 0.32.0
transformers    >= 4.47.0
accelerate      >= 0.25.0
CUDA            >= 12.1
cuDNN           >= 8.9
Driver          >= 535.x (for H100), >= 525.x (for A100)
```

### Docker (recommended)

```bash
# PyTorch NGC container (includes CUDA, cuDNN, NCCL)
docker pull nvcr.io/nvidia/pytorch:24.12-py3

# Inside container
pip install -e ".[all]"
```

---

## 7. Profiling & Monitoring

### Peak VRAM measurement

```python
# Add to any experiment script
torch.cuda.reset_peak_memory_stats()
# ... run generation ...
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM: {peak_gb:.2f} GB")
```

### Per-step memory logging

```python
def memory_callback(step, timestep, latent):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Step {step}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

result = guided_pipeline.generate(
    ...,
    callback=memory_callback,
)
```

### NVIDIA SMI monitoring

```bash
# Watch GPU utilization during generation
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 > gpu_log.csv
```

---

## 8. Known Bottlenecks & Mitigations

| Bottleneck | Impact | Mitigation |
|-----------|--------|-----------|
| VAE decode backprop | ~40% of guidance step time | Frame subsampling (4-8x), gradient checkpointing |
| Multiple guidance models | Linear VRAM increase per model | Sequential evaluation (default in adaptive mode) |
| RAFT flow estimation | ~2s per frame pair on A100 | Use raft_small, resize to 256px, subsample pairs |
| 14B DiT forward pass | ~3s per step on A100 | bf16, torch.compile (1.3x speedup) |
| Full 33-frame decode | ~12 GB activation memory | Subsample to 5-9 frames |

### Torch.compile speedup (optional)

```python
# ~1.3x faster DiT inference after warmup
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
# Note: does not help guidance backprop (dynamic shapes)
```
