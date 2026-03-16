# Parallax — Research Project Scope

## Paper Title

**"Parallax: Composable Test-Time Guidance for Image-to-Video Diffusion via Vision Foundation Models"**

**Target venues:** NeurIPS 2026 (May deadline), CVPR 2027 (Nov deadline)

---

## 1. Core Contribution

A **training-free** framework for controllable Image-to-Video generation that composes **multiple heterogeneous vision foundation model signals** as gradient-based guidance during diffusion denoising. Unlike prior work that trains separate ControlNets per signal or uses single guidance signals, Parallax:

1. **Composes 5+ vision model signals** (depth, normals, semantic, segmentation, optical flow) at test time with zero training
2. **Introduces surface normal guidance for video** — completely unexplored in prior work
3. **Uses optical flow as a temporal consistency regularizer** — addressing the #1 open problem in test-time video guidance
4. **Proposes adaptive gradient-normalized composition** — a principled alternative to manual alpha-weighting

---

## 2. Novelty Claims (vs. Prior Art)

| Aspect | Prior Art | Parallax |
|--------|-----------|----------|
| **Guidance signals** | 1-2 signals (TITAN-Guide: aesthetic/style; SG-I2V: self-attention only) | 5+ composable signals from off-the-shelf VFMs |
| **Normal guidance** | FreeControl (images only, not gradient-based) | First gradient-based normal guidance for video diffusion |
| **Temporal coherence** | Per-frame guidance, no cross-frame loss | Optical flow consistency loss across adjacent frames |
| **Composition** | Manual alpha weighting | Adaptive gradient-normalized composition with conflict detection |
| **Architecture** | Mostly U-Net based (SD 1.5, VideoCrafter) | DiT-based (Wan2.1), modern architecture |
| **Training** | ControlNet, adapter fine-tuning | Fully training-free, test-time only |

---

## 3. Technical Approach

### 3.1 New Guidance Modules

#### A. Surface Normal Guidance (`NormalGuidance`)
- **Model:** `prs-eth/marigold-normals-v1-1` (Marigold Normals, diffusion-based)
- **Loss:** Cosine angular error between predicted normals and target normal map
- **Why novel:** No prior work uses normals for video guidance. Normals encode surface orientation (complementary to depth which encodes distance). Critical for controlling lighting, surface detail, and 3D consistency.

#### B. Optical Flow Consistency Guidance (`FlowGuidance`)
- **Model:** RAFT (`princeton-vl/RAFT` via torchvision, or `hf-internal-testing/raft_things`)
- **Loss:** End-Point Error (EPE) between optical flow of adjacent decoded frames and a target flow field (or smoothness regularizer)
- **Why novel:** Uses flow not as a motion control signal, but as a **temporal consistency regularizer** during multi-signal guidance. Penalizes temporal flickering caused by per-frame guidance.

### 3.2 Adaptive Composite Guidance (`AdaptiveCompositeGuidance`)

Replaces manual alpha-weighting with:
1. **Gradient normalization:** Scale each module's gradient to unit norm before combining, preventing any single signal from dominating
2. **Conflict detection:** Compute cosine similarity between gradient directions; when signals conflict (cos < 0), reduce the conflicting signal's weight
3. **Temporal scheduling:** Different signals activate at different denoising phases (depth early, semantics mid, flow late)

### 3.3 Evaluation Framework

Control-specific metrics computed on generated videos:
- **Depth accuracy:** RMSE, AbsRel between predicted depth of output and target
- **Normal accuracy:** Mean angular error (degrees) between predicted and target normals
- **Semantic consistency:** DINOv2 cosine similarity between output and reference
- **Temporal consistency:** Optical flow warping error between adjacent frames
- **Overall quality:** FID on individual frames, CLIP score for text alignment

---

## 4. Implementation Plan

### Phase 1: New Guidance Modules (this PR)
- [x] `NormalGuidance` — Marigold Normals integration
- [x] `FlowGuidance` — RAFT optical flow consistency
- [x] `AdaptiveCompositeGuidance` — gradient-normalized composition
- [x] Evaluation metrics module
- [x] Updated CLI, configs, tests

### Phase 2: Experiments
- [ ] Single-signal ablation: depth vs. normal vs. semantic vs. flow
- [ ] Pairwise composition: depth+normal, depth+semantic, depth+flow
- [ ] Full composition: depth+normal+semantic+flow
- [ ] Guidance strength sweeps (α ∈ {10, 30, 50, 100})
- [ ] Temporal scheduling ablation
- [ ] Comparison vs. unguided Wan2.1 baseline

### Phase 3: Paper
- [ ] VBench++ I2V evaluation
- [ ] Control-specific metric tables
- [ ] Human evaluation (pairwise preference)
- [ ] Qualitative figure grid
- [ ] Ablation tables

---

## 5. Key HuggingFace Models

| Signal | Model ID | Size | License |
|--------|----------|------|---------|
| Base I2V | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 14B | Apache 2.0 |
| Depth | `depth-anything/Depth-Anything-V2-Small-hf` | 24M | Apache 2.0 |
| Normals | `prs-eth/marigold-normals-v1-1` | ~1B | Apache 2.0 |
| Semantic | `facebook/dinov2-small` | 22M | Apache 2.0 |
| Segmentation | `facebook/sam2-hiera-small` | 38M | Apache 2.0 |
| Flow | `torchvision.models.optical_flow.raft_small` | 1M | BSD-3 |

---

## 6. Competitive Position

```
                    Training Required?
                YES                     NO (Test-Time)
                 |                          |
  Single:     ControlNet              SG-I2V, OnlyFlow
              ControlVideo            TITAN-Guide
                 |                          |
  Multi:      Video-As-Prompt         Video-MSG (noise-init)
              ATI                          |
                 |                     ★ PARALLAX ★
                 |                   (5+ signals, gradient-based,
                 |                    adaptive composition,
                 |                    DiT backbone, normals+flow)
```

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Memory OOM during multi-guidance | Frame subsampling (decode every 4th frame), gradient checkpointing, sequential module evaluation |
| Normal model not differentiable end-to-end | Use feature-space matching (intermediate features) instead of full pipeline output |
| Flow guidance too expensive | Compute flow only on subsampled frame pairs, use RAFT-Small |
| Guidance degrades quality | Cosine annealing schedule (strong early, weak late), gradient clipping |
| Signals conflict | Adaptive gradient normalization + conflict detection in composite module |

---

## 8. Expected Results

- Normal guidance produces videos with consistent surface geometry (measurable via angular error)
- Flow consistency reduces temporal flickering by 20-40% (measurable via warping error)
- Adaptive composition outperforms manual weighting on composite control tasks
- Full system achieves controllable I2V generation across 5 signal types with no training
