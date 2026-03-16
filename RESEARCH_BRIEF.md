# Parallax Research Brief: Zero-Shot Controllable I2V via Test-Time Guidance

*Compiled: March 2026*

---

## 1. Current State of Controllable I2V (2024--2026)

### 1.1 Foundational Works

| Paper | Venue | Key Idea |
|-------|-------|----------|
| **Universal Guidance for Diffusion Models** (Bansal et al.) | CVPR 2023 Workshop / ICLR 2024 | Generic algorithm to guide any diffusion model using arbitrary off-the-shelf classifiers/regressors via gradient-based steering. Supports segmentation, face recognition, object detection, and classifier signals. |
| **FreeControl** (Mo et al.) | CVPR 2024 | Training-free spatial control of any T2I diffusion model by modeling the linear subspace of intermediate diffusion features. Supports depth, normal, edge, pose, segmentation, and sketch control across SD 1.5, 2.1, and SDXL. |
| **ControlVideo** (Zhang et al.) | ICLR 2024 | Training-free controllable T2V generation adapted from ControlNet. Uses depth maps and other conditioning inputs for video diffusion. |

### 1.2 Recent Training-Free Video Guidance Methods (2025--2026)

| Paper | Year/Venue | Method |
|-------|------------|--------|
| **TITAN-Guide** (Simon et al.) | ICCV 2025 | Inference-time alignment for guided T2V diffusion. Eliminates backpropagation via **forward gradients**, solving the memory bottleneck. Supports aesthetic, compositional, multimodal (audio-video), and style guidance. **Most directly relevant competitor.** |
| **Video-MSG** (Li et al.) | arXiv 2025 | Training-free guidance via multimodal planning + structured noise initialization. Creates a "video sketch" (background + foreground trajectories) and uses inversion-based initialization. No attention manipulation or fine-tuning. Tested on VideoCrafter2 and CogVideoX-5B. |
| **SG-I2V** (Namekata et al.) | ICLR 2025 | Self-guided trajectory control for I2V. Zero-shot, no external knowledge -- exploits the model's own internal representations for motion control. |
| **I2V3D** (Zhang et al.) | ICCV 2025 | Controllable I2V via 3D geometry guidance. Two-stage: 3D-guided keyframe generation + training-free video interpolation with bidirectional guidance. |
| **FlowMotion** (Wang et al.) | arXiv 2026 | Training-free optical flow guidance for video motion transfer. |
| **OnlyFlow** (2024) | arXiv 2024 | Optical flow conditioning via a flow encoder for motion-guided video diffusion. |
| **Time-to-Move (TTM)** | arXiv 2025 | Plug-and-play framework for motion/appearance-controlled I2V generation using dual-clock denoising. |
| **MotionMaster** | ACM MM 2024 | Training-free camera motion transfer for video generation. |
| **Motion Prompting** (Geng et al.) | CVPR 2025 | Controlling video generation with motion trajectories. |
| **ATI** | 2025 | Trajectory-based motion control on Wan2.1-I2V-14B; unifies object, local, and camera movements. |
| **Video-As-Prompt** | 2025 | Unified semantic-controlled video generation on Wan2.1-14B-I2V with Mixture-of-Transformers for concept, style, motion, and camera control. |

### 1.3 Key Observation

**Parallax sits in a unique position**: it combines (a) gradient-based test-time guidance (like Universal Guidance / TITAN-Guide) with (b) multiple vision foundation models (Depth-Anything V2, DINOv2, SAM2) applied to (c) a modern DiT-based video model (Wan2.1). Most existing work uses either a single guidance signal OR targets older U-Net architectures. The **multi-modal guidance composition on DiT-based video diffusion** is relatively unexplored.

---

## 2. Gaps and Open Problems

### 2.1 Underexplored Control Signals

| Signal Type | Current Status | Opportunity |
|-------------|---------------|-------------|
| **Optical flow** | OnlyFlow (encoder-based, not gradient-guided), FlowMotion (2026) | Gradient-based flow guidance at test time -- directly optimizing flow consistency across frames |
| **Surface normals** | Used in FreeControl for images only | **No work on normal-guided video generation**. Normals provide 3D geometric cues complementary to depth. |
| **Human pose** | ControlNet-based (requires training) | Training-free pose guidance using ViTPose/RTMPose features |
| **Aesthetic/style** | TITAN-Guide explores this | Gradient guidance with LAION aesthetic predictor or CLIP-based style embeddings |
| **Object permanence** | Largely unaddressed | SAM2 tracking + identity consistency loss across frames |
| **Edge/sketch** | ControlNet-based only | Training-free edge guidance using HED/PiDiNet for stylistic control |

### 2.2 Known Limitations of Test-Time Guidance for Video Diffusion

1. **Memory cost**: Backpropagating through a full DiT + VAE decode + vision model is extremely expensive. TITAN-Guide cannot exceed 384x384 on consumer GPUs. Parallax must solve this.
2. **Temporal consistency**: Zero-shot image diffusion models produce independent hallucinations per frame. Gradient guidance applied per-frame does not inherently enforce cross-frame coherence.
3. **Guidance-quality tradeoff**: Too-strong guidance degrades visual quality; too-weak guidance has no effect. Adaptive scheduling is not well studied.
4. **Latent-space vs. pixel-space guidance**: Guiding in pixel space (after VAE decode) introduces VAE decode artifacts and is expensive. Latent-space guidance is faster but less semantically meaningful.
5. **Composing multiple guidance signals**: No principled framework exists for weighting and scheduling multiple heterogeneous guidance losses. Current approaches use manual alpha-weighting.
6. **Efficiency of 3D VAE decoding**: Wan2.1 uses a 3D spatio-temporal VAE. Differentiable decode of the full video is prohibitively expensive; frame subsampling strategies are ad hoc.

### 2.3 Specific Open Research Questions

- Can **forward gradients** (TITAN-Guide's approach) be effectively combined with **multi-modal guidance** to achieve memory-efficient composite control?
- How should guidance be **temporally scheduled** (e.g., stronger guidance in early steps, weaker later) to balance quality and controllability?
- Can vision foundation model features be matched in **latent space** (bypassing VAE decode entirely) for efficiency?
- How do different guidance signals **interact** -- does depth guidance conflict with semantic guidance? Can they be made synergistic?

---

## 3. Available HuggingFace Models for Novel Guidance Signals

### 3.1 Depth Estimation (Already Implemented)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| Depth-Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | Currently used in Parallax |
| Depth-Anything V2 Large | `depth-anything/Depth-Anything-V2-Large-hf` | Higher quality |
| Video Depth Anything | github.com/DepthAnything/Video-Depth-Anything | Temporally consistent video depth (CVPR 2025 Highlight). Could replace per-frame depth for better temporal coherence. |
| Marigold Depth | `prs-eth/marigold-depth-v1-0` | Diffusion-based depth |

### 3.2 Surface Normal Estimation (HIGH PRIORITY -- Novel)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| **Marigold Normals v1.1** | `prs-eth/marigold-normals-v1-1` | Diffusion-based; SOTA on multiple benchmarks. Produces 3D unit vectors in screen space. |
| Marigold Normals LCM | `prs-eth/marigold-normals-lcm-v0-1` | Faster LCM variant |
| **StableNormal** | `Stable-X/StableNormal` (space) | Robust under extreme lighting, blur, transparent/reflective surfaces. Good for challenging videos. |
| GeoWizard | `lemonaddie/geowizard` | Joint depth + normal estimation. Ensures geometric consistency between depth and normals. |

### 3.3 Semantic Features (Already Implemented)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| DINOv2 Small | `facebook/dinov2-small` | Currently used |
| DINOv2 Base/Large/Giant | `facebook/dinov2-base`, `-large`, `-giant` | Scaling options |
| DINOv2 with registers | `facebook/dinov2-large-res4` | Better attention maps |

### 3.4 Segmentation (Already Implemented)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| SAM2 | github.com/facebookresearch/segment-anything-2 | Currently used |

### 3.5 Optical Flow (HIGH PRIORITY -- Novel)

| Model | HuggingFace ID / Source | Notes |
|-------|------------------------|-------|
| **RAFT** | `opencv/optical_flow_estimation_raft` | Classic, fast. HuggingFace space: `fffiloni/RAFT` |
| **SEA-RAFT** | github.com/princeton-vl/SEA-RAFT | ECCV 2024 Oral, Best Paper Candidate. Simple, Efficient, Accurate. |
| Perceiver Optical Flow | `deepmind/optical-flow-perceiver` | DeepMind's perceiver-based flow |
| FlowFormer | github.com/drinkingcoder/FlowFormer-Official | Transformer-based flow |

### 3.6 Human Pose Estimation (MEDIUM PRIORITY)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| **ViTPose Base** | `danelcsb/vitpose-base-simple` | Integrated in HF Transformers. 81.1 AP on COCO. |
| ViTPose++ Large | `usyd-community/vitpose-plus-large` | MoE variant; multi-dataset |
| RTMPose | mmpose library | Not natively on HF but widely used |

### 3.7 Edge / Sketch Detection (MEDIUM PRIORITY)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| HED (via ControlNet preprocessor) | `lllyasviel/Annotators` | Holistically-Nested Edge Detection |
| PiDiNet | controlnet_aux library | Cleaner edges than HED |
| Canny | OpenCV (no model needed) | Classic; differentiable approximations exist |

### 3.8 Aesthetic / Style Scoring (NOVEL ANGLE)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| **LAION Aesthetic Predictor V2** | `camenduru/improved-aesthetic-predictor` | CLIP ViT-L/14 + MLP. Used to curate Stable Diffusion training data. |
| Image Quality Fusion | `matthewyuan/image-quality-fusion` | Multi-metric quality predictor |

### 3.9 Object Detection (LOWER PRIORITY)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| DETR | `facebook/detr-resnet-50` | End-to-end detection |
| YOLO (various) | ultralytics library | Fast but not natively differentiable |
| Grounding DINO | `IDEA-Research/grounding-dino-base` | Open-vocabulary detection |

---

## 4. Benchmarks and Evaluation

### 4.1 Primary Benchmarks

| Benchmark | Focus | Key Features |
|-----------|-------|-------------|
| **VBench++** (TPAMI 2025) | Comprehensive video gen eval | 16 dimensions: subject consistency, background consistency, temporal flickering, motion smoothness, aesthetic quality, imaging quality, dynamic degree. Supports I2V evaluation with adaptive Image Suite. HF Leaderboard: `Vchitect/VBench_Leaderboard` |
| **T2VCompBench** | Compositional T2V | Tests spatial relationships, attribute binding, motion |
| **VBench-2.0** | Extended | Adds controllability evaluation |

### 4.2 Metrics

| Metric | Type | What It Measures | Limitations |
|--------|------|-----------------|-------------|
| **FVD** (Frechet Video Distance) | Set-to-set | Overall video quality + some temporal info via I3D features | Content-biased; prioritizes per-frame quality over temporal consistency |
| **CLIP Score** | Unary | Text-video alignment (cosine similarity per frame) | Does not penalize temporal inconsistencies |
| **FVMD** (Frechet Video Motion Distance) | Set-to-set | **Motion/temporal consistency** via keypoint tracking (PIPs++), velocity/acceleration fields | Newer, less widely adopted |
| **WCS** (World Consistency Score) | Unary | Object permanence, relation stability, causal compliance, flicker penalty | Very new (2025), comprehensive but complex |
| **SSIM / PSNR** | Frame-level | Pixel-level fidelity | Does not capture perceptual quality |
| **LPIPS** | Frame-level | Perceptual similarity | Per-frame only |

### 4.3 Recommended Evaluation Protocol for Parallax

1. **VBench++ I2V suite** -- primary benchmark for overall quality
2. **Control-specific metrics**:
   - Depth accuracy: RMSE / AbsRel between predicted depth of generated video and target depth map
   - Semantic consistency: DINOv2 cosine similarity between generated and reference frames
   - Segmentation accuracy: mIoU between SAM2 output on generated video and target masks
   - Optical flow error: EPE (End-Point Error) between generated and target flow
   - Normal accuracy: Angular error between predicted and target normals
3. **Temporal consistency**: FVMD + warping error (warp frame t with flow to t+1, measure reconstruction error)
4. **Ablation**: Guided vs. unguided, single vs. multi-guidance, guidance strength sweeps
5. **Human evaluation**: Pairwise preference on (a) controllability, (b) visual quality, (c) temporal smoothness

---

## 5. Publishable Novelty Angles (Ranked by Impact x Feasibility)

### Tier 1: Strong Novelty, High Feasibility

#### 1A. "Parallax: Composable Test-Time Guidance for Video Diffusion via Vision Foundation Models"
- **Claim**: First framework for composing multiple heterogeneous vision foundation model signals (depth + semantic + segmentation + normals + flow) as gradient-based guidance for DiT-based video diffusion -- no training required.
- **Key novelties**:
  - Principled multi-guidance composition with learned or adaptive weighting (not just manual alphas)
  - Temporal guidance scheduling strategies (curriculum: strong early, weak late; or alternating signals)
  - Comprehensive ablation of guidance signal interactions on a modern DiT backbone (Wan2.1/2.2)
- **Why it publishes**: No existing work composes >2 heterogeneous guidance signals at test time for video. TITAN-Guide is the closest but focuses on forward gradients for efficiency, not multi-modal composition. FreeControl is image-only.
- **Target venues**: CVPR 2027, ECCV 2026, NeurIPS 2026
- **Feasibility**: HIGH -- builds directly on existing Parallax codebase; add 2-3 new guidance modules

#### 1B. "Normal-Guided Video Generation: 3D Geometric Control Without Training"
- **Claim**: Surface normals as a novel guidance signal for video diffusion, using Marigold Normals or StableNormal.
- **Key insight**: Normals encode surface orientation (complementary to depth which encodes distance). Normals are especially useful for controlling lighting-dependent appearance, surface detail, and 3D consistency.
- **Why it publishes**: No prior work uses normals for video guidance. GeoWizard shows depth+normals are synergistic for images. Joint depth+normal guidance for video is completely unexplored.
- **Feasibility**: HIGH -- Marigold Normals is differentiable and on HuggingFace

### Tier 2: Strong Novelty, Moderate Feasibility

#### 2A. "Temporally Coherent Test-Time Guidance via Flow-Consistency Losses"
- **Claim**: Use optical flow models (SEA-RAFT) as a temporal consistency regularizer during guided generation. Instead of (or in addition to) guiding per-frame content, explicitly penalize flow inconsistency between consecutive generated frames.
- **Key novelties**:
  - Flow-consistency loss: compute optical flow on consecutive decoded frames, penalize deviation from expected flow (smooth motion, target trajectories)
  - Jointly guides content (depth/semantic) AND temporal coherence (flow), addressing the key limitation of per-frame guidance
- **Why it publishes**: Temporal consistency is the #1 open problem for test-time guidance in video. This directly addresses it.
- **Feasibility**: MODERATE -- requires differentiable flow estimation (RAFT is differentiable), but memory cost of computing flow on decoded frames is significant

#### 2B. "Memory-Efficient Multi-Modal Guidance via Latent-Space Feature Matching"
- **Claim**: Instead of decoding to pixel space and running vision models, align vision model features with latent-space features directly, bypassing VAE decode.
- **Key insight**: Train a lightweight probe (single linear layer) to predict vision model features from Wan2.1 latent features. At test time, guide latents to match target vision features without VAE decode.
- **Why it publishes**: Addresses the fundamental memory bottleneck. TITAN-Guide uses forward gradients; this is an orthogonal approach. Could enable guidance at 720p+ resolution.
- **Feasibility**: MODERATE -- requires training probes (lightweight, but technically not "zero training")

### Tier 3: High Novelty, Lower Feasibility (Moonshots)

#### 3A. "Adaptive Guidance Scheduling via Reinforcement Learning at Test Time"
- **Claim**: Learn to adaptively select which guidance signals to apply, at what strength, at each denoising step.
- **Why it publishes**: Completely novel meta-learning angle. Addresses the composition problem formally.
- **Feasibility**: LOW -- requires RL optimization during generation, very slow

#### 3B. "Physics-Informed Video Guidance: Enforcing Physical Plausibility at Test Time"
- **Claim**: Use physics-based losses (gravity, collision, rigid body constraints) as guidance signals alongside geometric/semantic guidance.
- **Related**: "Inference-time Physics Alignment of Video Generative Models with Latent World Models" (arXiv 2026)
- **Feasibility**: LOW-MODERATE -- physics simulators are generally not differentiable end-to-end

---

## 6. Concrete Next Steps

### Phase 1 (Weeks 1-4): Expand Guidance Modules
1. Implement **NormalGuidance** using `prs-eth/marigold-normals-v1-1`
2. Implement **FlowConsistencyGuidance** using RAFT / SEA-RAFT
3. Test joint depth + normal guidance (leveraging GeoWizard's insight)
4. Benchmark against unguided Wan2.1 I2V on VBench++ I2V suite

### Phase 2 (Weeks 5-8): Multi-Guidance Composition
1. Develop principled composition framework (gradient normalization, conflict detection)
2. Implement temporal guidance scheduling (early/late, alternating)
3. Ablation study: single vs. pairwise vs. all-guidance composition
4. Compare against TITAN-Guide and Video-MSG

### Phase 3 (Weeks 9-12): Paper Writing and Evaluation
1. Full evaluation on VBench++, T2VCompBench
2. Control-specific metrics (depth RMSE, normal angular error, flow EPE, DINO cosine sim)
3. Human evaluation study
4. Ablation tables, qualitative comparisons, failure analysis

### Target Submission
- **CVPR 2027** (deadline likely Nov 2026) or **ECCV 2026** (deadline likely Mar 2026)
- **NeurIPS 2026** (deadline May 2026) -- most realistic near-term target

---

## 7. Key References

1. Bansal et al., "Universal Guidance for Diffusion Models," CVPR 2023W / ICLR 2024. https://arxiv.org/abs/2302.07121
2. Mo et al., "FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model," CVPR 2024
3. Zhang et al., "ControlVideo: Training-free Controllable Text-to-video Generation," ICLR 2024
4. Simon et al., "TITAN-Guide: Taming Inference-Time Alignment for Guided T2V Diffusion Models," ICCV 2025. https://arxiv.org/abs/2508.00289
5. Li et al., "Video-MSG: Training-free Guidance via Multimodal Planning and Structured Noise Initialization," 2025. https://arxiv.org/abs/2504.08641
6. Namekata et al., "SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation," ICLR 2025
7. Zhang et al., "I2V3D: Controllable Image-to-Video Generation with 3D Guidance," ICCV 2025
8. Wang et al., "FlowMotion: Training-Free Flow Guidance for Video Motion Transfer," 2026
9. Geng et al., "Motion Prompting: Controlling Video Generation with Motion Trajectories," CVPR 2025
10. Huang et al., "VBench: Comprehensive Benchmark Suite for Video Generative Models," CVPR 2024 Highlight
11. VBench++ (TPAMI 2025). https://arxiv.org/abs/2411.13503
12. Wan2.1: https://github.com/Wan-Video/Wan2.1 / Wan2.2: https://github.com/Wan-Video/Wan2.2
13. Marigold Normals: https://huggingface.co/prs-eth/marigold-normals-v1-1
14. StableNormal: https://huggingface.co/spaces/Stable-X/StableNormal
15. GeoWizard (ECCV 2024): https://github.com/fuxiao0719/GeoWizard
16. SEA-RAFT (ECCV 2024 Oral): https://github.com/princeton-vl/SEA-RAFT
17. Video Depth Anything (CVPR 2025 Highlight): https://github.com/DepthAnything/Video-Depth-Anything
18. ViTPose: https://huggingface.co/docs/transformers/model_doc/vitpose
19. LAION Aesthetic Predictor: https://github.com/LAION-AI/aesthetic-predictor
20. World Consistency Score (2025): https://arxiv.org/abs/2508.00144

---

## 8. Competitive Landscape Summary

```
                        Training Required?
                    YES                     NO (Test-Time)
                     |                          |
  Control Signal     |                          |
  Variety:           |                          |
                     |                          |
  Single signal:   ControlNet            SG-I2V, MotionMaster,
                   ControlVideo(trained)  OnlyFlow, TTM
                     |                          |
  Multi-signal:    Video-As-Prompt       TITAN-Guide (limited),
                   ATI                   Video-MSG (noise-init only)
                     |                          |
                     |                     *PARALLAX* <-- HERE
                     |                   (multi-modal gradient
                     |                    guidance, composable,
                     |                    Wan2.1 DiT backbone)
```

**Parallax's unique position**: The only framework combining (1) gradient-based test-time guidance, (2) multiple heterogeneous vision foundation model signals, (3) principled multi-guidance composition, and (4) a modern DiT-based video backbone. This gap is real and publishable.
