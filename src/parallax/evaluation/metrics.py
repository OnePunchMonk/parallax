"""Control-specific evaluation metrics for guided video generation.

Metrics:
- Depth accuracy (RMSE, AbsRel)
- Normal accuracy (mean angular error)
- Semantic consistency (DINOv2 cosine similarity)
- Temporal consistency (optical flow warping error)
- Text alignment (CLIP score)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# Depth accuracy                                                               #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_depth_accuracy(
    frames: Tensor,
    target_depth: Tensor,
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute depth accuracy between generated video and target depth map.

    Parameters
    ----------
    frames:
        ``(T, C, H, W)`` or ``(T, H, W, C)`` generated video frames in ``[0, 1]``.
    target_depth:
        ``(1, 1, H, W)`` or ``(H, W)`` target depth map in ``[0, 1]``.
    model_id:
        Depth estimation model to use.

    Returns
    -------
    Dict with ``"rmse"``, ``"abs_rel"``, ``"delta_1"`` (% under 1.25 threshold).
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    # Normalize frames shape
    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    frames = frames.to(device).float()

    # Normalize target
    if target_depth.ndim == 2:
        target_depth = target_depth.unsqueeze(0).unsqueeze(0)
    target_depth = target_depth.to(device).float()
    target_depth = (target_depth - target_depth.min()) / (target_depth.max() - target_depth.min() + 1e-8)

    # Predict depth for each frame
    T = frames.shape[0]
    model_size = 518
    frames_resized = F.interpolate(frames, size=(model_size, model_size), mode="bilinear", align_corners=False)

    pred_depths = model(frames_resized).predicted_depth  # (T, h, w)

    # Normalize predictions per-frame
    pred_min = pred_depths.amin(dim=(-2, -1), keepdim=True)
    pred_max = pred_depths.amax(dim=(-2, -1), keepdim=True)
    pred_depths = (pred_depths - pred_min) / (pred_max - pred_min + 1e-8)

    # Resize target to match predictions
    target_resized = F.interpolate(
        target_depth, size=pred_depths.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)  # (h, w)
    target_expanded = target_resized.unsqueeze(0).expand_as(pred_depths)

    # Metrics
    diff = (pred_depths - target_expanded).abs()
    rmse = diff.pow(2).mean().sqrt().item()
    abs_rel = (diff / (target_expanded + 1e-8)).mean().item()

    # delta_1: fraction of pixels where max(pred/target, target/pred) < 1.25
    ratio = torch.max(
        pred_depths / (target_expanded + 1e-8),
        target_expanded / (pred_depths + 1e-8),
    )
    delta_1 = (ratio < 1.25).float().mean().item()

    return {"rmse": rmse, "abs_rel": abs_rel, "delta_1": delta_1}


# --------------------------------------------------------------------------- #
# Normal accuracy                                                              #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_normal_accuracy(
    frames: Tensor,
    target_normal: Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute surface normal accuracy.

    Uses depth-to-normal conversion to estimate normals from generated
    frames and compares against target normal map.

    Parameters
    ----------
    frames:
        ``(T, C, H, W)`` generated frames in ``[0, 1]``.
    target_normal:
        ``(1, 3, H, W)`` target normals in ``[-1, 1]``.

    Returns
    -------
    Dict with ``"mean_angular_error"`` (degrees), ``"median_angular_error"``.
    """
    from parallax.guidance.normal import NormalGuidance

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        frames = frames.permute(0, 3, 1, 2)
    frames = frames.to(device).float()

    target_normal = target_normal.to(device).float()
    target_normal = F.normalize(target_normal, dim=1, eps=1e-6)

    # Estimate depth then convert to normals (same as NormalGuidance)
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device).eval()

    model_size = 518
    frames_resized = F.interpolate(frames, size=(model_size, model_size), mode="bilinear", align_corners=False)
    pred_depth = model(frames_resized).predicted_depth

    pred_min = pred_depth.amin(dim=(-2, -1), keepdim=True)
    pred_max = pred_depth.amax(dim=(-2, -1), keepdim=True)
    pred_depth = (pred_depth - pred_min) / (pred_max - pred_min + 1e-8)

    pred_normals = NormalGuidance._depth_to_normals(pred_depth.unsqueeze(1))

    # Resize target to match
    target_resized = F.interpolate(
        target_normal, size=pred_normals.shape[-2:], mode="bilinear", align_corners=False
    )
    target_resized = F.normalize(target_resized, dim=1, eps=1e-6)
    target_expanded = target_resized.expand(pred_normals.shape[0], -1, -1, -1)

    # Angular error
    cos_sim = F.cosine_similarity(pred_normals, target_expanded, dim=1)  # (T, h, w)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    angular_error_rad = torch.acos(cos_sim)
    angular_error_deg = angular_error_rad * (180.0 / torch.pi)

    return {
        "mean_angular_error": angular_error_deg.mean().item(),
        "median_angular_error": angular_error_deg.median().item(),
    }


# --------------------------------------------------------------------------- #
# Semantic consistency                                                         #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_semantic_consistency(
    frames: Tensor,
    reference_image: Tensor,
    model_id: str = "facebook/dinov2-small",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute DINOv2 semantic consistency between frames and reference.

    Parameters
    ----------
    frames:
        ``(T, C, H, W)`` generated frames in ``[0, 1]``.
    reference_image:
        ``(1, C, H, W)`` reference image in ``[0, 1]``.

    Returns
    -------
    Dict with ``"mean_cosine_sim"``, ``"min_cosine_sim"``, ``"std_cosine_sim"``.
    """
    from transformers import AutoModel

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        frames = frames.permute(0, 3, 1, 2)
    frames = frames.to(device).float()
    reference_image = reference_image.to(device).float()

    model = AutoModel.from_pretrained(model_id).to(device).eval()

    model_size = 518
    frames_resized = F.interpolate(frames, size=(model_size, model_size), mode="bilinear", align_corners=False)
    ref_resized = F.interpolate(reference_image, size=(model_size, model_size), mode="bilinear", align_corners=False)

    # Extract features
    frame_feats = model(pixel_values=frames_resized).last_hidden_state  # (T, N+1, D)
    ref_feats = model(pixel_values=ref_resized).last_hidden_state  # (1, N+1, D)

    # Use CLS token for global similarity
    frame_cls = frame_feats[:, 0]  # (T, D)
    ref_cls = ref_feats[:, 0].expand(frame_cls.shape[0], -1)  # (T, D)

    cos_sim = F.cosine_similarity(frame_cls, ref_cls, dim=-1)  # (T,)

    return {
        "mean_cosine_sim": cos_sim.mean().item(),
        "min_cosine_sim": cos_sim.min().item(),
        "std_cosine_sim": cos_sim.std().item(),
    }


# --------------------------------------------------------------------------- #
# Temporal consistency                                                         #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_temporal_consistency(
    frames: Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute temporal consistency via optical flow warping error.

    For each pair of consecutive frames, estimate flow, warp frame t by
    the flow, and measure reconstruction error against frame t+1.

    Parameters
    ----------
    frames:
        ``(T, C, H, W)`` generated frames in ``[0, 1]``.

    Returns
    -------
    Dict with ``"mean_warp_error"``, ``"max_warp_error"``,
    ``"mean_flow_magnitude"``.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        frames = frames.permute(0, 3, 1, 2)
    frames = frames.to(device).float()

    T, C, H, W = frames.shape
    if T < 2:
        return {"mean_warp_error": 0.0, "max_warp_error": 0.0, "mean_flow_magnitude": 0.0}

    model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    # Resize for RAFT
    new_h = (H // 8) * 8
    new_w = (W // 8) * 8
    if new_h != H or new_w != W:
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
        H, W = new_h, new_w

    warp_errors = []
    flow_mags = []

    for t in range(T - 1):
        f1 = frames[t:t+1] * 255.0
        f2 = frames[t+1:t+2] * 255.0

        flow = model(f1, f2)[-1]  # (1, 2, H, W)

        # Warp frame1 by flow
        from parallax.guidance.flow import FlowGuidance
        warped = FlowGuidance._warp_frame(frames[t:t+1], flow)

        # Warping error
        error = F.mse_loss(warped, frames[t+1:t+2]).item()
        warp_errors.append(error)

        # Flow magnitude
        mag = flow.pow(2).sum(dim=1).sqrt().mean().item()
        flow_mags.append(mag)

    return {
        "mean_warp_error": float(np.mean(warp_errors)),
        "max_warp_error": float(np.max(warp_errors)),
        "mean_flow_magnitude": float(np.mean(flow_mags)),
    }


# --------------------------------------------------------------------------- #
# CLIP score                                                                   #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_clip_score(
    frames: Tensor,
    prompt: str,
    model_id: str = "openai/clip-vit-base-patch32",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute CLIP score for text-video alignment.

    Parameters
    ----------
    frames:
        ``(T, C, H, W)`` generated frames in ``[0, 1]``.
    prompt:
        Text prompt used for generation.

    Returns
    -------
    Dict with ``"mean_clip_score"``, ``"min_clip_score"``.
    """
    from transformers import CLIPModel, CLIPProcessor

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        frames = frames.permute(0, 3, 1, 2)
    frames = frames.to(device).float()

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    T = frames.shape[0]
    clip_size = 224

    frames_resized = F.interpolate(
        frames, size=(clip_size, clip_size), mode="bilinear", align_corners=False
    )

    # Encode text
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)  # (1, D)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Encode frames
    image_features = model.get_image_features(pixel_values=frames_resized)  # (T, D)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Cosine similarity
    scores = (image_features @ text_features.T).squeeze(-1)  # (T,)

    return {
        "mean_clip_score": scores.mean().item(),
        "min_clip_score": scores.min().item(),
    }


# --------------------------------------------------------------------------- #
# Aggregate evaluation                                                         #
# --------------------------------------------------------------------------- #


def evaluate_all(
    frames: Tensor,
    prompt: str,
    target_depth: Optional[Tensor] = None,
    target_normal: Optional[Tensor] = None,
    reference_image: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Run all applicable evaluation metrics.

    Parameters
    ----------
    frames:
        ``(T, C, H, W)`` generated video frames in ``[0, 1]``.
    prompt:
        Text prompt.
    target_depth:
        Optional target depth map for depth accuracy.
    target_normal:
        Optional target normal map for normal accuracy.
    reference_image:
        Optional reference image for semantic consistency.

    Returns
    -------
    Nested dict of metric groups and values.
    """
    results: Dict[str, Dict[str, float]] = {}

    # Always compute temporal consistency and CLIP score
    results["temporal"] = compute_temporal_consistency(frames, device=device)
    results["clip"] = compute_clip_score(frames, prompt, device=device)

    if target_depth is not None:
        results["depth"] = compute_depth_accuracy(frames, target_depth, device=device)

    if target_normal is not None:
        results["normal"] = compute_normal_accuracy(frames, target_normal, device=device)

    if reference_image is not None:
        results["semantic"] = compute_semantic_consistency(
            frames, reference_image, device=device
        )

    return results
