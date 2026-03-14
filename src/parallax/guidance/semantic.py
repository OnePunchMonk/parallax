"""Semantic structure guidance using DINOv2."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from parallax.guidance.base import GuidanceModule


class SemanticGuidance(GuidanceModule):
    """Steer video generation to match the semantic structure of a reference.

    Uses `DINOv2 <https://huggingface.co/facebook/dinov2-small>`_ to extract
    dense feature maps from decoded frames and minimises the cosine distance
    to features extracted from a user-provided reference image.

    This is useful for maintaining semantic/structural consistency — e.g.
    *"keep the overall layout looking like this reference image"*.

    Parameters
    ----------
    model_id:
        HuggingFace model id, e.g. ``"facebook/dinov2-small"``.
    guidance_scale:
        Per-module guidance weight α.
    loss_type:
        ``"cosine"`` (feature cosine similarity) or ``"mse"`` (feature MSE).
    guidance_ratio:
        Fraction of denoising steps to apply guidance (0–1).
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        guidance_scale: float = 30.0,
        loss_type: str = "cosine",
        guidance_ratio: float = 0.5,
    ) -> None:
        self._model_id = model_id
        self._guidance_scale_value = guidance_scale
        self._loss_type = loss_type
        self._guidance_ratio = guidance_ratio
        self._model: Optional[torch.nn.Module] = None
        self._processor: Optional[Any] = None

    # -- Properties -------------------------------------------------------- #

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def default_scale(self) -> float:
        return self._guidance_scale_value

    # -- Lifecycle --------------------------------------------------------- #

    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    # -- Targets ----------------------------------------------------------- #

    def prepare_targets(
        self,
        reference_image: Union[Tensor, Image.Image],
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract DINOv2 features from a reference image.

        Parameters
        ----------
        reference_image:
            A ``(1, C, H, W)`` tensor in ``[0, 1]`` or a PIL Image.
        """
        assert self._model is not None, "Call load_model() first."

        if isinstance(reference_image, Image.Image):
            import torchvision.transforms.functional as TF

            t = TF.to_tensor(reference_image.convert("RGB"))  # (3, H, W)
            t = t.unsqueeze(0)  # (1, 3, H, W)
        else:
            t = reference_image

        if device is not None:
            t = t.to(device)

        # Resize to DINOv2 expected size (518 × 518, patch 14 → 37 × 37 grid)
        t_resized = F.interpolate(
            t, size=(518, 518), mode="bilinear", align_corners=False
        )

        # Extract features (no grad needed for target)
        with torch.no_grad():
            ref_features = self._model(
                pixel_values=t_resized.to(next(self._model.parameters()).dtype)
            ).last_hidden_state  # (1, N_patches + 1, D)

        return {"reference_features": ref_features}

    # -- Loss -------------------------------------------------------------- #

    def _extract_features(self, frames: Tensor) -> Tensor:
        """Run DINOv2 on a batch of frames and return patch features."""
        frames_resized = F.interpolate(
            frames, size=(518, 518), mode="bilinear", align_corners=False
        )
        out = self._model(
            pixel_values=frames_resized.to(next(self._model.parameters()).dtype)
        )
        return out.last_hidden_state  # (B, N+1, D)

    def compute_loss(
        self,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        timestep: int,
    ) -> Tensor:
        """Compute cosine/MSE loss between frame features and reference.

        Parameters
        ----------
        decoded_frames:
            ``(B, C, T, H, W)`` — decoded pixels in ``[0, 1]``.
        """
        assert self._model is not None, "Call load_model() first."
        ref_feats = targets["reference_features"]  # (1, N+1, D)

        B, C, T, H, W = decoded_frames.shape
        frames_flat = decoded_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        pred_feats = self._extract_features(frames_flat)  # (B*T, N+1, D)
        ref_expanded = ref_feats.expand(pred_feats.shape[0], -1, -1)

        if self._loss_type == "cosine":
            # 1 − cos_sim → loss to minimise (high similarity = low loss)
            cos = F.cosine_similarity(pred_feats, ref_expanded, dim=-1)  # (B*T, N+1)
            loss = (1.0 - cos).mean()
        else:
            loss = F.mse_loss(pred_feats, ref_expanded)

        return loss

    def should_guide_at_step(self, step: int, total_steps: int) -> bool:
        return step < total_steps * self._guidance_ratio
