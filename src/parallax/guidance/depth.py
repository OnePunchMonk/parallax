"""Depth guidance using Depth-Anything v2."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from parallax.guidance.base import GuidanceModule


class DepthGuidance(GuidanceModule):
    """Steer video generation toward a target depth layout.

    Uses `Depth-Anything-V2 <https://huggingface.co/depth-anything>`_
    to extract a monocular depth map from each (subset of) decoded frame(s)
    and minimises the MSE between this prediction and a user-supplied target
    depth map.

    Parameters
    ----------
    model_id:
        HuggingFace model id, e.g.
        ``"depth-anything/Depth-Anything-V2-Small-hf"``.
    guidance_scale:
        Per-module guidance weight α.
    loss_type:
        ``"mse"`` (pixel-level) or ``"feature"`` (intermediate DPT features).
    guidance_ratio:
        Fraction of denoising steps to apply guidance (0–1).
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        guidance_scale: float = 50.0,
        loss_type: str = "mse",
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
        return "depth"

    @property
    def default_scale(self) -> float:
        return self._guidance_scale_value

    # -- Lifecycle --------------------------------------------------------- #

    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(device)
        # We need gradients through the depth model
        self._model.eval()
        # Enable grad for all parameters so autograd graph is preserved
        for p in self._model.parameters():
            p.requires_grad_(False)

    # -- Targets ----------------------------------------------------------- #

    def prepare_targets(
        self,
        target_depth: Union[Tensor, Image.Image],
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process the target depth map.

        Parameters
        ----------
        target_depth:
            Either a ``(1, 1, H, W)`` tensor or a PIL Image (grayscale depth).
        """
        if isinstance(target_depth, Image.Image):
            import torchvision.transforms.functional as TF

            t = TF.to_tensor(target_depth.convert("L"))  # (1, H, W)
            t = t.unsqueeze(0)  # (1, 1, H, W)
        else:
            t = target_depth
        if device is not None:
            t = t.to(device)
        # Normalise to [0, 1]
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return {"target_depth": t}

    # -- Loss -------------------------------------------------------------- #

    def compute_loss(
        self,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        timestep: int,
    ) -> Tensor:
        """Compute MSE loss between predicted and target depth maps.

        Parameters
        ----------
        decoded_frames:
            ``(B, C, T, H, W)`` — decoded pixels in ``[0, 1]``.
        targets:
            Dict containing ``target_depth`` tensor.
        """
        assert self._model is not None, "Call load_model() first."
        target = targets["target_depth"]  # (1, 1, H, W)
        device = decoded_frames.device

        B, C, T, H, W = decoded_frames.shape
        # Reshape: (B*T, C, H, W)
        frames_flat = decoded_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Resize to model input size (518 × 518 for Depth-Anything)
        model_size = 518
        frames_resized = F.interpolate(
            frames_flat, size=(model_size, model_size), mode="bilinear", align_corners=False
        )

        # Run depth prediction (differentiably)
        pred_depth = self._model(frames_resized).predicted_depth  # (B*T, h, w)
        # Normalise predicted depth to [0, 1]
        pred_min = pred_depth.amin(dim=(-2, -1), keepdim=True)
        pred_max = pred_depth.amax(dim=(-2, -1), keepdim=True)
        pred_depth = (pred_depth - pred_min) / (pred_max - pred_min + 1e-8)

        # Resize target to match prediction spatial dims
        target_resized = F.interpolate(
            target.to(device).to(pred_depth.dtype),
            size=pred_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Broadcast target across all frames: (1, 1, h, w) → (B*T, h, w)
        target_expanded = target_resized.squeeze(1).expand_as(pred_depth)

        loss = F.mse_loss(pred_depth, target_expanded)
        return loss

    def should_guide_at_step(self, step: int, total_steps: int) -> bool:
        return step < total_steps * self._guidance_ratio
