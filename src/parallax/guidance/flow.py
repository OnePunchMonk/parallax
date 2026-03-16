"""Optical flow consistency guidance using RAFT.

Uses optical flow estimation between adjacent decoded frames as a temporal
consistency regularizer. This addresses the key limitation of per-frame
guidance: independently guided frames can flicker or exhibit temporal
inconsistencies.

The flow guidance can operate in two modes:
1. **Smoothness mode** (default): Penalizes large, sudden flow changes
   between adjacent frames, encouraging smooth motion.
2. **Target mode**: Steers generation toward a user-provided target flow
   field, enabling explicit motion control.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from parallax.guidance.base import GuidanceModule


class FlowGuidance(GuidanceModule):
    """Temporal consistency guidance via optical flow estimation.

    Uses `RAFT <https://arxiv.org/abs/2003.12039>`_ (via torchvision) to
    estimate optical flow between consecutive decoded frames and penalises
    either flow magnitude (smoothness) or deviation from a target flow field.

    Parameters
    ----------
    model_name:
        RAFT variant: ``"raft_small"`` (fast) or ``"raft_large"`` (accurate).
    guidance_scale:
        Per-module guidance weight alpha.
    mode:
        ``"smoothness"`` — penalise large flow magnitudes (temporal stability).
        ``"target"`` — match a user-provided target flow field.
        ``"warp"`` — penalise warping error (frame t warped by flow should
        match frame t+1).
    guidance_ratio:
        Fraction of denoising steps to apply guidance (0-1).
    flow_scale:
        Scale factor for flow magnitude in smoothness loss. Smaller values
        allow more motion before penalty kicks in.
    """

    def __init__(
        self,
        model_name: str = "raft_small",
        guidance_scale: float = 35.0,
        mode: str = "smoothness",
        guidance_ratio: float = 0.4,
        flow_scale: float = 1.0,
    ) -> None:
        self._model_name = model_name
        self._guidance_scale_value = guidance_scale
        self._mode = mode
        self._guidance_ratio = guidance_ratio
        self._flow_scale = flow_scale
        self._model: Optional[torch.nn.Module] = None

    # -- Properties -------------------------------------------------------- #

    @property
    def name(self) -> str:
        return "flow"

    @property
    def default_scale(self) -> float:
        return self._guidance_scale_value

    # -- Lifecycle --------------------------------------------------------- #

    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        """Load RAFT optical flow model from torchvision."""
        from torchvision.models.optical_flow import (
            raft_small,
            raft_large,
            Raft_Small_Weights,
            Raft_Large_Weights,
        )

        if self._model_name == "raft_large":
            self._model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
        else:
            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device)

        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    # -- Targets ----------------------------------------------------------- #

    def prepare_targets(
        self,
        target_flow: Optional[Union[Tensor, str]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare flow targets.

        Parameters
        ----------
        target_flow:
            For ``"target"`` mode: a ``(1, 2, H, W)`` flow field tensor
            (u, v components) or ``None`` for smoothness/warp modes.
        """
        targets: Dict[str, Any] = {"mode": self._mode}

        if target_flow is not None and isinstance(target_flow, Tensor):
            if device is not None:
                target_flow = target_flow.to(device)
            targets["target_flow"] = target_flow

        return targets

    # -- Loss -------------------------------------------------------------- #

    def _estimate_flow(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """Estimate optical flow from frame1 to frame2.

        Parameters
        ----------
        frame1, frame2:
            ``(B, 3, H, W)`` frames in ``[0, 1]``.

        Returns
        -------
        ``(B, 2, H, W)`` optical flow field.
        """
        assert self._model is not None

        # RAFT expects input in [0, 255] range and size divisible by 8
        h, w = frame1.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            frame1 = F.pad(frame1, (0, pad_w, 0, pad_h), mode="replicate")
            frame2 = F.pad(frame2, (0, pad_w, 0, pad_h), mode="replicate")

        f1 = frame1 * 255.0
        f2 = frame2 * 255.0

        # RAFT returns a list of flow predictions (iterative refinement)
        # Take the last (most refined) prediction
        flow_predictions = self._model(f1, f2)
        flow = flow_predictions[-1]  # (B, 2, H, W)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :h, :w]

        return flow

    @staticmethod
    def _warp_frame(frame: Tensor, flow: Tensor) -> Tensor:
        """Warp a frame using an optical flow field (differentiable).

        Parameters
        ----------
        frame:
            ``(B, C, H, W)`` source frame.
        flow:
            ``(B, 2, H, W)`` flow field (u, v).

        Returns
        -------
        ``(B, C, H, W)`` warped frame.
        """
        B, C, H, W = frame.shape
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=frame.dtype),
            torch.arange(W, device=frame.device, dtype=frame.dtype),
            indexing="ij",
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        # Apply flow offset
        new_x = grid_x + flow[:, 0]  # (B, H, W)
        new_y = grid_y + flow[:, 1]

        # Normalise to [-1, 1] for grid_sample
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0

        grid = torch.stack([new_x, new_y], dim=-1)  # (B, H, W, 2)
        warped = F.grid_sample(
            frame, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped

    def compute_loss(
        self,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        timestep: int,
    ) -> Tensor:
        """Compute optical flow consistency loss.

        Parameters
        ----------
        decoded_frames:
            ``(B, C, T, H, W)`` -- decoded pixels in ``[0, 1]``.
        targets:
            Dict with ``mode`` and optionally ``target_flow``.
        """
        assert self._model is not None, "Call load_model() first."

        B, C, T, H, W = decoded_frames.shape
        if T < 2:
            # Need at least 2 frames for flow computation
            return torch.tensor(0.0, device=decoded_frames.device, requires_grad=True)

        mode = targets.get("mode", "smoothness")

        # Resize frames for RAFT (needs reasonable resolution)
        flow_size = min(H, W, 256)
        scale = flow_size / min(H, W)
        new_h = int(H * scale) // 8 * 8
        new_w = int(W * scale) // 8 * 8

        frames_resized = F.interpolate(
            decoded_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W),
            size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).reshape(B, T, C, new_h, new_w)

        total_loss = torch.tensor(0.0, device=decoded_frames.device)
        n_pairs = 0

        for t_idx in range(T - 1):
            frame1 = frames_resized[:, t_idx]  # (B, C, h, w)
            frame2 = frames_resized[:, t_idx + 1]

            flow = self._estimate_flow(frame1, frame2)  # (B, 2, h, w)

            if mode == "smoothness":
                # Penalise flow magnitude — encourage temporal stability
                flow_magnitude = (flow ** 2).sum(dim=1).sqrt()  # (B, h, w)
                loss_pair = (flow_magnitude * self._flow_scale).mean()

            elif mode == "warp":
                # Warp frame1 by flow, compare to frame2
                warped = self._warp_frame(frame1, flow)
                loss_pair = F.mse_loss(warped, frame2)

            elif mode == "target":
                target_flow = targets.get("target_flow")
                if target_flow is not None:
                    target_resized = F.interpolate(
                        target_flow.to(flow.device).to(flow.dtype),
                        size=flow.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                    target_expanded = target_resized.expand_as(flow)
                    loss_pair = F.mse_loss(flow, target_expanded)
                else:
                    loss_pair = torch.tensor(0.0, device=decoded_frames.device)
            else:
                raise ValueError(f"Unknown flow mode: {mode}")

            total_loss = total_loss + loss_pair
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss

    def should_guide_at_step(self, step: int, total_steps: int) -> bool:
        return step < total_steps * self._guidance_ratio
