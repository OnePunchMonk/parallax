"""Visualization utilities for guided I2V generation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor


def tensor_to_numpy_frames(
    video_tensor: Tensor,
    value_range: str = "0_1",
) -> np.ndarray:
    """Convert a video tensor to a numpy array of uint8 frames.

    Parameters
    ----------
    video_tensor:
        ``(B, C, T, H, W)`` or ``(C, T, H, W)`` or ``(T, H, W, C)``.
    value_range:
        ``"0_1"`` or ``"-1_1"`` — how to interpret pixel values.

    Returns
    -------
    ``(T, H, W, C)`` uint8 numpy array.
    """
    t = video_tensor.detach().cpu().float()

    # Normalise shape to (T, C, H, W)
    if t.ndim == 5:
        t = t[0]  # drop batch
    if t.shape[0] == 3 or t.shape[0] == 1:
        # (C, T, H, W) → (T, C, H, W)
        t = t.permute(1, 0, 2, 3)
    elif t.ndim == 4 and t.shape[-1] in (1, 3):
        # Already (T, H, W, C)
        t = t.permute(0, 3, 1, 2)  # → (T, C, H, W)

    if value_range == "-1_1":
        t = (t + 1.0) * 0.5

    t = t.clamp(0, 1) * 255.0
    # (T, C, H, W) → (T, H, W, C)
    t = t.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    return t


def export_video(
    frames: Union[Tensor, np.ndarray, List],
    output_path: Union[str, Path],
    fps: int = 16,
) -> Path:
    """Export frames to an MP4 video file.

    Parameters
    ----------
    frames:
        Video frames — tensor ``(B, C, T, H, W)`` or numpy ``(T, H, W, C)``,
        or list of PIL Images.
    output_path:
        Destination ``.mp4`` path.
    fps:
        Frames per second.

    Returns
    -------
    The resolved output path.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, Tensor):
        frames_np = tensor_to_numpy_frames(frames)
    elif isinstance(frames, list):
        # Assume list of PIL Images
        frames_np = np.stack([np.array(f) for f in frames])
    else:
        frames_np = frames

    T, H, W, C = frames_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for i in range(T):
        frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return output_path


def export_comparison_video(
    guided_frames: Union[Tensor, np.ndarray],
    unguided_frames: Union[Tensor, np.ndarray],
    output_path: Union[str, Path],
    fps: int = 16,
    labels: tuple = ("Guided", "Unguided"),
) -> Path:
    """Create a side-by-side comparison video.

    Parameters
    ----------
    guided_frames:
        Frames from the guided generation.
    unguided_frames:
        Frames from baseline (unguided) generation.
    output_path:
        Destination ``.mp4`` path.
    fps:
        Frames per second.
    labels:
        Labels for the left and right panels.

    Returns
    -------
    The resolved output path.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(guided_frames, Tensor):
        guided_np = tensor_to_numpy_frames(guided_frames)
    else:
        guided_np = guided_frames

    if isinstance(unguided_frames, Tensor):
        unguided_np = tensor_to_numpy_frames(unguided_frames)
    else:
        unguided_np = unguided_frames

    T = min(len(guided_np), len(unguided_np))
    H, W, C = guided_np[0].shape
    canvas_w = W * 2 + 20  # 20px gap

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, H))

    for i in range(T):
        canvas = np.zeros((H, canvas_w, 3), dtype=np.uint8)
        canvas[:, :W, :] = guided_np[i]
        canvas[:, W + 20 :, :] = unguided_np[i]

        # Add labels
        cv2.putText(
            canvas, labels[0], (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            canvas, labels[1], (W + 30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        writer.write(canvas_bgr)

    writer.release()
    return output_path


def overlay_depth_on_frames(
    frames: Union[Tensor, np.ndarray],
    depth_maps: Union[Tensor, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay depth maps on video frames as a coloured heat map.

    Parameters
    ----------
    frames:
        ``(T, H, W, 3)`` uint8 array.
    depth_maps:
        ``(T, H, W)`` float array normalised to ``[0, 1]``.
    alpha:
        Blending weight for the overlay.

    Returns
    -------
    ``(T, H, W, 3)`` uint8 array with depth overlay.
    """
    import cv2

    if isinstance(frames, Tensor):
        frames = tensor_to_numpy_frames(frames)
    if isinstance(depth_maps, Tensor):
        depth_maps = depth_maps.detach().cpu().float().numpy()

    T, H, W, C = frames.shape
    result = np.zeros_like(frames)

    for i in range(T):
        depth = depth_maps[i] if i < len(depth_maps) else depth_maps[-1]
        depth_uint8 = (depth * 255).clip(0, 255).astype(np.uint8)
        depth_colour = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_colour = cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB)

        if depth_colour.shape[:2] != (H, W):
            depth_colour = cv2.resize(depth_colour, (W, H))

        blended = cv2.addWeighted(frames[i], 1 - alpha, depth_colour, alpha, 0)
        result[i] = blended

    return result
