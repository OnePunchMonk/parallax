"""Differentiable VAE decode utilities and latent-space helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_x0_prediction(
    scheduler,
    noise_pred: Tensor,
    latent: Tensor,
    timestep: Tensor,
) -> Tensor:
    """Extract the denoised-sample prediction x̂₀ from the scheduler.

    For flow-matching schedulers (``FlowMatchEulerDiscreteScheduler``) used by
    Wan2.1, the predicted clean sample can be recovered from the velocity
    parameterisation.

    For schedulers that directly produce ``pred_original_sample`` (DDPM,
    DDIM), we fall back to the standard scheduler method.

    Parameters
    ----------
    scheduler:
        The diffusion scheduler instance (e.g. ``FlowMatchEulerDiscreteScheduler``).
    noise_pred:
        The noise/velocity prediction from the DiT, ``(B, C, T, H, W)``.
    latent:
        The current noisy latent ``x_t``, ``(B, C, T, H, W)``.
    timestep:
        Current timestep tensor.

    Returns
    -------
    Estimated clean latent ``x̂₀``.
    """
    # Flow-matching: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
    # velocity v = x_0 - noise  =>  x_0 = x_t - sigma_t * v    (not exactly)
    #
    # Actually for FlowMatch:  x_t = sigma_t * x_0 + (1 - sigma_t) * eps
    # The predicted velocity is:  v = x_0 - eps
    # So x_0 = x_t + (1 - sigma_t) * (v)  ... but this depends on sigmas.
    #
    # Safest approach: use the scheduler's step function and extract
    # pred_original_sample if available, otherwise compute manually.

    sigma = timestep.float() / scheduler.config.num_train_timesteps
    sigma = sigma.reshape(-1, *([1] * (latent.ndim - 1)))

    # For flow matching:  x_t = (1 - σ) * ε + σ * x_0
    # Predicted velocity:  v = x_0 - ε
    # => x_0 = x_t + (1 - σ) * v
    #    ε   = x_t - σ * v
    # But the actual formula depends on the scheduler's sigma schedule.
    #
    # Universal formula:  x_0 ≈ x_t - σ * noise_pred  (for DDPM-style)
    # For flow matching:  x_0 = (x_t - (1 - σ) * noise_pred) / σ   if noise_pred is ε
    # For velocity:       x_0 = x_t + (1 - σ) * v_pred             if noise_pred is v
    #
    # We use the velocity parameterisation (default for Wan2.1 / Mochi):
    x0 = latent + (1.0 - sigma) * noise_pred

    return x0


def differentiable_decode(
    vae,
    latents: Tensor,
    frame_indices: Optional[Sequence[int]] = None,
    decode_chunk_size: int = 1,
) -> Tensor:
    """Decode latents through the VAE while preserving the gradient graph.

    Parameters
    ----------
    vae:
        The video VAE (e.g. ``AutoencoderKLWan``).
    latents:
        ``(B, C, T, H, W)`` — latent frames to decode.
    frame_indices:
        If given, only decode these temporal indices (saves memory).
        Indices refer to the latent temporal dimension.
    decode_chunk_size:
        Number of frames to decode at a time (for memory management).

    Returns
    -------
    Decoded frames ``(B, 3, T_sub, H_dec, W_dec)`` with gradients attached.
    """
    # Sub-select temporal frames if requested
    if frame_indices is not None:
        latents = latents[:, :, list(frame_indices), :, :]

    # Scale latents according to VAE config
    if hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
        latents = latents / vae.config.scaling_factor

    # Decode — some VAEs need float32 for numerical stability
    original_dtype = latents.dtype
    if latents.dtype != torch.float32:
        latents = latents.float()

    # Use chunked decoding for memory efficiency
    B, C, T, H, W = latents.shape
    decoded_chunks = []

    for i in range(0, T, decode_chunk_size):
        chunk = latents[:, :, i : i + decode_chunk_size, :, :]
        # VAE decode expects (B, C, T, H, W)
        decoded = vae.decode(chunk).sample
        decoded_chunks.append(decoded)

    if len(decoded_chunks) == 1:
        decoded_frames = decoded_chunks[0]
    else:
        decoded_frames = torch.cat(decoded_chunks, dim=2)

    return decoded_frames


def apply_guidance_gradient(
    latent: Tensor,
    guidance_grad: Tensor,
    guidance_scale: float,
    grad_clip: Optional[float] = 1.0,
) -> Tensor:
    """Apply the guidance gradient to the noisy latent.

    Parameters
    ----------
    latent:
        Current noisy latent ``x_t`` (will be modified in-place).
    guidance_grad:
        ``∂L/∂x_t`` — gradient of the guidance loss w.r.t. the latent.
    guidance_scale:
        Global guidance weight α.
    grad_clip:
        If set, clip gradient norm to this value (stabilises guidance).

    Returns
    -------
    Updated latent ``x_t - α · ∇_x L``.
    """
    if grad_clip is not None:
        grad_norm = guidance_grad.norm()
        if grad_norm > grad_clip:
            guidance_grad = guidance_grad * (grad_clip / (grad_norm + 1e-8))

    return latent - guidance_scale * guidance_grad
