"""GuidedI2VPipeline — Zero-shot controllable Image-to-Video generation.

This module wraps a standard diffusers I2V pipeline (e.g. ``WanImageToVideoPipeline``)
and injects gradient-based guidance from vision foundation models at each
denoising step.  Guidance is **training-free** — it operates purely at
inference time by back-propagating through ``(VisionModel ∘ VAE.decode)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm

from parallax.guidance.base import GuidanceModule
from parallax.utils.latent_utils import (
    apply_guidance_gradient,
    compute_x0_prediction,
    differentiable_decode,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class GuidanceConfig:
    """Configuration for the guided denoising loop.

    Parameters
    ----------
    guidance_scale:
        Global multiplier for all guidance gradients (α).
    guidance_steps_ratio:
        Apply guidance for the first ``ratio`` fraction of denoising steps.
    frame_subsample_rate:
        Decode every *n*-th latent frame for guidance (memory optimisation).
    grad_clip:
        Gradient clipping norm (``None`` to disable).
    decode_dtype:
        dtype for VAE decode (``torch.float32`` recommended for stability).
    """

    guidance_scale: float = 50.0
    guidance_steps_ratio: float = 0.5
    frame_subsample_rate: int = 4
    grad_clip: Optional[float] = 1.0
    decode_dtype: torch.dtype = torch.float32


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #


class GuidedI2VPipeline:
    """Training-free guided Image-to-Video pipeline.

    This wraps *any* diffusers-based video generation pipeline and injects
    vision-model guidance into its denoising loop.

    Example
    -------
    ::

        from diffusers import WanImageToVideoPipeline
        from parallax.pipeline import GuidedI2VPipeline, GuidanceConfig
        from parallax.guidance import DepthGuidance

        base = WanImageToVideoPipeline.from_pretrained(...)
        depth = DepthGuidance()
        guided = GuidedI2VPipeline(base, [depth], GuidanceConfig())
        frames = guided.generate(image=img, prompt="...", target_depth=depth_map)
    """

    def __init__(
        self,
        base_pipeline,
        guidance_modules: List[GuidanceModule],
        config: Optional[GuidanceConfig] = None,
    ) -> None:
        self.pipe = base_pipeline
        self.guidance_modules = guidance_modules
        self.config = config or GuidanceConfig()

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #

    def load_guidance_models(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Load all vision foundation models used for guidance."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or torch.float32
        for mod in self.guidance_modules:
            logger.info("Loading guidance model: %s", mod.name)
            mod.load_model(device, dtype)
            logger.info("  ✓ %s loaded", mod.name)

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, Tensor],
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 33,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_cfg_scale: float = 5.0,
        callback: Optional[Callable] = None,
        **target_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a guided I2V video.

        Parameters
        ----------
        image:
            Conditioning image (PIL or tensor).
        prompt:
            Text prompt.
        negative_prompt:
            Negative text prompt for CFG.
        num_frames:
            Number of video frames to generate.
        height, width:
            Output resolution (``None`` = model default).
        num_inference_steps:
            Number of denoising steps.
        guidance_cfg_scale:
            Classifier-free guidance scale (standard CFG, not ours).
        callback:
            Optional callback ``fn(step, timestep, latent)`` called each step.
        **target_kwargs:
            Targets for guidance modules.  Pass targets using the module
            name as prefix, e.g. ``target_depth=..., reference_image=...``.

        Returns
        -------
        Dict with ``"frames"`` (list of PIL Images), ``"latents"`` (final),
        and ``"guided_steps"`` (number of steps where guidance was applied).
        """
        # --- 1. Prepare targets for all guidance modules ---
        all_targets: Dict[str, Dict[str, Any]] = {}
        device = self._get_device()

        for mod in self.guidance_modules:
            # Match kwargs to module by naming convention
            mod_targets = self._extract_module_targets(mod, target_kwargs)
            if mod_targets:
                all_targets[mod.name] = mod.prepare_targets(
                    device=device, **mod_targets
                )
            else:
                logger.warning(
                    "No targets provided for guidance module '%s'", mod.name
                )

        # --- 2. Encode text ---
        pipe = self.pipe
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_cfg_scale,
            output_type="latent",
        )
        if height is not None:
            pipe_kwargs["height"] = height
        if width is not None:
            pipe_kwargs["width"] = width

        # --- 3. Run the base pipeline but intercept the denoising loop ---
        # We use the `callback_on_step_end` mechanism from diffusers
        guided_step_count = 0

        def guidance_callback(pipe_instance, step_index, timestep, callback_kwargs):
            nonlocal guided_step_count
            latents = callback_kwargs["latents"]

            # Check if we should apply guidance at this step
            should_guide = any(
                m.should_guide_at_step(step_index, num_inference_steps)
                for m in self.guidance_modules
            )

            if not should_guide or not all_targets:
                if callback:
                    callback(step_index, timestep, latents)
                return callback_kwargs

            # --- Apply gradient guidance ---
            guided_latents = self._apply_guidance_step(
                latents=latents,
                timestep=timestep,
                step_index=step_index,
                total_steps=num_inference_steps,
                targets=all_targets,
            )
            guided_step_count += 1

            callback_kwargs["latents"] = guided_latents
            if callback:
                callback(step_index, timestep, guided_latents)
            return callback_kwargs

        pipe_kwargs["callback_on_step_end"] = guidance_callback
        pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        # --- 4. Run generation ---
        logger.info(
            "Starting guided I2V generation: %d steps, %d guidance modules",
            num_inference_steps,
            len(self.guidance_modules),
        )
        output = pipe(**pipe_kwargs)

        # --- 5. Decode final latents ---
        latents = output.frames  # When output_type="latent"
        if isinstance(latents, Tensor):
            # Decode through VAE
            with torch.no_grad():
                decoded = pipe.vae.decode(
                    latents / pipe.vae.config.scaling_factor
                ).sample
            frames = self._latents_to_pil(decoded)
        else:
            frames = latents  # Already decoded

        return {
            "frames": frames,
            "latents": latents if isinstance(latents, Tensor) else None,
            "guided_steps": guided_step_count,
        }

    # ------------------------------------------------------------------ #
    # Core guidance step                                                  #
    # ------------------------------------------------------------------ #

    def _apply_guidance_step(
        self,
        latents: Tensor,
        timestep: Union[int, Tensor],
        step_index: int,
        total_steps: int,
        targets: Dict[str, Dict[str, Any]],
    ) -> Tensor:
        """Apply gradient guidance to latents at a single denoising step.

        1. Enable gradients on the latent
        2. Decode a subset of frames through the VAE (differentiably)
        3. Run each guidance module, accumulate losses
        4. Backprop to get ∂L/∂latent
        5. Update latent with gradient descent step
        """
        cfg = self.config
        vae = self.pipe.vae

        # Work with gradients enabled
        latents_for_grad = latents.detach().clone().requires_grad_(True)

        # --- Differentiable decode of subsampled frames ---
        B, C, T_lat, H_lat, W_lat = latents_for_grad.shape
        frame_indices = list(range(0, T_lat, cfg.frame_subsample_rate))
        if not frame_indices:
            frame_indices = [0]

        decoded = differentiable_decode(
            vae,
            latents_for_grad,
            frame_indices=frame_indices,
            decode_chunk_size=1,
        )

        # Normalise to [0, 1] (VAE output is typically [-1, 1])
        decoded_01 = (decoded + 1.0) * 0.5
        decoded_01 = decoded_01.clamp(0, 1)

        # --- Compute guidance losses ---
        total_loss = torch.tensor(0.0, device=latents.device, requires_grad=True)

        for mod in self.guidance_modules:
            if not mod.should_guide_at_step(step_index, total_steps):
                continue

            mod_targets = targets.get(mod.name, {})
            if not mod_targets:
                continue

            loss = mod.compute_loss(decoded_01, mod_targets, step_index)
            total_loss = total_loss + mod.default_scale * loss

        if total_loss.requires_grad:
            total_loss.backward()

            if latents_for_grad.grad is not None:
                guided_latents = apply_guidance_gradient(
                    latents.detach().clone(),
                    latents_for_grad.grad.detach(),
                    guidance_scale=cfg.guidance_scale,
                    grad_clip=cfg.grad_clip,
                )

                logger.debug(
                    "Step %d: guidance loss=%.4f, grad_norm=%.6f",
                    step_index,
                    total_loss.item(),
                    latents_for_grad.grad.norm().item(),
                )
                return guided_latents
            else:
                logger.warning("Step %d: no gradient computed", step_index)

        return latents

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_device(self) -> torch.device:
        """Infer the device from the base pipeline."""
        if hasattr(self.pipe, "device"):
            return self.pipe.device
        if hasattr(self.pipe, "_execution_device"):
            return self.pipe._execution_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _extract_module_targets(
        mod: GuidanceModule, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract target kwargs for a specific module.

        Naming convention:
        - ``target_depth`` → depth module receives ``target_depth``
        - ``reference_image`` → semantic module receives ``reference_image``
        - ``target_mask`` → segmentation module receives ``target_mask``
        """
        target_map = {
            "depth": ["target_depth"],
            "semantic": ["reference_image"],
            "segmentation": ["target_mask", "input_points", "input_boxes"],
        }
        keys = target_map.get(mod.name, [])
        return {k: kwargs[k] for k in keys if k in kwargs}

    @staticmethod
    def _latents_to_pil(decoded: Tensor) -> List[Image.Image]:
        """Convert decoded tensor ``(B, C, T, H, W)`` to list of PIL Images."""
        decoded_01 = (decoded + 1.0) * 0.5
        decoded_01 = decoded_01.clamp(0, 1)

        # (B, C, T, H, W) → (T, H, W, C)
        frames = decoded_01[0].permute(1, 2, 3, 0).cpu().float().numpy()
        pil_frames = []
        for i in range(frames.shape[0]):
            frame = (frames[i] * 255).clip(0, 255).astype("uint8")
            pil_frames.append(Image.fromarray(frame))
        return pil_frames
