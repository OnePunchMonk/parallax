"""Adaptive composite guidance with gradient normalization and conflict detection.

This replaces naive alpha-weighted guidance composition with a principled
approach that:
1. Normalises each module's gradient to unit norm (prevents dominance)
2. Detects conflicts between gradient directions (negative cosine similarity)
3. Supports per-phase temporal scheduling of different signals
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from parallax.guidance.base import GuidanceModule

logger = logging.getLogger(__name__)


class AdaptiveCompositeGuidance(GuidanceModule):
    """Compose multiple guidance modules with adaptive gradient normalization.

    Unlike :class:`CompositeGuidance` which sums weighted losses, this module
    operates at the **gradient level**: it computes per-module gradients
    independently, normalises them, detects conflicts, and produces a combined
    gradient update.

    This requires access to the latent tensor and VAE, so it overrides the
    standard ``compute_loss`` interface to return a *pseudo-loss* whose
    gradient is the adaptively combined gradient.

    Parameters
    ----------
    modules:
        List of ``(GuidanceModule, weight)`` tuples.
    normalize_gradients:
        If ``True``, normalize each module's gradient to unit norm before
        combining.
    conflict_threshold:
        Cosine similarity threshold below which two gradients are considered
        conflicting. When conflict is detected, the lower-weight module's
        contribution is reduced.
    schedule:
        Optional per-module temporal schedule as dict mapping module name
        to ``(start_ratio, end_ratio)`` tuples. E.g.,
        ``{"depth": (0.0, 0.5), "flow": (0.3, 0.8)}``.
    """

    def __init__(
        self,
        modules: List[Tuple[GuidanceModule, float]],
        normalize_gradients: bool = True,
        conflict_threshold: float = -0.1,
        schedule: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self._modules = modules
        self._normalize_gradients = normalize_gradients
        self._conflict_threshold = conflict_threshold
        self._schedule = schedule or {}

    # -- Properties -------------------------------------------------------- #

    @property
    def name(self) -> str:
        names = "+".join(m.name for m, _ in self._modules)
        return f"adaptive({names})"

    @property
    def default_scale(self) -> float:
        return 1.0

    # -- Lifecycle --------------------------------------------------------- #

    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        for mod, _ in self._modules:
            mod.load_model(device, dtype)

    # -- Targets ----------------------------------------------------------- #

    def prepare_targets(self, **kwargs: Any) -> Dict[str, Any]:
        """Prepare targets for all sub-modules.

        Pass targets for each sub-module under a key matching its ``name``.
        """
        all_targets: Dict[str, Any] = {}
        for mod, _ in self._modules:
            mod_kwargs = kwargs.get(mod.name, {})
            all_targets[mod.name] = mod.prepare_targets(**mod_kwargs)
        return all_targets

    # -- Scheduling -------------------------------------------------------- #

    def _module_active_at_step(
        self, mod: GuidanceModule, step: int, total_steps: int
    ) -> bool:
        """Check if a module should be active at this step."""
        if mod.name in self._schedule:
            start_ratio, end_ratio = self._schedule[mod.name]
            progress = step / total_steps
            return start_ratio <= progress < end_ratio
        return mod.should_guide_at_step(step, total_steps)

    # -- Loss -------------------------------------------------------------- #

    def compute_loss(
        self,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        timestep: int,
    ) -> Tensor:
        """Compute adaptively weighted composite loss.

        The adaptive weighting happens at the gradient level during the
        pipeline's ``_apply_guidance_step``. Here we compute the weighted
        sum of losses (same as ``CompositeGuidance``) but log additional
        diagnostic information about gradient interactions.

        For full adaptive behavior (gradient normalization and conflict
        detection), use ``compute_adaptive_gradients()`` instead, which
        is called by the pipeline when it detects an AdaptiveCompositeGuidance
        module.
        """
        total_loss = torch.tensor(0.0, device=decoded_frames.device)
        for mod, weight in self._modules:
            sub_targets = targets.get(mod.name, {})
            sub_loss = mod.compute_loss(decoded_frames, sub_targets, timestep)
            total_loss = total_loss + weight * mod.default_scale * sub_loss
        return total_loss

    def compute_adaptive_gradients(
        self,
        latents_for_grad: Tensor,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        step_index: int,
        total_steps: int,
    ) -> Tensor:
        """Compute adaptively combined gradient from all modules.

        This is the core of the adaptive composition:
        1. Compute each module's loss independently
        2. Backprop each to get per-module gradients
        3. Normalize gradients (optional)
        4. Detect and resolve conflicts
        5. Return combined gradient

        Parameters
        ----------
        latents_for_grad:
            Latent tensor with ``requires_grad=True``.
        decoded_frames:
            Decoded frames tensor (in the autograd graph).
        targets:
            All module targets.
        step_index:
            Current denoising step.
        total_steps:
            Total denoising steps.

        Returns
        -------
        Combined guidance gradient of the same shape as ``latents_for_grad``.
        """
        module_grads: List[Tuple[str, float, Tensor]] = []

        for mod, weight in self._modules:
            if not self._module_active_at_step(mod, step_index, total_steps):
                continue

            sub_targets = targets.get(mod.name, {})
            if not sub_targets:
                continue

            # Compute loss for this module
            loss = mod.compute_loss(decoded_frames, sub_targets, step_index)
            scaled_loss = weight * mod.default_scale * loss

            if scaled_loss.requires_grad:
                # Compute gradient w.r.t. latent
                grad = torch.autograd.grad(
                    scaled_loss, latents_for_grad,
                    retain_graph=True, allow_unused=True,
                )[0]
                if grad is None:
                    continue
                module_grads.append((mod.name, weight, grad))
                logger.debug(
                    "Module %s: loss=%.4f, grad_norm=%.6f",
                    mod.name, loss.item(), grad.norm().item(),
                )

        if not module_grads:
            return torch.zeros_like(latents_for_grad)

        # Normalize gradients if requested
        if self._normalize_gradients:
            normalized = []
            for name, weight, grad in module_grads:
                grad_norm = grad.norm()
                if grad_norm > 1e-8:
                    normalized.append((name, weight, grad / grad_norm * weight))
                else:
                    normalized.append((name, weight, grad))
            module_grads = normalized

        # Conflict detection and resolution
        if len(module_grads) > 1:
            module_grads = self._resolve_conflicts(module_grads)

        # Combine gradients
        combined = torch.zeros_like(latents_for_grad)
        for name, weight, grad in module_grads:
            combined = combined + grad

        return combined

    def _resolve_conflicts(
        self,
        grads: List[Tuple[str, float, Tensor]],
    ) -> List[Tuple[str, float, Tensor]]:
        """Detect and resolve gradient conflicts between modules.

        When two modules produce gradients with negative cosine similarity
        (pointing in opposite directions), reduce the contribution of the
        lower-weight module.
        """
        resolved = list(grads)

        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                name_i, weight_i, grad_i = resolved[i]
                name_j, weight_j, grad_j = resolved[j]

                # Compute cosine similarity between flattened gradients
                cos_sim = F.cosine_similarity(
                    grad_i.flatten().unsqueeze(0),
                    grad_j.flatten().unsqueeze(0),
                ).item()

                if cos_sim < self._conflict_threshold:
                    # Conflict detected: reduce the lower-weight gradient
                    # by projecting out the conflicting component
                    reduction = max(0.0, 1.0 + cos_sim)  # 0 at cos=-1, 1 at cos=0

                    if weight_i < weight_j:
                        resolved[i] = (name_i, weight_i, grad_i * reduction)
                        logger.debug(
                            "Conflict: %s vs %s (cos=%.3f), reducing %s by %.1f%%",
                            name_i, name_j, cos_sim, name_i,
                            (1 - reduction) * 100,
                        )
                    else:
                        resolved[j] = (name_j, weight_j, grad_j * reduction)
                        logger.debug(
                            "Conflict: %s vs %s (cos=%.3f), reducing %s by %.1f%%",
                            name_i, name_j, cos_sim, name_j,
                            (1 - reduction) * 100,
                        )

        return resolved

    def should_guide_at_step(self, step: int, total_steps: int) -> bool:
        return any(
            self._module_active_at_step(m, step, total_steps)
            for m, _ in self._modules
        )
