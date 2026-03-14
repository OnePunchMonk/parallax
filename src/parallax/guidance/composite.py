"""Composite guidance — combine multiple guidance modules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from parallax.guidance.base import GuidanceModule


class CompositeGuidance(GuidanceModule):
    """Combine multiple guidance modules into a single weighted objective.

    Parameters
    ----------
    modules:
        List of ``(GuidanceModule, weight)`` tuples.
        - *weight* multiplies the module's own ``default_scale``
        - e.g. ``[(depth_mod, 1.0), (semantic_mod, 0.5)]``
    """

    def __init__(
        self,
        modules: List[Tuple[GuidanceModule, float]],
    ) -> None:
        self._modules = modules

    # -- Properties -------------------------------------------------------- #

    @property
    def name(self) -> str:
        names = "+".join(m.name for m, _ in self._modules)
        return f"composite({names})"

    @property
    def default_scale(self) -> float:
        return 1.0  # Weights are per-sub-module

    # -- Lifecycle --------------------------------------------------------- #

    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        for mod, _ in self._modules:
            mod.load_model(device, dtype)

    # -- Targets ----------------------------------------------------------- #

    def prepare_targets(self, **kwargs: Any) -> Dict[str, Any]:
        """Prepare targets for all sub-modules.

        Pass targets for each sub-module under a key matching its ``name``,
        e.g.::

            composite.prepare_targets(
                depth={"target_depth": depth_tensor},
                semantic={"reference_image": ref_image},
            )
        """
        all_targets: Dict[str, Any] = {}
        for mod, _ in self._modules:
            mod_kwargs = kwargs.get(mod.name, {})
            all_targets[mod.name] = mod.prepare_targets(**mod_kwargs)
        return all_targets

    # -- Loss -------------------------------------------------------------- #

    def compute_loss(
        self,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        timestep: int,
    ) -> Tensor:
        """Sum of weighted sub-module losses."""
        total_loss = torch.tensor(0.0, device=decoded_frames.device)
        for mod, weight in self._modules:
            sub_targets = targets.get(mod.name, {})
            sub_loss = mod.compute_loss(decoded_frames, sub_targets, timestep)
            total_loss = total_loss + weight * mod.default_scale * sub_loss
        return total_loss

    def should_guide_at_step(self, step: int, total_steps: int) -> bool:
        return any(m.should_guide_at_step(step, total_steps) for m, _ in self._modules)
