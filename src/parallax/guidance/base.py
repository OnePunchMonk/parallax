"""Abstract base class for guidance modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class GuidanceModule(ABC):
    """Base class for all test-time guidance modules.

    A guidance module computes a differentiable loss between the decoded video
    frames (produced during the denoising process) and a user-provided target
    signal.  The gradient of this loss w.r.t. the noisy latent is used to
    *steer* the diffusion model's sampling trajectory at inference time.

    Subclasses must implement:
        - ``compute_loss``   — returns a scalar loss tensor
        - ``prepare_targets`` — preprocesses user-provided targets
        - ``load_model``     — loads the underlying vision foundation model

    Lifecycle:
        1. ``load_model()``       — call once to load weights
        2. ``prepare_targets()``  — call once per generation request
        3. ``compute_loss()``     — called at every guided denoising step
    """

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g. ``'depth'``, ``'semantic'``)."""

    @property
    def default_scale(self) -> float:
        """Default guidance scale (α).  Override per-module as needed."""
        return 50.0

    # ------------------------------------------------------------------ #
    # Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        """Load the vision foundation model onto *device*."""

    @abstractmethod
    def prepare_targets(self, **kwargs: Any) -> Dict[str, Any]:
        """Pre-process user-provided targets into tensors.

        Returns a dict that will be forwarded to ``compute_loss`` at every
        guided step.
        """

    @abstractmethod
    def compute_loss(
        self,
        decoded_frames: Tensor,
        targets: Dict[str, Any],
        timestep: int,
    ) -> Tensor:
        """Compute a *scalar* guidance loss.

        Parameters
        ----------
        decoded_frames:
            ``(B, C, T_sub, H, W)`` — a subset of decoded frames for which
            gradients are tracked.
        targets:
            The dict returned by ``prepare_targets``.
        timestep:
            Current denoising timestep (useful for scheduling).

        Returns
        -------
        Scalar loss tensor with ``requires_grad=True``.
        """

    # ------------------------------------------------------------------ #
    # Optional hooks                                                      #
    # ------------------------------------------------------------------ #

    def should_guide_at_step(self, step: int, total_steps: int) -> bool:
        """Return ``True`` if guidance should be applied at this step.

        By default, guide for the first 50 % of steps.
        """
        return step < total_steps * 0.5

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_frames(frames: Tensor) -> Tensor:
        """Scale pixel values from ``[−1, 1]`` (VAE output) to ``[0, 1]``."""
        return (frames + 1.0) * 0.5

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r}, scale={self.default_scale})"
