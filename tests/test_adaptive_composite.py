"""Unit tests for AdaptiveCompositeGuidance."""

from __future__ import annotations

import pytest
import torch

from parallax.guidance.base import GuidanceModule


class MockModule(GuidanceModule):
    """Simple mock guidance module for testing composition."""

    def __init__(self, name: str, scale: float = 10.0, ratio: float = 0.5):
        self._name = name
        self._scale = scale
        self._ratio = ratio

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_scale(self) -> float:
        return self._scale

    def load_model(self, device, dtype):
        pass

    def prepare_targets(self, **kwargs):
        return {"dummy": True}

    def compute_loss(self, decoded_frames, targets, timestep):
        return decoded_frames.mean()

    def should_guide_at_step(self, step, total_steps):
        return step < total_steps * self._ratio


class TestAdaptiveCompositeGuidance:
    """Tests for adaptive gradient-normalized composition."""

    def test_name(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod_a = MockModule("depth")
        mod_b = MockModule("normal")
        comp = AdaptiveCompositeGuidance([(mod_a, 1.0), (mod_b, 0.9)])
        assert comp.name == "adaptive(depth+normal)"

    def test_load_model_delegates(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod_a = MockModule("depth")
        mod_b = MockModule("semantic")
        comp = AdaptiveCompositeGuidance([(mod_a, 1.0), (mod_b, 0.5)])
        # Should not raise
        comp.load_model(torch.device("cpu"), torch.float32)

    def test_prepare_targets(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod_a = MockModule("depth")
        mod_b = MockModule("semantic")
        comp = AdaptiveCompositeGuidance([(mod_a, 1.0), (mod_b, 0.5)])
        targets = comp.prepare_targets(depth={}, semantic={})
        assert "depth" in targets
        assert "semantic" in targets

    def test_should_guide_any_active(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod_a = MockModule("depth", ratio=0.3)
        mod_b = MockModule("semantic", ratio=0.7)
        comp = AdaptiveCompositeGuidance([(mod_a, 1.0), (mod_b, 0.5)])

        assert comp.should_guide_at_step(5, 20) is True   # both active
        assert comp.should_guide_at_step(7, 20) is True   # only semantic
        assert comp.should_guide_at_step(15, 20) is False  # neither

    def test_schedule_overrides_module_ratio(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod = MockModule("depth", ratio=0.5)
        comp = AdaptiveCompositeGuidance(
            [(mod, 1.0)],
            schedule={"depth": (0.0, 0.2)},  # Only first 20%
        )

        assert comp.should_guide_at_step(1, 20) is True   # 5% < 20%
        assert comp.should_guide_at_step(5, 20) is False  # 25% > 20%

    def test_compute_loss_returns_scalar(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod_a = MockModule("depth")
        mod_b = MockModule("semantic")
        comp = AdaptiveCompositeGuidance([(mod_a, 1.0), (mod_b, 0.5)])

        frames = torch.rand(1, 3, 2, 64, 64)
        targets = comp.prepare_targets(depth={}, semantic={})
        loss = comp.compute_loss(frames, targets, timestep=0)
        assert loss.ndim == 0  # scalar

    def test_compute_adaptive_gradients(self):
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        mod_a = MockModule("depth", scale=1.0)
        mod_b = MockModule("semantic", scale=1.0)
        comp = AdaptiveCompositeGuidance(
            [(mod_a, 1.0), (mod_b, 1.0)],
            normalize_gradients=True,
        )

        latent = torch.randn(1, 4, 2, 8, 8, requires_grad=True)
        frames = torch.randn(1, 3, 2, 32, 32, requires_grad=True)
        targets = comp.prepare_targets(depth={}, semantic={})

        grad = comp.compute_adaptive_gradients(
            latents_for_grad=latent,
            decoded_frames=frames,
            targets=targets,
            step_index=0,
            total_steps=20,
        )
        assert grad.shape == latent.shape

    def test_conflict_resolution(self):
        """Test that conflicting gradients are reduced."""
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        comp = AdaptiveCompositeGuidance([], conflict_threshold=-0.1)

        # Create opposing gradients
        grad_a = torch.ones(1, 4, 2, 8, 8)
        grad_b = -torch.ones(1, 4, 2, 8, 8)

        grads = [("a", 1.0, grad_a), ("b", 0.5, grad_b)]
        resolved = comp._resolve_conflicts(grads)

        # The lower-weight gradient (b) should be reduced
        _, _, resolved_b = resolved[1]
        assert resolved_b.norm() < grad_b.norm()

    def test_no_conflict_passes_through(self):
        """Aligned gradients should not be modified."""
        from parallax.guidance.adaptive_composite import AdaptiveCompositeGuidance

        comp = AdaptiveCompositeGuidance([], conflict_threshold=-0.1)

        grad_a = torch.ones(1, 4, 2, 8, 8)
        grad_b = torch.ones(1, 4, 2, 8, 8) * 0.5

        grads = [("a", 1.0, grad_a), ("b", 1.0, grad_b)]
        resolved = comp._resolve_conflicts(grads)

        # No conflict, gradients unchanged
        _, _, resolved_a = resolved[0]
        _, _, resolved_b = resolved[1]
        assert torch.allclose(resolved_a, grad_a)
        assert torch.allclose(resolved_b, grad_b)
