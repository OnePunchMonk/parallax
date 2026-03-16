"""Unit tests for NormalGuidance module."""

from __future__ import annotations

import pytest
import torch


class TestNormalGuidance:
    """Tests for the NormalGuidance module."""

    def test_instantiation(self):
        from parallax.guidance.normal import NormalGuidance

        mod = NormalGuidance()
        assert mod.name == "normal"
        assert mod.default_scale == 45.0

    def test_prepare_targets_from_tensor(self):
        from parallax.guidance.normal import NormalGuidance

        mod = NormalGuidance()
        # Normal map as unit vectors: (1, 3, H, W)
        target = torch.randn(1, 3, 64, 64)
        targets = mod.prepare_targets(target_normal=target)
        assert "target_normal" in targets
        # Should be unit-normalized along dim=1
        norms = targets["target_normal"].norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_prepare_targets_from_pil(self):
        from PIL import Image
        from parallax.guidance.normal import NormalGuidance

        mod = NormalGuidance()
        # Create an RGB image representing a normal map
        img = Image.new("RGB", (64, 64), color=(128, 128, 255))  # Upward-facing normals
        targets = mod.prepare_targets(target_normal=img)
        assert "target_normal" in targets
        assert targets["target_normal"].shape == (1, 3, 64, 64)

    def test_should_guide_at_step(self):
        from parallax.guidance.normal import NormalGuidance

        mod = NormalGuidance(guidance_ratio=0.5)
        assert mod.should_guide_at_step(0, 20) is True
        assert mod.should_guide_at_step(9, 20) is True
        assert mod.should_guide_at_step(10, 20) is False

    def test_depth_to_normals_differentiable(self):
        """Verify depth-to-normal conversion preserves gradients."""
        from parallax.guidance.normal import NormalGuidance

        depth = torch.rand(2, 1, 32, 32, requires_grad=True)
        normals = NormalGuidance._depth_to_normals(depth)

        assert normals.shape == (2, 3, 32, 32)
        # Check unit normals
        norms = normals.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

        # Check gradient flow
        loss = normals.sum()
        loss.backward()
        assert depth.grad is not None
        assert depth.grad.shape == depth.shape

    def test_depth_to_normals_flat_surface(self):
        """A flat depth map should produce upward-pointing normals."""
        from parallax.guidance.normal import NormalGuidance

        depth = torch.ones(1, 1, 32, 32) * 0.5
        normals = NormalGuidance._depth_to_normals(depth)

        # For a flat surface (constant depth), normals should point in z direction
        # i.e., (-dz/dx, -dz/dy, 1) normalized = (0, 0, 1)
        z_component = normals[0, 2]  # Should be ~1.0
        assert z_component.mean().item() > 0.99
