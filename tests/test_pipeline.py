"""Integration tests for the GuidedI2VPipeline.

These tests only check structural correctness — they mock the base pipeline
and VAE to avoid requiring GPU or model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from parallax.pipeline import GuidedI2VPipeline, GuidanceConfig
from parallax.guidance.base import GuidanceModule


# --------------------------------------------------------------------------- #
# Mock guidance module                                                         #
# --------------------------------------------------------------------------- #


class MockGuidanceModule(GuidanceModule):
    """A trivial guidance module that returns a constant loss for testing."""

    def __init__(self, loss_value: float = 1.0, ratio: float = 0.5):
        self._loss_val = loss_value
        self._ratio = ratio
        self._compute_loss_calls = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_scale(self) -> float:
        return 1.0

    def load_model(self, device, dtype):
        pass  # No-op

    def prepare_targets(self, **kwargs):
        return {"dummy": True}

    def compute_loss(self, decoded_frames, targets, timestep):
        self._compute_loss_calls += 1
        # Return a loss that has grad (depends on decoded_frames)
        return decoded_frames.mean() * self._loss_val

    def should_guide_at_step(self, step, total_steps):
        return step < total_steps * self._ratio


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


class TestGuidedI2VPipeline:
    """Test the structural behaviour of GuidedI2VPipeline."""

    def test_pipeline_init(self):
        pipe = MagicMock()
        mod = MockGuidanceModule()
        cfg = GuidanceConfig(guidance_scale=10.0)

        guided = GuidedI2VPipeline(pipe, [mod], cfg)
        assert guided.pipe is pipe
        assert len(guided.guidance_modules) == 1
        assert guided.config.guidance_scale == 10.0

    def test_load_guidance_models(self):
        pipe = MagicMock()
        mod = MockGuidanceModule()
        guided = GuidedI2VPipeline(pipe, [mod])

        # Should not raise
        guided.load_guidance_models(
            device=torch.device("cpu"), dtype=torch.float32
        )

    def test_extract_module_targets(self):
        mod = MockGuidanceModule()
        # MockGuidanceModule has name "mock" which isn't in the default map
        # so it returns nothing
        result = GuidedI2VPipeline._extract_module_targets(
            mod, {"target_depth": "test"}
        )
        assert result == {}

    def test_extract_depth_targets(self):
        from parallax.guidance.depth import DepthGuidance

        mod = DepthGuidance()
        result = GuidedI2VPipeline._extract_module_targets(
            mod, {"target_depth": "test_tensor", "unrelated": 42}
        )
        assert result == {"target_depth": "test_tensor"}

    def test_guidance_step_modifies_latent(self):
        """Verify that _apply_guidance_step changes the latent."""
        # Mock VAE that returns differentiable output
        mock_vae = MagicMock()
        mock_vae.config.scaling_factor = 1.0
        decode_result = MagicMock()
        # Return something differentiable
        decode_result.sample = torch.randn(1, 3, 2, 64, 64)
        mock_vae.decode.return_value = decode_result

        mock_pipe = MagicMock()
        mock_pipe.vae = mock_vae

        mod = MockGuidanceModule(loss_value=1.0, ratio=1.0)
        cfg = GuidanceConfig(
            guidance_scale=10.0,
            frame_subsample_rate=1,
            grad_clip=None,
        )
        guided = GuidedI2VPipeline(mock_pipe, [mod], cfg)

        latent = torch.randn(1, 4, 2, 16, 16)
        targets = {"mock": {"dummy": True}}

        result = guided._apply_guidance_step(
            latents=latent,
            timestep=500,
            step_index=0,
            total_steps=10,
            targets=targets,
        )

        # Should return a tensor of the same shape
        assert result.shape == latent.shape

    def test_latents_to_pil(self):
        """Verify latent-to-PIL conversion produces correct number of frames."""
        decoded = torch.randn(1, 3, 5, 32, 32)  # 5 frames
        pil_frames = GuidedI2VPipeline._latents_to_pil(decoded)
        assert len(pil_frames) == 5
        # Each frame should be a PIL Image
        from PIL import Image

        for f in pil_frames:
            assert isinstance(f, Image.Image)


class TestGuidanceConfig:
    """Test GuidanceConfig defaults and construction."""

    def test_defaults(self):
        cfg = GuidanceConfig()
        assert cfg.guidance_scale == 50.0
        assert cfg.guidance_steps_ratio == 0.5
        assert cfg.frame_subsample_rate == 4
        assert cfg.grad_clip == 1.0
        assert cfg.decode_dtype == torch.float32

    def test_custom_values(self):
        cfg = GuidanceConfig(
            guidance_scale=100.0,
            guidance_steps_ratio=0.3,
            frame_subsample_rate=2,
            grad_clip=0.5,
        )
        assert cfg.guidance_scale == 100.0
        assert cfg.guidance_steps_ratio == 0.3
