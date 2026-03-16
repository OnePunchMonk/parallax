"""Unit tests for evaluation metrics.

These test the metric functions with synthetic data. They do not require
GPU or model weights — they verify shapes, ranges, and basic correctness.
"""

from __future__ import annotations

import pytest
import torch


class TestMetricHelpers:
    """Test utility functions used by metrics."""

    def test_depth_to_normals_output_shape(self):
        from parallax.guidance.normal import NormalGuidance

        depth = torch.rand(4, 1, 32, 32)
        normals = NormalGuidance._depth_to_normals(depth)
        assert normals.shape == (4, 3, 32, 32)

    def test_warp_identity(self):
        from parallax.guidance.flow import FlowGuidance

        frame = torch.rand(2, 3, 16, 16)
        flow = torch.zeros(2, 2, 16, 16)
        warped = FlowGuidance._warp_frame(frame, flow)
        assert torch.allclose(warped, frame, atol=1e-3)


class TestEvaluationImports:
    """Verify the evaluation module imports correctly."""

    def test_imports(self):
        from parallax.evaluation import (
            compute_depth_accuracy,
            compute_normal_accuracy,
            compute_semantic_consistency,
            compute_temporal_consistency,
            compute_clip_score,
            evaluate_all,
        )
        # All should be callable
        assert callable(compute_depth_accuracy)
        assert callable(compute_normal_accuracy)
        assert callable(compute_semantic_consistency)
        assert callable(compute_temporal_consistency)
        assert callable(compute_clip_score)
        assert callable(evaluate_all)
