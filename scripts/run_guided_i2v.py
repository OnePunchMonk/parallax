#!/usr/bin/env python
"""CLI script for guided Image-to-Video generation.

Usage
-----
::

    python scripts/run_guided_i2v.py \\
        --image input.png \\
        --prompt "A cat walking across a table" \\
        --guidance depth \\
        --target-depth target_depth.png \\
        --guidance-scale 50.0 \\
        --num-frames 33 \\
        --output output.mp4

    # Semantic guidance:
    python scripts/run_guided_i2v.py \\
        --image input.png \\
        --prompt "A street scene" \\
        --guidance semantic \\
        --reference-image reference.png \\
        --output output.mp4

    # Multi-guidance:
    python scripts/run_guided_i2v.py \\
        --image input.png \\
        --prompt "A room interior" \\
        --guidance depth+semantic \\
        --target-depth depth.png \\
        --reference-image ref.png \\
        --output output.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot guided I2V generation via test-time guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- I/O ---
    p.add_argument("--image", type=str, required=True, help="Input conditioning image")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt")
    p.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    p.add_argument("--output", type=str, default="output.mp4", help="Output video path")

    # --- Guidance ---
    p.add_argument(
        "--guidance",
        type=str,
        default="depth",
        help="Guidance type: depth, semantic, segmentation, or depth+semantic etc.",
    )
    p.add_argument("--target-depth", type=str, default=None, help="Target depth map image")
    p.add_argument("--reference-image", type=str, default=None, help="Reference image for semantic guidance")
    p.add_argument("--target-mask", type=str, default=None, help="Target segmentation mask")
    p.add_argument("--guidance-scale", type=float, default=50.0, help="Global guidance scale α")
    p.add_argument("--guidance-steps-ratio", type=float, default=0.5, help="Fraction of steps to guide")
    p.add_argument("--frame-subsample-rate", type=int, default=4, help="Decode every N-th frame for guidance")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")

    # --- Model ---
    p.add_argument(
        "--model-id",
        type=str,
        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        help="Base I2V model HuggingFace ID",
    )
    p.add_argument("--depth-model-id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--dino-model-id", type=str, default="facebook/dinov2-small")
    p.add_argument("--sam-model-id", type=str, default="facebook/sam2-hiera-small")

    # --- Generation ---
    p.add_argument("--num-frames", type=int, default=33, help="Number of video frames")
    p.add_argument("--num-steps", type=int, default=30, help="Denoising steps")
    p.add_argument("--cfg-scale", type=float, default=5.0, help="CFG guidance scale")
    p.add_argument("--height", type=int, default=None, help="Output height")
    p.add_argument("--width", type=int, default=None, help="Output width")
    p.add_argument("--fps", type=int, default=16, help="Output FPS")

    # --- System ---
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Also generate an unguided version and create a comparison video",
    )

    return p.parse_args()


def build_guidance_modules(args: argparse.Namespace):
    """Create guidance module instances based on CLI args."""
    from parallax.guidance import DepthGuidance, SemanticGuidance, SegmentationGuidance, CompositeGuidance

    guidance_types = [g.strip() for g in args.guidance.split("+")]
    modules = []

    for gtype in guidance_types:
        if gtype == "depth":
            modules.append(DepthGuidance(
                model_id=args.depth_model_id,
                guidance_scale=args.guidance_scale,
                guidance_ratio=args.guidance_steps_ratio,
            ))
        elif gtype == "semantic":
            modules.append(SemanticGuidance(
                model_id=args.dino_model_id,
                guidance_scale=args.guidance_scale,
                guidance_ratio=args.guidance_steps_ratio,
            ))
        elif gtype == "segmentation":
            modules.append(SegmentationGuidance(
                model_id=args.sam_model_id,
                guidance_scale=args.guidance_scale,
                guidance_ratio=args.guidance_steps_ratio,
            ))
        else:
            raise ValueError(f"Unknown guidance type: {gtype}")

    return modules


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("parallax.cli")

    # --- Device & dtype ---
    device = torch.device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info("Seed: %d", args.seed)

    # --- Load base pipeline ---
    logger.info("Loading base I2V pipeline: %s", args.model_id)
    from diffusers import AutoModel, WanImageToVideoPipeline, AutoencoderKLWan
    from transformers import CLIPVisionModel

    image_encoder = CLIPVisionModel.from_pretrained(
        args.model_id, subfolder="image_encoder", torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.to(device)
    logger.info("Base pipeline loaded ✓")

    # --- Build guidance ---
    from parallax.pipeline import GuidedI2VPipeline, GuidanceConfig

    guidance_modules = build_guidance_modules(args)
    config = GuidanceConfig(
        guidance_scale=args.guidance_scale,
        guidance_steps_ratio=args.guidance_steps_ratio,
        frame_subsample_rate=args.frame_subsample_rate,
        grad_clip=args.grad_clip,
    )

    guided_pipeline = GuidedI2VPipeline(pipe, guidance_modules, config)
    guided_pipeline.load_guidance_models(device=device, dtype=torch.float32)
    logger.info("Guidance modules loaded: %s", [m.name for m in guidance_modules])

    # --- Prepare inputs ---
    input_image = Image.open(args.image).convert("RGB")

    target_kwargs = {}
    if args.target_depth:
        target_kwargs["target_depth"] = Image.open(args.target_depth)
    if args.reference_image:
        target_kwargs["reference_image"] = Image.open(args.reference_image)
    if args.target_mask:
        target_kwargs["target_mask"] = Image.open(args.target_mask)

    # --- Generate ---
    logger.info("Generating %d-frame video with %d denoising steps...", args.num_frames, args.num_steps)
    result = guided_pipeline.generate(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        guidance_cfg_scale=args.cfg_scale,
        **target_kwargs,
    )

    frames = result["frames"]
    logger.info(
        "Generation complete. Guided steps applied: %d/%d",
        result["guided_steps"],
        args.num_steps,
    )

    # --- Export ---
    from parallax.utils.visualization import export_video

    output_path = export_video(frames, args.output, fps=args.fps)
    logger.info("Video saved to: %s", output_path)

    # --- Optional comparison ---
    if args.compare:
        from parallax.utils.visualization import export_comparison_video

        logger.info("Generating unguided baseline for comparison...")
        unguided_result = pipe(
            image=input_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
        )
        unguided_frames = unguided_result.frames[0]

        compare_path = Path(args.output).with_stem(
            Path(args.output).stem + "_comparison"
        )
        export_comparison_video(
            frames, unguided_frames, compare_path, fps=args.fps,
        )
        logger.info("Comparison video saved to: %s", compare_path)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
