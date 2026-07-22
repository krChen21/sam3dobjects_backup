# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
import sys

sys.path.append("notebook")

from inference import interactive_visualizer, ready_gaussian_for_video_rendering, render_video
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

# Must match the Gaussian init params used during inference (see slat_decoder_gs.yaml).
DEFAULT_GAUSSIAN_PARAMS = {
    "sh_degree": 0,
    "aabb": [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
    "mininum_kernel_size": 0.0009,
    "scaling_bias": 0.004,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
}


def load_splat_ply(ply_path: str) -> Gaussian:
    """Load a Gaussian splat saved by demo.py (output['gs'].save_ply(...))."""
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    gs = Gaussian(**DEFAULT_GAUSSIAN_PARAMS)
    gs.load_ply(ply_path)
    return gs


def main():
    parser = argparse.ArgumentParser(
        description="Load a previously exported splat.ply from SAM 3D Objects."
    )
    parser.add_argument(
        "ply_path",
        nargs="?",
        default="splat.ply",
        help="Path to the exported Gaussian splat PLY file (default: splat.ply)",
    )
    parser.add_argument(
        "--render-gif",
        metavar="PATH",
        help="Render a turntable preview and save as GIF",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the Gradio interactive 3D viewer",
    )
    args = parser.parse_args()

    gs = load_splat_ply(args.ply_path)
    print(f"Loaded {gs.get_xyz.shape[0]} Gaussians from {args.ply_path}")

    if args.render_gif:
        import imageio

        scene_gs = ready_gaussian_for_video_rendering(gs)
        video = render_video(
            scene_gs,
            r=1,
            fov=60,
            pitch_deg=15,
            yaw_start_deg=-45,
            resolution=512,
        )["color"]
        imageio.mimsave(
            args.render_gif,
            video,
            format="GIF",
            duration=1000 / 30,
            loop=0,
        )
        print(f"Saved preview GIF to {args.render_gif}")

    if args.interactive:
        interactive_visualizer(args.ply_path)


if __name__ == "__main__":
    main()
