#!/usr/bin/env python3
"""Run SAM3D-Objects inference and export Gaussian splat + mesh + GLB assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebook"))

# Match notebook/inference.py environment setup.
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import sam3d_objects  # noqa: F401,E402
from inference import (  # noqa: E402
    BLACKLIST_FILTERS,
    WHITELIST_FILTERS,
    check_hydra_safety,
    load_image,
    load_mask,
)
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils  # noqa: E402


TEXTURE_BAKING_MODULES = ("nvdiffrast", "xatlas", "pyvista", "igraph", "cv2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3D-Objects and export splat.ply, mesh, and tai.glb."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input RGB/RGBA image")
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional mask image (if omitted, uses alpha channel when present)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/hf/pipeline.yaml",
        help="Pipeline config YAML",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs/tai_mesh_export",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Compile models (slower startup)")
    return parser.parse_args()


def resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute():
        return config_path
    return (PROJECT_ROOT / config_path).resolve()


def load_inputs(image_path: Path, mask_path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
    image = load_image(str(image_path))
    if mask_path is not None:
        mask = load_mask(str(mask_path))
        return image, mask
    if image.ndim == 3 and image.shape[-1] == 4:
        mask = image[..., 3] > 0
        image = image[..., :3]
        return image, mask
    raise ValueError(
        "No --mask provided and image has no alpha channel. "
        "Pass --mask or use an RGBA image."
    )


def create_pipeline(config_path: Path, compile_model: bool = False):
    config = OmegaConf.load(str(config_path))
    config.rendering_engine = "pytorch3d"
    config.compile_model = compile_model
    config.workspace_dir = str(config_path.parent)
    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    return instantiate(config)


def missing_texture_baking_dependencies(rendering_engine: str) -> list[str]:
    missing: list[str] = []
    if rendering_engine == "nvdiffrast":
        try:
            import nvdiffrast.torch  # noqa: F401
        except ImportError:
            missing.append("nvdiffrast")
    for module_name in TEXTURE_BAKING_MODULES:
        if module_name == "nvdiffrast":
            continue
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def is_texture_baking_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "nvdiffrast",
        "texture baking",
        "xatlas",
        "pyvista",
        "igraph",
        "cv2",
        "modulenotfounderror",
        "importerror",
    )
    return isinstance(exc, (ImportError, ModuleNotFoundError)) or any(
        marker in message for marker in markers
    )


def run_inference(
    pipeline,
    image: np.ndarray,
    mask: np.ndarray | None,
    seed: int,
    with_mesh_postprocess: bool,
    with_texture_baking: bool,
    use_vertex_color: bool,
) -> dict:
    # Scheme A: mirror notebook/inference.py but enable mesh + texture export flags.
    return pipeline.run(
        image,
        mask,
        seed,
        stage1_only=False,
        with_mesh_postprocess=with_mesh_postprocess,
        with_texture_baking=with_texture_baking,
        with_layout_postprocess=False,
        use_vertex_color=use_vertex_color,
        stage1_inference_steps=None,
        pointmap=None,
        decode_formats=["gaussian", "mesh"],
    )


def retry_postprocess_without_texture(
    pipeline,
    output: dict,
    warnings: list[str],
) -> dict:
    if "gaussian" not in output or "mesh" not in output:
        warnings.append("Cannot retry untextured export: missing gaussian or mesh in output.")
        output["glb"] = None
        return output

    print("texture baking dependency missing")
    warnings.append("texture baking dependency missing")
    try:
        output["glb"] = postprocessing_utils.to_glb(
            output["gaussian"][0],
            output["mesh"][0],
            simplify=0.95,
            texture_size=1024,
            verbose=False,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            use_vertex_color=True,
            rendering_engine=pipeline.rendering_engine,
        )
    except Exception as exc:
        warnings.append(f"Untextured GLB export failed: {exc}")
        output["glb"] = None
    return output


def _vertex_colors_from_attrs(vertex_attrs) -> np.ndarray | None:
    if vertex_attrs is None:
        return None
    if hasattr(vertex_attrs, "detach"):
        colors = vertex_attrs[:, :3].detach().float().cpu().numpy()
    else:
        colors = np.asarray(vertex_attrs)[:, :3]
    if colors.size == 0:
        return None
    if colors.max() <= 1.0:
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.clip(colors, 0.0, 255.0) / 255.0
    rgba = np.zeros((colors.shape[0], 4), dtype=np.uint8)
    rgba[:, :3] = (colors * 255.0).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def export_raw_mesh(mesh_result, out_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "mesh_exported": False,
        "mesh_vertex_count": 0,
        "mesh_face_count": 0,
        "has_vertex_color": False,
    }
    vertices = mesh_result.vertices.detach().cpu().numpy()
    faces = mesh_result.faces.detach().cpu().numpy()
    info["mesh_vertex_count"] = int(vertices.shape[0])
    info["mesh_face_count"] = int(faces.shape[0])

    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return info

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vertex_colors = _vertex_colors_from_attrs(mesh_result.vertex_attrs)
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors
        info["has_vertex_color"] = True

    obj_path = out_dir / "tai_visual.obj"
    ply_path = out_dir / "tai_visual.ply"
    stl_path = out_dir / "tai_visual.stl"
    mesh.export(str(obj_path))
    mesh.export(str(ply_path))
    try:
        mesh.export(str(stl_path))
    except Exception:
        stl_path.unlink(missing_ok=True)

    info["mesh_exported"] = True
    info["mesh_paths"] = {
        "obj": str(obj_path),
        "ply": str(ply_path),
        "stl": str(stl_path) if stl_path.exists() else None,
    }
    return info


def export_glb(glb_obj, out_dir: Path) -> tuple[bool, bool, bool]:
    if glb_obj is None:
        return False, False, False
    glb_path = out_dir / "tai.glb"
    glb_obj.export(str(glb_path))

    has_texture = False
    has_vertex_color = False
    if isinstance(glb_obj, trimesh.Scene):
        geometries = list(glb_obj.geometry.values())
    else:
        geometries = [glb_obj]

    for geom in geometries:
        visual = getattr(geom, "visual", None)
        if visual is None:
            continue
        material = getattr(visual, "material", None)
        if material is not None and getattr(material, "baseColorTexture", None) is not None:
            has_texture = True
        if getattr(visual, "vertex_colors", None) is not None:
            has_vertex_color = True
    return True, has_texture, has_vertex_color

def main() -> int:
    args = parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_config_path(args.config)
    image_path = args.image.resolve()
    mask_path = args.mask.resolve() if args.mask is not None else None

    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "image": str(image_path),
        "mask": str(mask_path) if mask_path is not None else None,
        "config": str(config_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "gs_exported": False,
        "mesh_exported": False,
        "glb_exported": False,
        "mesh_vertex_count": 0,
        "mesh_face_count": 0,
        "has_texture": False,
        "has_vertex_color": False,
        "texture_baking_attempted": False,
        "texture_baking_succeeded": False,
        "warnings": warnings,
    }

    image, mask = load_inputs(image_path, mask_path)
    pipeline = create_pipeline(config_path, compile_model=args.compile)
    rendering_engine = getattr(pipeline, "rendering_engine", "pytorch3d")

    missing_deps = missing_texture_baking_dependencies(rendering_engine)
    with_texture_baking = True
    use_vertex_color = False
    if missing_deps:
        print("texture baking dependency missing")
        warnings.append(
            "texture baking dependency missing: "
            + ", ".join(sorted(set(missing_deps)))
        )
        with_texture_baking = False
        use_vertex_color = True

    metadata["texture_baking_attempted"] = with_texture_baking

    try:
        output = run_inference(
            pipeline,
            image,
            mask,
            args.seed,
            with_mesh_postprocess=True,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
        )
        if with_texture_baking:
            metadata["texture_baking_succeeded"] = output.get("glb") is not None
    except Exception as exc:
        if with_texture_baking and is_texture_baking_error(exc):
            warnings.append(f"texture baking failed during inference: {exc}")
            warnings.append(traceback.format_exc())
            print("texture baking dependency missing")
            warnings.append("texture baking dependency missing")
            output = run_inference(
                pipeline,
                image,
                mask,
                args.seed,
                with_mesh_postprocess=True,
                with_texture_baking=False,
                use_vertex_color=True,
            )
        else:
            raise

    if with_texture_baking and output.get("glb") is None and "mesh" in output and "gaussian" in output:
        output = retry_postprocess_without_texture(pipeline, output, warnings)

    output_keys = sorted(output.keys())
    metadata["output_keys"] = output_keys
    print("output keys:", output_keys)

    if "gs" in output and output["gs"] is not None:
        splat_path = out_dir / "splat.ply"
        output["gs"].save_ply(str(splat_path))
        metadata["gs_exported"] = True
        metadata["splat_ply"] = str(splat_path)

    if "mesh" in output and output["mesh"]:
        mesh_info = export_raw_mesh(output["mesh"][0], out_dir)
        metadata.update(
            {
                "mesh_exported": mesh_info["mesh_exported"],
                "mesh_vertex_count": mesh_info["mesh_vertex_count"],
                "mesh_face_count": mesh_info["mesh_face_count"],
                "has_vertex_color": mesh_info.get("has_vertex_color", False),
            }
        )
        if "mesh_paths" in mesh_info:
            metadata["mesh_paths"] = mesh_info["mesh_paths"]

    if "glb" in output:
        glb_exported, has_texture, has_vertex_color_glb = export_glb(output["glb"], out_dir)
        metadata["glb_exported"] = glb_exported
        if glb_exported:
            metadata["glb_path"] = str(out_dir / "tai.glb")
            metadata["has_texture"] = has_texture
            if has_vertex_color_glb:
                metadata["has_vertex_color"] = True

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
