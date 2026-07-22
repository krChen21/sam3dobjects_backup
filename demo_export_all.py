#!/usr/bin/env python3
"""Export all available SAM3D-Objects output formats (Gaussian splat, mesh, GLB)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Align with demo.py: run from project root so relative checkpoint paths resolve.
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Mirror notebook/inference.py env setup before importing inference/torch.
if os.environ.get("CONDA_PREFIX"):
    os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

_DINO_HUB_PATCHED = False


def install_dino_local_hub_patch() -> None:
    """Redirect torch.hub DINO loads to a local repo before Inference() is created."""
    global _DINO_HUB_PATCHED
    if _DINO_HUB_PATCHED:
        return

    import torch

    def _is_valid_dino_repo(path: str) -> bool:
        return bool(path) and os.path.isfile(os.path.join(path, "hubconf.py"))

    candidates: list[str] = []
    env_repo = os.environ.get("DINO_LOCAL_REPO")
    if env_repo:
        candidates.append(env_repo)
    candidates.append("/home/ubuntu/third_party/dinov2")
    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    candidates.append(os.path.join(torch_home, "hub", "facebookresearch_dinov2_main"))

    local_repo: str | None = None
    for candidate in candidates:
        candidate = os.path.expanduser(candidate)
        if _is_valid_dino_repo(candidate):
            local_repo = os.path.abspath(candidate)
            break

    print("[DINO PATCH] torch hub dir:", torch.hub.get_dir())
    print("[DINO PATCH] local repo:", local_repo)

    if local_repo is None:
        print("WARNING: no local DINO repo found; torch.hub may access GitHub.")
        return

    _original_torch_hub_load = torch.hub.load

    def patched_torch_hub_load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir) == "facebookresearch/dinov2":
            print(
                f"[DINO PATCH] Redirect torch.hub.load from facebookresearch/dinov2 to {local_repo}"
            )
            repo_or_dir = str(local_repo)
            kwargs["source"] = "local"
            kwargs.pop("skip_validation", None)
            kwargs.pop("trust_repo", None)
        return _original_torch_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = patched_torch_hub_load
    _DINO_HUB_PATCHED = True


install_dino_local_hub_patch()

# Match demo.py import order: notebook inference only, no early sam3d/trimesh imports.
sys.path.append("notebook")
from inference import Inference, load_image, load_mask  # noqa: E402

import argparse
import importlib.util
import inspect
import json
import traceback
from typing import Any

_GSPLAT_PATCHED = False
HAS_GSPLAT = importlib.util.find_spec("gsplat") is not None
HAS_INRIA_GS = importlib.util.find_spec("diff_gaussian_rasterization") is not None


def check_gaussian_render_deps(texture_baking: bool) -> None:
    if texture_baking and HAS_GSPLAT and not HAS_INRIA_GS:
        print("Texture baking will use gsplat backend.")


def install_gsplat_render_patch() -> None:
    """Force texture-baking Gaussian renders to use gsplat when inria is unavailable."""
    global _GSPLAT_PATCHED
    if _GSPLAT_PATCHED or not HAS_GSPLAT:
        return

    from sam3d_objects.model.backbone.tdfy_dit.renderers import gaussian_render
    from sam3d_objects.model.backbone.tdfy_dit.representations import Gaussian
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
    from sam3d_objects.model.backbone.tdfy_dit.utils.random_utils import (
        sphere_hammersley_sequence,
    )

    _orig_render = gaussian_render.render

    def patched_render(*args, backend="inria", **kwargs):
        if backend == "inria" and not HAS_INRIA_GS:
            print(
                "[GSPLAT PATCH] render backend changed from inria to gsplat "
                "because diff_gaussian_rasterization is unavailable"
            )
            backend = "gsplat"
        return _orig_render(*args, backend=backend, **kwargs)

    gaussian_render.render = patched_render

    _orig_gr_init = gaussian_render.GaussianRenderer.__init__

    def patched_gr_init(self, rendering_options=None):
        opts = dict(rendering_options or {})
        backend = opts.get("backend", "inria")
        if backend == "inria" and not HAS_INRIA_GS:
            opts["backend"] = "gsplat"
            print("[GSPLAT PATCH] GaussianRenderer backend forced to gsplat")
        elif "backend" not in opts:
            opts["backend"] = "gsplat"
            print("[GSPLAT PATCH] GaussianRenderer backend forced to gsplat")
        _orig_gr_init(self, opts)

    gaussian_render.GaussianRenderer.__init__ = patched_gr_init

    _orig_render_frames = render_utils.render_frames

    def patched_render_frames(
        sample,
        extrinsics,
        intrinsics,
        options=None,
        colors_overwrite=None,
        verbose=True,
        **kwargs,
    ):
        opts = dict(options or {})
        if isinstance(sample, Gaussian):
            backend = opts.get("backend", "inria")
            if backend == "inria" and not HAS_INRIA_GS:
                opts["backend"] = "gsplat"
                print(
                    "[GSPLAT PATCH] render_frames backend changed from inria to gsplat "
                    "because diff_gaussian_rasterization is unavailable"
                )
            elif "backend" not in opts:
                opts["backend"] = "gsplat"
                print("[GSPLAT PATCH] GaussianRenderer backend forced to gsplat")
        return _orig_render_frames(
            sample,
            extrinsics,
            intrinsics,
            opts,
            colors_overwrite=colors_overwrite,
            verbose=verbose,
            **kwargs,
        )

    render_utils.render_frames = patched_render_frames

    def patched_render_multiview(sample, resolution=512, nviews=30):
        r = 2
        fov = 40
        cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, r, fov
        )
        res = render_utils.render_frames(
            sample,
            extrinsics,
            intrinsics,
            {"resolution": resolution, "bg_color": (0, 0, 0), "backend": "gsplat"},
        )
        return res["color"], extrinsics, intrinsics

    render_utils.render_multiview = patched_render_multiview
    postprocessing_utils.render_multiview = patched_render_multiview
    _GSPLAT_PATCHED = True


install_gsplat_render_patch()

DEFAULT_TAG = "hf"
DEFAULT_CONFIG = f"checkpoints/{DEFAULT_TAG}/pipeline.yaml"


def resolve_config_path(config: Path | str) -> str:
    """Use the same relative config string style as demo.py when possible."""
    path = Path(config)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def create_inference_like_demo(config: Path | str, compile_model: bool = False) -> Inference:
    """Load models exactly like demo.py: Inference(relative_config, compile=False)."""
    return Inference(resolve_config_path(config), compile=compile_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3D-Objects (demo.py path) and export all available formats."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image path")
    parser.add_argument("--mask", type=Path, required=True, help="Input mask image path")
    parser.add_argument("--out", type=Path, default=Path("outputs/tai_export_all"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline config YAML")
    parser.add_argument("--compile", action="store_true", help="Compile models (same as demo.py compile flag)")

    parser.add_argument("--export-gs", action="store_true", help="Export Gaussian splat.ply")
    parser.add_argument("--export-mesh", action="store_true", help="Export raw mesh-related files")
    parser.add_argument("--export-glb", action="store_true", help="Export object.glb")
    parser.add_argument("--export-obj", action="store_true", help="Export mesh.obj")
    parser.add_argument("--export-stl", action="store_true", help="Export mesh.stl")
    parser.add_argument("--export-ply", action="store_true", help="Export triangle mesh.ply (not splat)")
    parser.add_argument("--texture-baking", action="store_true", help="Enable texture baking in pipeline")
    parser.add_argument("--mesh-postprocess", action="store_true", help="Enable mesh postprocess in pipeline")
    parser.add_argument("--print-output-keys", action="store_true", help="Print output keys and value types")
    parser.add_argument("--dry-run", action="store_true", help="Run inference and print summary without writing large files")
    return parser.parse_args()


def resolve_export_flags(args: argparse.Namespace) -> dict[str, bool]:
    explicit = any(
        [
            args.export_gs,
            args.export_mesh,
            args.export_glb,
            args.export_obj,
            args.export_stl,
            args.export_ply,
        ]
    )
    if explicit:
        return {
            "gs": args.export_gs,
            "mesh": args.export_mesh,
            "glb": args.export_glb,
            "obj": args.export_obj,
            "stl": args.export_stl,
            "ply": args.export_ply,
        }
    return {"gs": True, "mesh": True, "glb": True, "obj": True, "stl": True, "ply": True}


def is_texture_baking_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "nvdiffrast",
        "texture",
        "xatlas",
        "pyvista",
        "igraph",
        "cv2",
        "modulenotfounderror",
        "importerror",
        "gaussianrasterizationsettings",
        "diff_gaussian_rasterization",
    )
    if isinstance(exc, NameError) and "gaussianrasterization" in message:
        return True
    return isinstance(exc, (ImportError, ModuleNotFoundError)) or any(m in message for m in markers)


def _run_pipeline_with_texture_fallback(
    pipeline,
    rgba_image,
    seed: int,
    mesh_postprocess: bool,
    texture_baking: bool,
    warnings: list[str],
) -> dict:
    try:
        return pipeline.run(
            rgba_image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=mesh_postprocess,
            with_texture_baking=texture_baking,
            with_layout_postprocess=False,
            use_vertex_color=not texture_baking,
            decode_formats=["gaussian", "mesh"],
        )
    except Exception as exc:
        if texture_baking and is_texture_baking_error(exc):
            warnings.append(f"texture baking failed: {exc}")
            warnings.append(traceback.format_exc())
            if isinstance(exc, NameError):
                print(
                    "texture baking failed (GaussianRasterizationSettings); "
                    "retrying without texture baking"
                )
            else:
                print("texture baking dependency missing")
            return pipeline.run(
                rgba_image,
                None,
                seed,
                stage1_only=False,
                with_mesh_postprocess=mesh_postprocess,
                with_texture_baking=False,
                with_layout_postprocess=False,
                use_vertex_color=True,
                decode_formats=["gaussian", "mesh"],
            )
        raise


def run_inference(
    inference: Inference,
    image,
    mask,
    seed: int,
    mesh_postprocess: bool,
    texture_baking: bool,
    warnings: list[str],
) -> dict:
    """Reuse demo.py Inference object; call pipeline.run when wrapper lacks mesh flags."""
    call_sig = inspect.signature(inference.__call__)
    if "with_mesh_postprocess" in call_sig.parameters:
        kwargs = {
            "seed": seed,
            "with_mesh_postprocess": mesh_postprocess,
            "with_texture_baking": texture_baking,
        }
        if "decode_formats" in call_sig.parameters:
            kwargs["decode_formats"] = ["gaussian", "mesh"]
        return inference(image, mask, **kwargs)

    rgba_image = inference.merge_mask_to_rgba(image, mask)
    pipeline = inference._pipeline
    return _run_pipeline_with_texture_fallback(
        pipeline,
        rgba_image,
        seed,
        mesh_postprocess,
        texture_baking,
        warnings,
    )


def _to_numpy_xyz(arr):
    import numpy as np

    if hasattr(arr, "detach"):
        return arr.detach().float().cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def _vertex_colors_from_attrs(vertex_attrs):
    import numpy as np

    if vertex_attrs is None:
        return None
    colors = _to_numpy_xyz(vertex_attrs)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None
    colors = colors[:, :3]
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


def extract_trimesh_from_output_mesh(mesh_obj):
    import numpy as np
    import trimesh

    warnings: list[str] = []

    if mesh_obj is None:
        warnings.append("mesh object is None")
        return None, warnings

    candidates = mesh_obj
    if isinstance(mesh_obj, (list, tuple)):
        if len(mesh_obj) == 0:
            warnings.append("mesh list is empty")
            return None, warnings
        candidates = mesh_obj[0]
        if len(mesh_obj) > 1:
            warnings.append(f"mesh is a sequence with {len(mesh_obj)} items; using index 0")

    if isinstance(candidates, trimesh.Trimesh):
        return candidates, warnings

    if isinstance(candidates, trimesh.Scene):
        meshes = [g for g in candidates.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            warnings.append("trimesh.Scene has no Trimesh geometry")
            return None, warnings
        if len(meshes) == 1:
            return meshes[0], warnings
        combined = trimesh.util.concatenate(meshes)
        warnings.append(f"merged {len(meshes)} meshes from Scene")
        return combined, warnings

    vertices = getattr(candidates, "vertices", None)
    faces = getattr(candidates, "faces", None)
    if vertices is None or faces is None:
        warnings.append(f"unsupported mesh type: {type(candidates)}")
        return None, warnings

    vertices_np = _to_numpy_xyz(vertices)
    faces_np = _to_numpy_xyz(faces).astype(np.int64)
    if vertices_np.shape[0] == 0 or faces_np.shape[0] == 0:
        warnings.append("mesh has zero vertices or faces")
        return None, warnings

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    vertex_colors = _vertex_colors_from_attrs(getattr(candidates, "vertex_attrs", None))
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors
    return mesh, warnings


def describe_output(output: dict, print_keys: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "output_type": type(output).__name__,
        "output_keys": sorted(output.keys()) if isinstance(output, dict) else [],
        "key_types": {},
        "mesh_detail": {},
        "glb_detail": {},
    }
    if not isinstance(output, dict):
        return summary

    for key, value in output.items():
        summary["key_types"][key] = type(value).__name__

    if print_keys:
        print("output type:", summary["output_type"])
        print("output keys:", summary["output_keys"])
        for key in summary["output_keys"]:
            print(f"  - {key}: {summary['key_types'][key]}")

    if "mesh" in output and output["mesh"] is not None:
        mesh_obj = output["mesh"][0] if isinstance(output["mesh"], (list, tuple)) else output["mesh"]
        vertices = getattr(mesh_obj, "vertices", None)
        faces = getattr(mesh_obj, "faces", None)
        summary["mesh_detail"] = {
            "type": type(mesh_obj).__name__,
            "vertices": int(_to_numpy_xyz(vertices).shape[0]) if vertices is not None else 0,
            "faces": int(_to_numpy_xyz(faces).shape[0]) if faces is not None else 0,
            "has_vertex_attrs": getattr(mesh_obj, "vertex_attrs", None) is not None,
        }
        if print_keys:
            print("mesh detail:", summary["mesh_detail"])

    if "glb" in output:
        glb_obj = output["glb"]
        summary["glb_detail"] = {
            "type": type(glb_obj).__name__ if glb_obj is not None else "None",
            "has_export": hasattr(glb_obj, "export"),
        }
        if glb_obj is not None and print_keys:
            import trimesh

            if isinstance(glb_obj, trimesh.Scene):
                geoms = list(glb_obj.geometry.values())
            else:
                geoms = [glb_obj]
            has_texture = False
            has_vertex_color = False
            for geom in geoms:
                visual = getattr(geom, "visual", None)
                if visual is None:
                    continue
                material = getattr(visual, "material", None)
                if material is not None and getattr(material, "baseColorTexture", None) is not None:
                    has_texture = True
                if getattr(visual, "vertex_colors", None) is not None:
                    has_vertex_color = True
            summary["glb_detail"]["has_texture"] = has_texture
            summary["glb_detail"]["has_vertex_color"] = has_vertex_color
            print("glb detail:", summary["glb_detail"])

    return summary


def mesh_info_dict(mesh, has_texture: bool = False) -> dict[str, Any]:
    if mesh is None:
        return {
            "vertices": 0,
            "faces": 0,
            "is_watertight": False,
            "bounds": None,
            "has_vertex_colors": False,
            "has_texture": has_texture,
        }
    bounds = mesh.bounds.tolist() if mesh.bounds is not None else None
    has_vertex_colors = False
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "vertex_colors", None) is not None:
        has_vertex_colors = True
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "bounds": bounds,
        "has_vertex_colors": has_vertex_colors,
        "has_texture": has_texture,
    }


def safe_export(export_fn, warnings: list[str], label: str) -> str | None:
    try:
        export_fn()
        return label
    except Exception as exc:
        msg = f"{label} export failed: {exc}"
        warnings.append(msg)
        warnings.append(traceback.format_exc())
        print(f"WARNING: {msg}")
        return None


def export_gaussian_ply(gs_obj, path: Path, warnings: list[str]) -> str | None:
    if gs_obj is None:
        warnings.append(f"Gaussian object missing for {path.name}")
        return None

    def _do():
        gs_obj.save_ply(str(path))

    return safe_export(_do, warnings, str(path))


def export_glb_object(glb_obj, path: Path, warnings: list[str]) -> tuple[str | None, bool, bool]:
    import trimesh

    if glb_obj is None:
        warnings.append("output['glb'] is None")
        return None, False, False
    if not hasattr(glb_obj, "export"):
        warnings.append(f"glb object has no export method: {type(glb_obj)}")
        return None, False, False

    def _do():
        glb_obj.export(str(path))

    exported = safe_export(_do, warnings, str(path))
    has_texture = False
    has_vertex_color = False
    geoms = list(glb_obj.geometry.values()) if isinstance(glb_obj, trimesh.Scene) else [glb_obj]
    for geom in geoms:
        visual = getattr(geom, "visual", None)
        if visual is None:
            continue
        material = getattr(visual, "material", None)
        if material is not None and getattr(material, "baseColorTexture", None) is not None:
            has_texture = True
        if getattr(visual, "vertex_colors", None) is not None:
            has_vertex_color = True
    return exported, has_texture, has_vertex_color


def main() -> int:
    args = parse_args()
    export_flags = resolve_export_flags(args)
    out_dir = args.out
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    exports: dict[str, str | None] = {
        "splat_ply": None,
        "gaussian_ply": None,
        "mesh_obj": None,
        "mesh_stl": None,
        "mesh_ply": None,
        "glb": None,
    }

    # Same model loading order as demo.py: Inference first, then image/mask.
    inference = create_inference_like_demo(args.config, compile_model=args.compile)

    image = load_image(str(args.image))
    mask = load_mask(str(args.mask))

    mesh_postprocess = args.mesh_postprocess or export_flags["mesh"] or export_flags["glb"]
    texture_baking = args.texture_baking or export_flags["glb"]

    check_gaussian_render_deps(texture_baking)

    output = run_inference(
        inference,
        image,
        mask,
        args.seed,
        mesh_postprocess=mesh_postprocess,
        texture_baking=texture_baking,
        warnings=warnings,
    )

    output_summary = describe_output(output, print_keys=args.print_output_keys or True)

    if args.dry_run:
        warnings.append("dry-run enabled: skipped writing large export files")
        metadata = {
            "image": str(args.image),
            "mask": str(args.mask),
            "seed": args.seed,
            "config": str(args.config),
            "output_keys": output_summary["output_keys"],
            "exports": exports,
            "mesh_info": mesh_info_dict(None),
            "splat_ply_type": "3d_gaussian_splat",
            "mesh_ply_type": "triangle_mesh",
            "warnings": warnings,
            "dry_run": True,
        }
        metadata_path = out_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        summary_path = logs_dir / "output_summary.txt"
        summary_path.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")
        print("dry-run complete; no large files written")
        print("output keys:", output_summary["output_keys"])
        return 0

    success_count = 0

    if export_flags["gs"]:
        if "gs" in output:
            path = export_gaussian_ply(output["gs"], out_dir / "splat.ply", warnings)
            exports["splat_ply"] = path
            if path:
                success_count += 1
        else:
            warnings.append("output has no 'gs' key")

        if "gaussian" in output:
            gaussian_obj = output["gaussian"][0] if isinstance(output["gaussian"], (list, tuple)) else output["gaussian"]
            path = export_gaussian_ply(gaussian_obj, out_dir / "gaussian.ply", warnings)
            exports["gaussian_ply"] = path
            if path:
                success_count += 1

    raw_mesh, mesh_warnings = extract_trimesh_from_output_mesh(output.get("mesh"))
    warnings.extend(mesh_warnings)

    glb_has_texture = False
    if export_flags["glb"] and "glb" in output:
        path, glb_has_texture, _ = export_glb_object(output["glb"], out_dir / "object.glb", warnings)
        exports["glb"] = path
        if path:
            success_count += 1

    if raw_mesh is None and export_flags["glb"] and exports["glb"] is None:
        if "gaussian" in output and "mesh" in output:
            try:
                from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils

                fallback_glb = postprocessing_utils.to_glb(
                    output["gaussian"][0],
                    output["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                    verbose=False,
                    with_mesh_postprocess=mesh_postprocess,
                    with_texture_baking=False,
                    use_vertex_color=True,
                    rendering_engine=getattr(inference._pipeline, "rendering_engine", "pytorch3d"),
                )
                path, glb_has_texture, _ = export_glb_object(
                    fallback_glb, out_dir / "object.glb", warnings
                )
                exports["glb"] = path
                if path:
                    success_count += 1
                    warnings.append("exported untextured glb via fallback to_glb")
            except Exception as exc:
                warnings.append(f"fallback glb export failed: {exc}")

    mesh_for_formats = raw_mesh
    if mesh_for_formats is None and output.get("glb") is not None:
        import trimesh

        glb_obj = output["glb"]
        if isinstance(glb_obj, trimesh.Trimesh):
            mesh_for_formats = glb_obj
        elif isinstance(glb_obj, trimesh.Scene):
            mesh_for_formats, extra = extract_trimesh_from_output_mesh(glb_obj)
            warnings.extend(extra)

    if mesh_for_formats is not None:
        if export_flags["obj"] or export_flags["mesh"]:
            def _obj():
                mesh_for_formats.export(str(out_dir / "mesh.obj"))
            path = safe_export(_obj, warnings, str(out_dir / "mesh.obj"))
            exports["mesh_obj"] = path
            if path:
                success_count += 1

        if export_flags["stl"] or export_flags["mesh"]:
            def _stl():
                mesh_for_formats.export(str(out_dir / "mesh.stl"))
            path = safe_export(_stl, warnings, str(out_dir / "mesh.stl"))
            exports["mesh_stl"] = path
            if path:
                success_count += 1

        if export_flags["ply"] or export_flags["mesh"]:
            def _ply():
                mesh_for_formats.export(str(out_dir / "mesh.ply"))
            path = safe_export(_ply, warnings, str(out_dir / "mesh.ply"))
            exports["mesh_ply"] = path
            if path:
                success_count += 1
    else:
        if export_flags["obj"] or export_flags["stl"] or export_flags["ply"] or export_flags["mesh"]:
            warnings.append(
                "mesh formats skipped: output has no usable mesh and glb could not be converted"
            )

    metadata = {
        "image": str(args.image),
        "mask": str(args.mask),
        "seed": args.seed,
        "config": str(args.config),
        "output_keys": output_summary["output_keys"],
        "exports": exports,
        "mesh_info": mesh_info_dict(mesh_for_formats, has_texture=glb_has_texture),
        "splat_ply_type": "3d_gaussian_splat",
        "mesh_ply_type": "triangle_mesh",
        "mesh_postprocess": mesh_postprocess,
        "texture_baking": texture_baking,
        "warnings": warnings,
    }
    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "SAM3D export summary",
        f"output keys: {output_summary['output_keys']}",
        f"mesh vertices: {metadata['mesh_info']['vertices']}",
        f"mesh faces: {metadata['mesh_info']['faces']}",
        f"splat.ply exported: {exports['splat_ply'] is not None}",
        f"glb exported: {exports['glb'] is not None}",
        "successful exports:",
    ]
    for key, value in exports.items():
        if value:
            summary_lines.append(f"  - {key}: {value}")
    if warnings:
        summary_lines.append("warnings:")
        summary_lines.extend(f"  - {w}" for w in warnings)

    summary_text = "\n".join(summary_lines)
    (logs_dir / "output_summary.txt").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"metadata: {metadata_path}")

    if success_count == 0:
        print("ERROR: no formats exported successfully")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
