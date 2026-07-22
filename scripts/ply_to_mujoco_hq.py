#!/usr/bin/env python3
"""Convert mesh PLY or SAM3D/3DGS Gaussian PLY into a MuJoCo-ready asset directory."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from lxml import etree
from plyfile import PlyData


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PLY (mesh or Gaussian splat) to MuJoCo asset directory."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input PLY path")
    parser.add_argument("--out", required=True, type=Path, help="Output asset directory")
    parser.add_argument("--model-name", default="sam3d_object", help="MuJoCo model name")
    parser.add_argument(
        "--scale-longest",
        type=float,
        default=0.10,
        help="Scale mesh so longest bbox edge equals this value (meters)",
    )
    parser.add_argument("--mass", type=float, default=0.5, help="Body mass in kg")
    parser.add_argument("--poisson-depth", type=int, default=10, help="Open3D Poisson depth")
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=80000,
        help="Target triangle count after decimation",
    )
    parser.add_argument(
        "--opacity-quantile",
        type=float,
        default=0.25,
        help="Drop points below this opacity quantile (Gaussian PLY)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=250000,
        help="Max points used for Poisson (sample by opacity if exceeded)",
    )
    parser.add_argument(
        "--collision",
        choices=("coacd", "convex_hull", "visual_decimated"),
        default="coacd",
        help="Collision mesh generation method",
    )
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.01,
        help="Remove Poisson vertices below this density quantile",
    )
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer after export")
    return parser.parse_args()


def detect_ply_type(ply_path: Path) -> str:
    ply = PlyData.read(str(ply_path))
    for element in ply.elements:
        if element.name == "face":
            return "mesh_ply"
        if element.name in ("polygon", "triangles"):
            return "mesh_ply"
    vertex_names = set(ply["vertex"].data.dtype.names or ())
    if not {"x", "y", "z"}.issubset(vertex_names):
        raise ValueError(f"PLY has no x/y/z vertex fields: {ply_path}")
    gaussian_hints = {"opacity", "scale_0", "f_dc_0", "rot_0"}
    if gaussian_hints.intersection(vertex_names):
        return "pointcloud_or_gaussian_ply"
    return "pointcloud_or_gaussian_ply"


def _load_opacity(vertex_data: np.ndarray, warnings: list[str]) -> np.ndarray | None:
    names = vertex_data.dtype.names or ()
    if "opacity" not in names:
        return None
    opacity = np.asarray(vertex_data["opacity"], dtype=np.float64)
    if opacity.size == 0:
        return None
    if opacity.min() < 0.0 or opacity.max() > 1.0:
        opacity = sigmoid(opacity)
        warnings.append("Applied sigmoid to opacity values outside [0, 1].")
    return np.clip(opacity, 0.0, 1.0)


def load_mesh_ply(ply_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(ply_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from PLY: {ply_path}")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh PLY has no vertices: {ply_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh PLY has no faces: {ply_path}")
    return mesh


def load_pointcloud_from_ply(
    ply_path: Path,
    opacity_quantile: float,
    max_points: int,
    warnings: list[str],
) -> tuple[np.ndarray, int]:
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {ply_path}")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex element missing x/y/z: {ply_path}")

    points = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ],
        axis=1,
    )
    if points.shape[0] < 100:
        raise ValueError(
            f"Too few points ({points.shape[0]}) for Poisson reconstruction; need >= 100."
        )

    opacity = _load_opacity(vertex, warnings)
    if opacity is not None:
        threshold = np.quantile(opacity, opacity_quantile)
        keep = opacity >= threshold
        points = points[keep]
        opacity = opacity[keep]
        warnings.append(
            f"Filtered points by opacity quantile {opacity_quantile:.3f} "
            f"(threshold={threshold:.4f}, kept={points.shape[0]})."
        )

    if points.shape[0] > max_points:
        if opacity is not None:
            weights = opacity / (opacity.sum() + 1e-12)
            idx = np.random.default_rng(0).choice(
                points.shape[0], size=max_points, replace=False, p=weights
            )
        else:
            idx = np.random.default_rng(0).choice(
                points.shape[0], size=max_points, replace=False
            )
        points = points[idx]
        warnings.append(f"Subsampled to max_points={max_points}.")

    return points, int(points.shape[0])


def poisson_mesh_from_points(
    points: np.ndarray,
    poisson_depth: int,
    target_triangles: int,
    density_quantile: float,
    warnings: list[str],
) -> trimesh.Trimesh:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    normal_radius = max(bbox_diag * 0.02, 1e-4)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    if len(mesh_o3d.vertices) == 0 or len(mesh_o3d.triangles) == 0:
        raise RuntimeError("Open3D Poisson reconstruction produced an empty mesh.")

    densities = np.asarray(densities)
    if densities.size > 0:
        density_threshold = np.quantile(densities, density_quantile)
        vertices_to_remove = densities < density_threshold
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()

    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_o3d = mesh_o3d.crop(bbox.scale(1.05, bbox.get_center()))

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()

    triangle_clusters, _, _ = mesh_o3d.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    if triangle_clusters.size > 0:
        counts = np.bincount(triangle_clusters)
        largest = int(np.argmax(counts))
        mesh_o3d.remove_triangles_by_mask(triangle_clusters != largest)
        mesh_o3d.remove_unreferenced_vertices()

    if len(mesh_o3d.triangles) == 0:
        raise RuntimeError("Poisson mesh became empty after cleanup.")

    if len(mesh_o3d.triangles) > target_triangles:
        try:
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(
                target_number_of_triangles=int(target_triangles)
            )
        except TypeError:
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(int(target_triangles))
        warnings.append(
            f"Simplified Poisson mesh to ~{target_triangles} triangles "
            f"(final={len(mesh_o3d.triangles)})."
        )

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()

    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=True,
    )
    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError("Failed to convert Open3D mesh to trimesh.")
    return mesh


def bbox_info(mesh: trimesh.Trimesh) -> dict[str, Any]:
    bounds = mesh.bounds.astype(float)
    extents = (bounds[1] - bounds[0]).tolist()
    center = mesh.centroid.astype(float).tolist()
    return {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist(),
        "extents": extents,
        "center": center,
        "longest_edge": float(np.max(bounds[1] - bounds[0])),
    }


def normalize_mesh(mesh: trimesh.Trimesh, scale_longest: float) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    original = bbox_info(mesh)
    longest = original["longest_edge"]
    if longest <= 0:
        raise ValueError("Mesh bbox longest edge is zero; cannot scale.")
    scale = scale_longest / longest
    centered = mesh.copy()
    centered.apply_translation(-centered.centroid)
    centered.apply_scale(scale)
    meta = {
        "original_bbox": original,
        "scale_factor": float(scale),
        "final_bbox": bbox_info(centered),
    }
    return centered, meta


def estimate_inertial(mesh: trimesh.Trimesh, mass: float, warnings: list[str]) -> dict[str, Any]:
    try:
        if not mesh.is_watertight:
            raise ValueError("mesh is not watertight")
        volume = float(mesh.volume)
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Invalid mesh volume")
        density = mass / volume
        inertia = np.asarray(mesh.moment_inertia, dtype=np.float64) * density
        com = np.asarray(mesh.center_mass, dtype=np.float64)
        return {
            "method": "mesh_mass_properties",
            "mass": float(mass),
            "center_of_mass": com.tolist(),
            "inertia": inertia.tolist(),
            "volume": volume,
        }
    except Exception as exc:
        warnings.append(f"mesh mass properties failed ({exc}); using bbox box inertia.")
        extents = mesh.bounds[1] - mesh.bounds[0]
        hx, hy, hz = extents / 2.0
        ixx = mass / 12.0 * (hy**2 + hz**2)
        iyy = mass / 12.0 * (hx**2 + hz**2)
        izz = mass / 12.0 * (hx**2 + hy**2)
        return {
            "method": "bbox_box_inertia",
            "mass": float(mass),
            "center_of_mass": [0.0, 0.0, 0.0],
            "inertia": [float(ixx), float(iyy), float(izz)],
            "volume": None,
        }


def build_collision_meshes(
    visual_mesh: trimesh.Trimesh,
    method: str,
    warnings: list[str],
) -> list[trimesh.Trimesh]:
    if method == "visual_decimated":
        return [visual_mesh.copy()]

    if method == "convex_hull":
        hull = visual_mesh.convex_hull
        if hull is None or hull.faces is None or len(hull.faces) == 0:
            raise RuntimeError("Failed to build convex hull collision mesh.")
        return [hull]

    # coacd with fallback
    try:
        import coacd

        coacd_mesh = coacd.Mesh(
            visual_mesh.vertices.astype(np.float64),
            visual_mesh.faces.astype(np.int32),
        )
        parts = coacd.run_coacd(coacd_mesh)
        if not parts:
            raise RuntimeError("coacd returned no parts")
        meshes = []
        for verts, faces in parts:
            part = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            if part.faces is not None and len(part.faces) > 0:
                meshes.append(part)
        if not meshes:
            raise RuntimeError("coacd produced only empty parts")
        warnings.append(f"coacd produced {len(meshes)} collision convex parts.")
        return meshes
    except Exception as exc:
        warnings.append(f"coacd failed ({exc}); falling back to convex hull.")
        hull = visual_mesh.convex_hull
        if hull is None or hull.faces is None or len(hull.faces) == 0:
            raise RuntimeError("coacd and convex hull fallback both failed.")
        return [hull]


def save_mesh_pair(mesh: trimesh.Trimesh, obj_path: Path, stl_path: Path) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(obj_path))
    mesh.export(str(stl_path))


def xml_value(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return " ".join(xml_value(x) for x in v)
    if isinstance(v, np.ndarray):
        return " ".join(xml_value(x) for x in v.tolist())
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return str(v)


def xml_attrs(**kwargs):
    return {k: xml_value(v) for k, v in kwargs.items() if v is not None}


def write_model_xml(
    out_dir: Path,
    model_name: str,
    inertial: dict[str, Any],
    num_collision_parts: int,
) -> Path:
    root = etree.Element("mujoco", **xml_attrs(model=model_name))
    etree.SubElement(
        root,
        "compiler",
        **xml_attrs(
            angle="degree",
            coordinate="local",
            inertiafromgeom="false",
            meshdir="meshes",
        ),
    )
    etree.SubElement(
        root,
        "option",
        **xml_attrs(gravity=[0, 0, -9.81], integrator="implicitfast"),
    )
    default = etree.SubElement(root, "default")
    etree.SubElement(
        default,
        "geom",
        **xml_attrs(friction=[0.8, 0.1, 0.1], condim=3),
    )

    asset = etree.SubElement(root, "asset")
    etree.SubElement(
        asset,
        "mesh",
        **xml_attrs(
            name="visual_mesh",
            file="visual.obj",
            scale=[1, 1, 1],
        ),
    )
    for i in range(num_collision_parts):
        if num_collision_parts == 1:
            mesh_name = "collision_mesh"
            mesh_file = "collision.obj"
        else:
            mesh_name = f"collision_{i:03d}"
            mesh_file = f"collision_{i:03d}.obj"
        etree.SubElement(
            asset,
            "mesh",
            **xml_attrs(
                name=mesh_name,
                file=mesh_file,
                scale=[1, 1, 1],
            ),
        )
    etree.SubElement(
        asset,
        "texture",
        **xml_attrs(
            name="grid",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
        ),
    )
    etree.SubElement(
        asset,
        "material",
        **xml_attrs(
            name="grid_mat",
            texture="grid",
            texrepeat=[8, 8],
            reflectance=0.1,
        ),
    )

    worldbody = etree.SubElement(root, "worldbody")
    etree.SubElement(
        worldbody,
        "geom",
        **xml_attrs(
            name="floor",
            type="plane",
            size=[2, 2, 0.05],
            pos=[0, 0, 0],
            material="grid_mat",
        ),
    )
    etree.SubElement(
        worldbody,
        "light",
        **xml_attrs(
            name="main_light",
            directional=True,
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.2, 0.2, 0.2],
            pos=[0, 0, 2],
            dir=[0, 0, -1],
        ),
    )
    etree.SubElement(
        worldbody,
        "camera",
        **xml_attrs(
            name="track",
            mode="trackcom",
            pos=[0.6, -0.6, 0.35],
            xyaxes=[0.7, 0.7, 0, -0.4, 0.4, 0.8],
        ),
    )

    body = etree.SubElement(
        worldbody,
        "body",
        **xml_attrs(name=f"{model_name}_body", pos=[0, 0, 0]),
    )
    etree.SubElement(body, "freejoint", **xml_attrs(name="root"))
    inertia = inertial["inertia"]
    etree.SubElement(
        body,
        "inertial",
        **xml_attrs(
            pos=[0, 0, 0],
            mass=inertial["mass"],
            diaginertia=inertia,
        ),
    )
    etree.SubElement(
        body,
        "geom",
        **xml_attrs(
            name="visual",
            type="mesh",
            mesh="visual_mesh",
            contype=0,
            conaffinity=0,
            rgba=[0.85, 0.85, 0.9, 1],
        ),
    )
    for i in range(num_collision_parts):
        if num_collision_parts == 1:
            mesh_name = "collision_mesh"
        else:
            mesh_name = f"collision_{i:03d}"
        etree.SubElement(
            body,
            "geom",
            **xml_attrs(
                name=mesh_name,
                type="mesh",
                mesh=mesh_name,
                rgba=[0.2, 0.6, 0.9, 0.35],
                group=3,
            ),
        )

    xml_path = out_dir / "model.xml"
    tree = etree.ElementTree(root)
    tree.write(
        str(xml_path),
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
    return xml_path


def save_preview(mesh: trimesh.Trimesh, preview_path: Path) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sample = mesh.copy()
    if len(sample.faces) > 5000:
        sample = sample.simplify_quadric_decimation(5000)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    verts = sample.vertices
    faces = sample.faces
    mesh_collection = Poly3DCollection(verts[faces], alpha=0.85)
    mesh_collection.set_facecolor((0.7, 0.75, 0.9))
    mesh_collection.set_edgecolor((0.2, 0.2, 0.2))
    mesh_collection.set_linewidth(0.05)
    ax.add_collection3d(mesh_collection)
    bounds = sample.bounds
    center = sample.centroid
    radius = np.max(bounds[1] - bounds[0]) * 0.6 + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title("Visual mesh preview")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)


def validate_mujoco(xml_path: Path, open_viewer: bool) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("WARNING: mujoco not installed; skipped model validation.")
        return

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"MuJoCo model loaded OK: nbody={model.nbody}, ngeom={model.ngeom}, nmesh={model.nmesh}")

    if not open_viewer:
        return
    try:
        mujoco.viewer.launch(model)
    except Exception as exc:
        print(f"WARNING: MuJoCo viewer failed to open ({exc}). Asset files were still generated.")


def main() -> int:
    args = parse_args()
    input_ply = args.input.resolve()
    out_dir = args.out.resolve()
    warnings: list[str] = []

    if not input_ply.is_file():
        raise FileNotFoundError(f"Input PLY does not exist: {input_ply}")

    detected_type = detect_ply_type(input_ply)
    print(f"Detected PLY type: {detected_type}")

    if detected_type == "mesh_ply":
        mesh = load_mesh_ply(input_ply)
        num_points_used = int(len(mesh.vertices))
    else:
        points, num_points_used = load_pointcloud_from_ply(
            input_ply,
            opacity_quantile=args.opacity_quantile,
            max_points=args.max_points,
            warnings=warnings,
        )
        mesh = poisson_mesh_from_points(
            points,
            poisson_depth=args.poisson_depth,
            target_triangles=args.target_triangles,
            density_quantile=args.density_quantile,
            warnings=warnings,
        )

    mesh, scale_meta = normalize_mesh(mesh, args.scale_longest)
    inertial = estimate_inertial(mesh, args.mass, warnings)
    collision_meshes = build_collision_meshes(mesh, args.collision, warnings)

    meshes_dir = out_dir / "meshes"
    preview_dir = out_dir / "preview"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    save_mesh_pair(mesh, meshes_dir / "visual.obj", meshes_dir / "visual.stl")

    if len(collision_meshes) == 1:
        save_mesh_pair(
            collision_meshes[0],
            meshes_dir / "collision.obj",
            meshes_dir / "collision.stl",
        )
    else:
        for i, cm in enumerate(collision_meshes):
            save_mesh_pair(
                cm,
                meshes_dir / f"collision_{i:03d}.obj",
                meshes_dir / f"collision_{i:03d}.stl",
            )

    xml_path = write_model_xml(
        out_dir,
        args.model_name,
        inertial,
        num_collision_parts=len(collision_meshes),
    )
    save_preview(mesh, preview_dir / "mesh_preview.png")

    metadata = {
        "input_ply": str(input_ply),
        "detected_type": detected_type,
        "num_points_used": num_points_used,
        "opacity_quantile": args.opacity_quantile,
        "poisson_depth": args.poisson_depth,
        "target_triangles": args.target_triangles,
        "scale_longest": args.scale_longest,
        "mass": args.mass,
        "inertia": inertial["inertia"],
        "center_of_mass": inertial["center_of_mass"],
        "inertial_method": inertial["method"],
        "collision_method": args.collision,
        "num_collision_parts": len(collision_meshes),
        "scale": scale_meta,
        "warnings": warnings,
        "model_xml": str(xml_path),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved visual mesh: {meshes_dir / 'visual.obj'}")
    print(f"Saved collision parts: {len(collision_meshes)}")
    print(f"Saved model.xml: {xml_path}")
    print(f"Saved metadata.json: {metadata_path}")
    print(f"Saved preview: {preview_dir / 'mesh_preview.png'}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    validate_mujoco(xml_path, args.viewer)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
