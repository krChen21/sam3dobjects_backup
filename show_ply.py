import argparse
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Show a PLY file as simple point cloud.")
    parser.add_argument("ply_path", help="Path to input PLY file")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)

    print(pcd)
    print("points:", len(pcd.points))

    if len(pcd.points) == 0:
        raise RuntimeError("No points loaded from PLY. The PLY may not be a standard point cloud.")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Simple PLY Point Cloud Viewer",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()