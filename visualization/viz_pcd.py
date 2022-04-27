import sys
import os

import ipdb
import copy
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import get_roi_mask
from utils.visualization import get_frame_from_pose

if __name__ == "__main__":

    limits = {
        "min_x": -0.5,
        "max_x": 0.3,
        "max_z": 1.3,
        "min_y": -0.5
    }

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    # frame = frame.rotate(frame.get_rotation_matrix_from_xyz((np.pi,0,np.pi/4)))
    if os.path.isfile(sys.argv[1].replace(".pcd", ".npy").replace(".ply", ".npy")):
        ee = np.load(sys.argv[1].replace(".pcd", ".npy"), allow_pickle=True)
    else:
        ee = np.zeros((7))

    ee_frame = get_frame_from_pose(frame, ee, switch_w=True)
    kinect_frame = get_frame_from_pose(frame, [0] * 7)

    pcd = o3d.io.read_point_cloud(sys.argv[1])

    points = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)

    roi_mask = get_roi_mask(points, **limits)

    points = points[roi_mask]
    rgb = rgb[roi_mask]

    print("# of points:", len(points))

    pcd_th = o3d.geometry.PointCloud()
    pcd_th.points = o3d.utility.Vector3dVector(points)
    pcd_th.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd_th, ee_frame, kinect_frame])

    # ipdb.set_trace()

    # o3d.io.write_point_cloud(
    #     sys.argv[1].replace(".pcd", "_processed.pcd"),
    #     pcd_th,
    #     print_progress=True
    # )
