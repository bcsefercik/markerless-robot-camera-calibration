import sys
import os

import ipdb
import copy
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import get_roi_mask
from utils.visualization import get_frame_from_pose, create_coordinate_frame
from utils.transformation import switch_w, transform_pose2pose

if __name__ == "__main__":

    limits = {
        # "min_x": -0.7,
        # "max_x": 0.7,
        # # "max_z": 1.5
    }

    base_pose = None

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    # frame = frame.rotate(frame.get_rotation_matrix_from_xyz((np.pi,0,np.pi/4)))
    if os.path.isfile(sys.argv[1].replace(".pcd", ".npy").replace(".ply", ".npy")):
        ee = np.load(sys.argv[1].replace(".pcd", ".npy"), allow_pickle=True)
    else:
        ee = np.zeros((7))

    if os.path.isfile(sys.argv[1].replace(".pcd", "_robot2ee_pose.npy").replace(".ply", "_robot2ee_pose.npy")):
        ee2base_pose = np.load(sys.argv[1].replace(".pcd", "_robot2ee_pose.npy"), allow_pickle=True)
        ee2base_pose = switch_w(ee2base_pose)
    else:
        ee2base_pose = None

    pose_w_first = switch_w(ee)

    if base_pose is not None:
        pose_w_first = transform_pose2pose(base_pose, ee2base_pose)

    ee_frame = create_coordinate_frame(pose_w_first, switch_w=False)
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

    # R = pcd_th.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    # pcd_th = pcd_th.rotate(R, center=(0,0,0))
    # ee_frame = ee_frame.rotate(R, center=(0,0,0))

    o3d.visualization.draw_geometries(
        [pcd_th],
        # [pcd_th, ee_frame],
        # [pcd_th, ee_frame, kinect_frame],
        zoom=0.01,
        front=[0., -0., -0.9],
        lookat=[0, 0, 0],
        up=[-0., -0.9768, 0.8]
    )

    print(ee)

    # ipdb.set_trace()

    # o3d.io.write_point_cloud(
    #     sys.argv[1].replace(".pcd", "_processed.pcd"),
    #     pcd_th,
    #     print_progress=True
    # )
