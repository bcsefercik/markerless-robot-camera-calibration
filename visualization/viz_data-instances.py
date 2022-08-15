import os
import sys
import glob
import pickle
import argparse
import random

import open3d as o3d
import numpy as np

sys.path.append("..")  # noqa
from utils import file_utils
from utils.visualization import create_coordinate_frame


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alive")
    parser.add_argument("--ref", type=str, default="ref.pickle")
    parser.add_argument("--ee", type=str, default="ee_poses.pickle")
    args = parser.parse_args()

    kinect_mesh = o3d.io.read_triangle_mesh(
        os.path.join(
            BASE_PATH, "..", "app", "hand_files", "kinect.obj"
        )  # seems to work better
    )
    kinect_mesh.paint_uniform_color(np.array([0.1, 0.1, 0.1]))
    kinect_mesh.scale(0.001, np.array([0, 0, 0]))
    kinect_frame = create_coordinate_frame([-0.00773637, -0.03,  0.01641659, 1, 0, 0, 0], switch_w=False, radius=0.002)

    k_rot_x = np.matrix([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float64)
    k_rot_y = np.matrix([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ], dtype=np.float64)
    k_rot_z = np.matrix([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    kinect_mesh.rotate(k_rot_y)
    kinect_mesh.rotate(k_rot_x)

    ee_poses = list()
    with open(args.ee, "rb") as fp:
        ee_poses = pickle.load(fp, encoding="bytes")

    ee_poses = ee_poses[0:len(ee_poses):(len(ee_poses) // 100)]

    ee_frames = [
        create_coordinate_frame(ep, length=0.05, radius=0.0015, switch_w=True) for ep in ee_poses
    ]

    # ee_frames = ee_frames[0:len(ee_frames):(len(ee_frames) // 100)]

    data, _ = file_utils.load_alive_file(args.ref)
    points = data['points']
    rgb = data['rgb']
    labels = data['labels']
    bg = labels == 0

    points = points[bg]
    rgb = rgb[bg]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # ipdb.set_trace()

    o3d.visualization.draw_geometries(
        [pcd, kinect_mesh, kinect_frame] + ee_frames,
        # [pcd, kinect_mesh, kinect_frame],
        # [pcd, ee_frame, ref_shapes, obbox],
        zoom=0.1,
        front=[0., -0., -0.1],
        lookat=[0, -0.3, -0.2],
        up=[-0., -0.2768, -1.9]
    )
