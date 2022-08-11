import os
import sys
import glob
import pickle
import argparse

import open3d as o3d
import numpy as np


sys.path.append("..")  # noqa
from utils import file_utils
from utils.visualization import create_coordinate_frame, get_kinect_mesh
from utils.transformation import get_pose_inverse, switch_w, transform_pose2pose
from utils.data import get_roi_mask


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

import ipdb


def create_kinect(path, cam2base_pose, coordinate_frame_enabled=True, scale=0.001):
    data, _ = file_utils.load_alive_file(path)
    pose = data['pose']
    pose = switch_w(pose)
    robot2ee_pose = data['robot2ee_pose']
    ee2robot_pose = get_pose_inverse(switch_w(robot2ee_pose))
    cam2base = transform_pose2pose(pose, ee2robot_pose)

    pose_inv = get_pose_inverse(cam2base)
    kinect_pose = transform_pose2pose(cam2base_pose, pose_inv)
    kinect_mesh = get_kinect_mesh(kinect_pose, coordinate_frame_enabled=coordinate_frame_enabled, scale=scale)

    return kinect_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alive")
    parser.add_argument("--ref", type=str, default="ref.pickle")
    parser.add_argument("--kinects", type=str, nargs="+")
    args = parser.parse_args()

    kinect_mesh = get_kinect_mesh(np.array([0, 0, 0, 1.0, 0, 0, 0]), coordinate_frame_enabled=True)

    data, _ = file_utils.load_alive_file(args.ref)
    points = data['points']
    rgb = data['rgb']
    pose = data['pose']
    pose = switch_w(pose)
    robot2ee_pose = data['robot2ee_pose']
    ee2robot_pose = get_pose_inverse(switch_w(robot2ee_pose))
    cam2base = transform_pose2pose(pose, ee2robot_pose)

    kinects = [create_kinect(kp, cam2base, scale=0.0012) for kp in args.kinects + [args.ref]]
    # kinects = [
    #     get_kinect_mesh(cam2base, coordinate_frame_enabled=True)
    #     ]

    roi_mask = get_roi_mask(
        points,
        max_z=1.5,
        max_y=0.5
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[roi_mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb[roi_mask])

    # ipdb.set_trace()

    o3d.visualization.draw_geometries(
        [pcd] + kinects,
        # [pcd, kinect_mesh, kinect_frame],
        # [pcd, ee_frame, ref_shapes, obbox],
        # zoom=0.2,
        # front=[0., -0., -0.1],
        # lookat=[0, -0.3, -0.2],
        # up=[-0., -0.2768, -1.9]
    )
