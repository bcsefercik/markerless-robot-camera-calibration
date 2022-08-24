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
from utils.transformation import (
    get_pose_inverse,
    get_transformation_matrix,
    switch_w,
    transform_pose2pose,
)
from utils.data import get_roi_mask
from utils.metrics import compute_rotational_diff, compute_translational_diff


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

import ipdb


def create_kinect(
    path,
    cam2base_pose,
    coordinate_frame_enabled=True,
    scale=0.001,
    color=[0.1, 0.1, 0.1],
):
    data, _ = file_utils.load_alive_file(path)
    pose = data["pose"]
    pose = switch_w(pose)
    robot2ee_pose = data["robot2ee_pose"]
    ee2robot_pose = get_pose_inverse(switch_w(robot2ee_pose))
    cam2base = transform_pose2pose(pose, ee2robot_pose)

    pose_inv = get_pose_inverse(cam2base)
    kinect_pose = transform_pose2pose(cam2base_pose, pose_inv)
    kinect_mesh = get_kinect_mesh(
        kinect_pose,
        coordinate_frame_enabled=coordinate_frame_enabled,
        scale=scale,
        color=color,
    )

    rot_diff = compute_rotational_diff(np.array([1, 0, 0, 0]), cam2base[3:])
    tr_diff = compute_translational_diff(np.array([0, 0, 0]), cam2base[:3])

    print(path)
    print(f'Rotational change:\t{round(rot_diff, 1)} degrees')
    print(f'Translation dist:\t{round(tr_diff, 1)} cm')

    return kinect_mesh, kinect_pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alive")
    parser.add_argument("--ref", type=str, default="ref.pickle")
    parser.add_argument("--kinects", type=str, nargs="+")
    parser.add_argument("--test_kinects", type=str, nargs="+")
    args = parser.parse_args()

    kinect_mesh = get_kinect_mesh(
        np.array([0, 0, 0, 1.0, 0, 0, 0]), coordinate_frame_enabled=True
    )

    robot_mesh = o3d.io.read_triangle_mesh(
        os.path.join(
            BASE_PATH, "..", "app", "hand_files", "franka_emika_panda.obj"
        )  # seems to work better
    )
    robot_mesh.paint_uniform_color(np.array([210, 218, 255]) / 255)
    # robot_mesh.scale(0.001, np.array([0, 0, 0]))

    reflect = np.matrix(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    k_rot_z = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    robot_mesh.rotate(k_rot_z)
    # ipdb.set_trace()
    robot_mesh.transform(reflect)
    # robot_mesh.translate(np.array([0.3,  -0.24,  0.17]))
    robot_mesh.translate(np.array([0.0, 0.0, 0.0]))

    data, _ = file_utils.load_alive_file(args.ref)
    points = data["points"]
    rgb = data["rgb"]
    pose = data["pose"]
    labels = data["labels"]
    # points = points[labels == 0]
    # rgb = rgb[labels == 0]
    pose = switch_w(pose)
    robot2ee_pose = data["robot2ee_pose"]
    ee2robot_pose = get_pose_inverse(switch_w(robot2ee_pose))
    cam2base = transform_pose2pose(pose, ee2robot_pose)
    cam2base_mtx = get_transformation_matrix(cam2base)

    kinects = [
        create_kinect(kp, cam2base, scale=0.0012) for kp in args.kinects
    ]
    kinect_meshes = [t[0] for t in kinects]
    kinect_poses = [t[1] for t in kinects]
    # kinects = [
    #     get_kinect_mesh(cam2base, coordinate_frame_enabled=True)
    #     ]

    # robot_mesh.transform(cam2base_mtx)

    test_kinects = [
        create_kinect(kp, cam2base, scale=0.0012, color=[0.6, 0.1, 0.1]) for kp in args.test_kinects
    ]
    test_kinect_meshes = [t[0] for t in test_kinects]
    test_kinect_poses = [t[1] for t in test_kinects]

    rot_dists = dict()
    trs_dists = dict()

    for i, trn_title in enumerate(args.kinects):
        trn_title_parts = trn_title.split("/p")
        trn_position = trn_title_parts[1].split("_")[0]

        for j, tst_title in enumerate(args.test_kinects):
            tst_title_parts = tst_title.split("/p")
            tst_position = tst_title_parts[1].split("_")[0]

            key = f"P{trn_position}_trn, P{tst_position}_tst"

            rot_diff = compute_rotational_diff(kinect_poses[i][3:], test_kinect_poses[j][3:])
            trs_diff = compute_translational_diff(kinect_poses[i][:3], test_kinect_poses[j][:3])

            rot_dists[key] = rot_diff
            trs_dists[key] = trs_diff

            print(f'{key} {round(rot_diff, 2)} degrees')
            print(f'{key} {round(trs_diff, 2)} cm')

    # print('rot:', rot_dists)
    # print('trans:', trs_dists)


    # robot_mesh.rotate(k_rot_y)

    roi_mask = get_roi_mask(
        points,
        max_z=1.5,
        max_y=0.4,
        max_x=0.45,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[roi_mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb[roi_mask])

    # ipdb.set_trace()

    o3d.visualization.draw_geometries(
        [pcd] + kinect_meshes + test_kinect_meshes,
        # [pcd, robot_mesh] + kinects,
        # [pcd, kinect_mesh, kinect_frame],
        # [pcd, ee_frame, ref_shapes, obbox],
        # zoom=0.2,
        # front=[0., -0., -0.1],
        # lookat=[0, -0.3, -0.2],
        # up=[-0., -0.2768, -1.9]
    )
