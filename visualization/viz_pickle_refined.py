from hashlib import new
import sys
import os
import copy
import math
import pickle
import json

import ipdb
import sklearn.preprocessing as preprocessing

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils, icp, transformation
from utils.data import get_6_key_points, get_ee_idx, get_roi_mask
from utils.visualization import create_coordinate_frame, generate_key_point_shapes, get_frame_from_pose


show_labels = False

if __name__ == "__main__":
    position = "any"

    _seg_colors = [
        (int(color[0:2], 16), int(color[2:4], 16), int(color[4:], 16))
        for color in ['2C3E50', 'E74C3C', 'F1C40F']
    ]
    _seg_colors = np.array(_seg_colors) / 255

    data, _ = file_utils.load_alive_file(sys.argv[1])

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
        ee2base_pose = data.get("robot2ee_pose")
        ee2base_pose_w_first = transformation.switch_w(ee2base_pose)
    else:
        points, rgb, labels, _, pose = data

    pose_w_first = transformation.switch_w(pose)

    labels = np.array(labels, dtype=int)

    arm_idx = np.where(labels == 1)[0]
    ee_idx = np.where(labels == 2)[0]
    if len(ee_idx) < 1:
        ee_idx = get_ee_idx(points, pose_w_first, switch_w=False, arm_idx=arm_idx, ee_dim={
            'min_z': -0.025,
            'max_z': 0.12,
            'min_x': -0.05,
            'max_x': 0.05,
            'min_y': -0.11,
            'max_y': 0.11
        })
        labels[ee_idx] = 2

    # labels[arm_idx[:1000]] = 2
    # min_yo = np.linalg.norm(points-np.array([-3, 3, 5]), axis=1).argsort()[:500]
    # labels[min_yo] = 2
    # min_yo = np.linalg.norm(points-np.array([-4, -1, 5]), axis=1).argsort()[:500]
    # labels[min_yo] = 2
    # ipdb.set_trace()

    if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
        with open(sys.argv[2], 'r') as fp:
            limits = json.load(fp)
    else:
        limits = {"any": dict()}

    original_base_pose = transformation.get_base2cam_pose(pose_w_first, ee2base_pose_w_first)
    match_icp = icp.get_point2point_matcher("../app/hand_files/hand_notblender.obj")
    if len(sys.argv) > 2 and not os.path.isfile(sys.argv[2]) and sys.argv[2] == '-fit':
        pose_w_first = match_icp(points[ee_idx], pose_w_first)
        labels[ee_idx] = 1
        ee_idx = get_ee_idx(points, pose_w_first, switch_w=False, arm_idx=arm_idx)
        labels[ee_idx] = 2

    base_pose = transformation.get_base2cam_pose(pose_w_first, ee2base_pose_w_first)

    print("original:", original_base_pose)
    print("refined:", base_pose)
    print('nane')

    print('# of points:', len(rgb))
    print('# of arm points:', len(arm_idx))

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    ee_frame = create_coordinate_frame(pose_w_first, switch_w=False)
    kinect_frame = get_frame_from_pose(frame, [0] * 7)
    roi_mask = get_roi_mask(points, **limits[position])

    points = points[roi_mask]
    rgb = rgb[roi_mask]
    if rgb.min() < 0:
        # WRONG approach, tries to shit from data prep code.
        rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
        rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
        rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    def switch_to_normal(vis):
        global show_labels
        show_labels = not show_labels
        pcd.colors = o3d.utility.Vector3dVector(_seg_colors[labels] if show_labels else rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)

    show_labels = True
    pcd.colors = o3d.utility.Vector3dVector(_seg_colors[labels] if show_labels else rgb)
    print('# of masked points:', len(rgb))

    ref_key_points, ref_p_idx = get_6_key_points(points[ee_idx], pose_w_first, switch_w=False)
    ref_shapes = generate_key_point_shapes(
        list(zip(list(range(len(ref_p_idx))), ref_key_points)),
        radius=0.008,
    )

    offset = frame.get_rotation_matrix_from_quaternion(pose_w_first[3:]) @ np.array([0, 0, 0.05])
    obbox = o3d.geometry.OrientedBoundingBox(
        pose_w_first[:3],
        frame.get_rotation_matrix_from_quaternion(pose_w_first[3:]),
        np.array([0.1, 0.24, 0.12])
    )
    obbox.color = [1, 1, 1]
    obbox.center = obbox.get_center() + offset

    key_to_callback = {ord("K"): switch_to_normal}
    # o3d.visualization.draw_geometries_with_key_callbacks(
    #     [pcd, ee_frame],
    #     key_to_callback
    # )
    o3d.visualization.draw_geometries(
        [pcd],
        # [pcd, ee_frame, ref_shapes, obbox],
        zoom=0.01,
        front=[0., -0., -0.9],
        lookat=[0, 0, 0],
        up=[-0., -0.2768, 0.9]
    )
