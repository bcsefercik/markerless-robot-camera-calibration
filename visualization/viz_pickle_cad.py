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
from utils.data import get_ee_idx, get_roi_mask
from utils.visualization import create_coordinate_frame, get_frame_from_pose


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
    else:
        points, rgb, labels, _, pose = data

    pose_w_first = transformation.switch_w(pose)
    print("pose:", pose)

    labels = np.array(labels, dtype=int)

    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as fp:
            limits = json.load(fp)
    else:
        limits = {"any": dict()}

    arm_idx = np.where(labels == 1)[0]
    ee_idx = get_ee_idx(points, pose_w_first, switch_w=False, arm_idx=arm_idx, ee_dim={
        'min_z': -0.02,
        'max_z': 0.12,
        'min_x': -0.05,
        'max_x': 0.05,
        'min_y': -0.11,
        'max_y': 0.11
    })
    labels[ee_idx] = 2

    print('# of points:', len(rgb))
    print('# of arm points:', arm_idx.sum())

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    ee_frame = create_coordinate_frame(pose, switch_w=True)
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

    textured_mesh = o3d.io.read_triangle_mesh("../app/hand_files/hand_notblender.obj")
    textured_mesh.paint_uniform_color(np.asarray([0, 0, 1.]))

    match_icp = icp.get_point2point_matcher("../app/hand_files/hand_notblender.obj")
    pose_w_first = match_icp(points[ee_idx], pose_w_first)

    tmat = transformation.get_transformation_matrix(pose_w_first, switch_w=False)
    textured_mesh.transform(tmat)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print('# of masked points:', len(rgb))

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, ee_frame, textured_mesh], key_to_callback
    )
