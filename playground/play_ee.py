import pickle
import sys
import os
import copy
import math
import json

import ipdb
import sklearn.preprocessing as preprocessing

import numpy as np
import open3d as o3d


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils
from utils.data import get_ee_cross_section_idx, get_ee_idx, get_roi_mask
from utils.visualization import create_coordinate_frame
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line
from utils.preprocess import center_at_origin


if __name__ == "__main__":
    data, semantic_pred = file_utils.load_alive_file(sys.argv[1])

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
    else:
        points, rgb, labels, _, pose = data
        points = np.array(points, dtype=np.float32)
        rgb = np.array(rgb, dtype=np.float32)
        pose = np.array(pose, dtype=np.float32)

    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

    arm_idx = labels == 1

    print('# of points:', len(rgb))
    print('# of arm points:', arm_idx.sum())

    # points = points[arm_idx]
    # rgb = rgb[arm_idx]
    # labels = [arm_idx]

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    kinect_frame = copy.deepcopy(frame).translate([0, 0, 0])
    kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

    # kinect_frame = create_coordinate_frame([0] * 7, switch_w=False)
    ee_frame = create_coordinate_frame(pose, switch_w=True)



    # ee_points = points - pose[:3]
    # ee_rgb = rgb

    # new_ee_points = ee_points - pose[:3]
    #
    ee_idx = get_ee_idx(points, pose, switch_w=True) # switch_w=False in dataloader

    closest_points_dists, closest_points_idx = get_ee_cross_section_idx(
        points[ee_idx],  # ee_points
        pose,
        switch_w=True
    )  # switch_w=False in dataloader

    ee_rgb = rgb[ee_idx]
    rgb[ee_idx] = np.array([1.0, 1.0, 0.13])

    rgb[ee_idx[closest_points_idx]] = np.array([1.0, 0, 0])
    # ipdb.set_trace()

    # new_ee_points, origin_offset = center_at_origin(new_ee_points)
    # new_ee_points = new_ee_points[:-1]
    # roi_mask = get_roi_mask(points)

    # new_ee_points = new_ee_points[roi_mask]
    # ee_rgb = ee_rgb[roi_mask]

    # if rgb.min() < 0:
    #     # WRONG approach, tries to shit from data prep code.
    #     rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
    #     rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
    #     rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)  # switch_w=False in dataloader
    ee_points = points[ee_idx]

    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))

    ee_rgb[closest_points_idx] = np.array([1.0, 0, 0])
    new_ee_points, origin_offset = center_at_origin(new_ee_points)
    new_ee_pose_pos = new_ee_points[-1]
    new_ee_points = new_ee_points[:-1]

    # reverse the transformation above
    new_ee_points_reverse = np.array(new_ee_points, copy=True)
    new_ee_points_reverse += origin_offset
    new_ee_points_reverse = (rot_mat @ np.concatenate((new_ee_points_reverse, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    new_ee_points_reverse = new_ee_points_reverse[:-1]
    ee_reverse_rgb = np.zeros_like(ee_rgb) + np.array([1.0, 0.13, 1.0])


    _, min_y, min_z = new_ee_points.min(axis=0)
    max_y = new_ee_points.max(axis=0)[1]

    ee_pos_magic = np.array([-0.01, 0.0, min_z])
    ee_pos_magic_reverse = ee_pos_magic + origin_offset
    ee_pos_magic_reverse = rot_mat @ ee_pos_magic_reverse

    ee_pos_magic_reverse_pose = ee_pos_magic_reverse.tolist() + ee_orientation
    # ee_pos_magic_reverse_pose = new_ee_pose_pos.tolist() + [0] * 4
    ee_magic_frame = create_coordinate_frame(ee_pos_magic_reverse_pose, switch_w=False)

    # ipdb.set_trace()
    points_show = np.concatenate((points, new_ee_points, new_ee_points_reverse), axis=0)
    rgb = np.concatenate((rgb, ee_rgb, ee_reverse_rgb), axis=0)
    # points_show = new_ee_points
    # rgb = ee_rgb


    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    ipdb.set_trace()

    print('# of masked points:', len(rgb))
    o3d.visualization.draw_geometries(
        [pcd, kinect_frame, ee_frame]
        # [pcd, kinect_frame]
        # [pcd, ee_magic_frame, ee_frame]
        # [pcd]
    )
