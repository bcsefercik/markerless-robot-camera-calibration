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
from utils.data import get_ee_idx, get_roi_mask
from utils.visualization import create_coordinate_frame
from utils.transformation import get_quaternion_rotation_matrix
from utils.preprocess import center_at_origin


if __name__ == "__main__":

    (
        points,
        rgb,
        labels,
        instance_label,
        pose,
    ), semantic_pred = file_utils.load_alive_file(sys.argv[1])

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

    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)

    # ee_points = points[ee_idx]
    # ee_rgb = rgb[ee_idx]

    ee_points = points - pose[:3]
    ee_rgb = rgb

    # new_points = ee_points - pose[:3]

    ee_idx = get_ee_idx(points, pose, switch_w=True)

    new_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1,3))).reshape((-1, 3, 1))).reshape((-1, 3))

    # new_points, origin_offset = center_at_origin(new_points)
    new_points = new_points[:-1]
    # roi_mask = get_roi_mask(points)

    # new_points = new_points[roi_mask]
    # ee_rgb = ee_rgb[roi_mask]

    # if rgb.min() < 0:
    #     # WRONG approach, tries to shit from data prep code.
    #     rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
    #     rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
    #     rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)


    # # points_show = np.concatenate((points, new_points), axis=0)
    # # rgb = np.concatenate((rgb, ee_rgb), axis=0)
    points_show = points
    rgb = rgb

    rgb[ee_idx] = np.array([1.0, 1.0, 0.1])

    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print('# of masked points:', len(rgb))
    o3d.visualization.draw_geometries(
        # [pcd, kinect_frame, ee_frame]
        [pcd, kinect_frame]
    )
