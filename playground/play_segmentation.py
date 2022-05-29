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
from utils.data import collect_closest_points, get_6_key_points, get_ee_cross_section_idx, get_ee_idx, get_key_points, get_roi_mask
from utils.visualization import create_coordinate_frame, generate_colors, generate_key_point_shapes, get_key_point_colors
from utils.transformation import get_quaternion_rotation_matrix, get_rigid_transform_3D, get_affine_transformation
from utils.preprocess import center_at_origin
from utils import augmentation as aug



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

    arm_idx = np.where(labels == 1)[0]
    print('# of points:', len(rgb))
    print('# of arm points:', len(arm_idx))

    # rgb[arm_idx] *= 0
    # rgb[arm_idx] += 0.3

    # rgb = aug.change_background(rgb, labels, "/Users/bugra.sefercik/Desktop/bcs_pp.jpg")
    # rgb = aug.change_background(rgb, labels, "/Users/bugra.sefercik/Desktop/D1x0ApAUwAAMr6C.jpg")
    points = aug.augment_segmentation(
        points,
        scale=200,
        elastic=True,
        noise=True,
        transform=True,
        flip=True,
        gravity=True
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print('# of masked points:', len(rgb))
    o3d.visualization.draw_geometries(
        # [pcd, kinect_frame, ee_frame]
        [pcd]
        # [pcd, ee_frame, kinect_frame, shapes]
        # [pcd, ee_magic_frame, ee_frame]
        # [pcd]
    )
