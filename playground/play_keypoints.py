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
from utils.data import get_6_key_points, get_ee_cross_section_idx, get_ee_idx, get_key_points, get_roi_mask
from utils.visualization import create_coordinate_frame, generate_colors, generate_key_point_shapes
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line
from utils.preprocess import center_at_origin


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == "__main__":
    data, semantic_pred = file_utils.load_alive_file(sys.argv[1])

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
    else:
        points, rgb, labels, _, pose = data
    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

    arm_idx = np.where(labels == 1)[0]
    print('# of points:', len(rgb))
    print('# of arm points:', len(arm_idx))
    rgb[arm_idx] = np.zeros_like(rgb[arm_idx]) + np.array([0.3, 0.3, 0.3])
    points = points[arm_idx]
    rgb = rgb[arm_idx]
    labels = [arm_idx]
    arm_idx = np.arange(len(arm_idx))

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
    ee_idx = get_ee_idx(points, pose, switch_w=True, arm_idx=arm_idx) # switch_w=False in dataloader

    ee_rgb = rgb[ee_idx]
    ee_rgb = np.zeros_like(ee_rgb) + np.array([0.3, 0.3, 0.3])
    # rgb[ee_idx] = np.array([1.0, 1.0, 0.13])

    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)  # switch_w=False in dataloader
    ee_points = points[ee_idx]

    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))

    # new_ee_points, origin_offset = base_at_origin(new_ee_points)
    # new_ee_points, y_offset = center_at_y(new_ee_points)
    new_ee_pose_pos = new_ee_points[-1:]
    new_ee_points = new_ee_points[:-1]
    new_ee_pose_points, ee_pose_offset = center_at_origin(new_ee_pose_pos)
    new_ee_points -= ee_pose_offset
    # ipdb.set_trace()

    key_points, key_points_idx = get_6_key_points(ee_points, pose, switch_w=True)
    key_points = key_points[key_points_idx > -1]
    key_points_idx = np.where(key_points_idx > -1)[0]

    shapes = generate_key_point_shapes(
        list(zip(key_points_idx, key_points)),
        radius=0.016,
        shape='octahedron'
    )

    # # reverse the transformation above
    # new_ee_points_reverse = np.array(new_ee_points, copy=True)
    # new_ee_points_reverse += origin_offset
    # new_ee_points_reverse = (rot_mat @ np.concatenate((new_ee_points_reverse, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    # new_ee_points_reverse = new_ee_points_reverse[:-1]
    # ee_reverse_rgb = np.zeros_like(ee_rgb) + np.array([1.0, 0.13, 1.0])


    # _, min_y, min_z = new_ee_points.min(axis=0)
    # max_y = new_ee_points.max(axis=0)[1]

    # ee_pos_magic = np.array([-0.01, 0.0, min_z])
    # ee_pos_magic_reverse = ee_pos_magic + origin_offset
    # ee_pos_magic_reverse = rot_mat @ ee_pos_magic_reverse

    # ee_pos_magic_reverse_pose = ee_pos_magic_reverse.tolist() + ee_orientation
    # # ee_pos_magic_reverse_pose = new_ee_pose_pos.tolist() + [0] * 4
    # ee_magic_frame = create_coordinate_frame(ee_pos_magic_reverse_pose, switch_w=False)

    # ipdb.set_trace()
    points_show = np.concatenate((points, new_ee_points), axis=0)
    rgb = np.concatenate((rgb, ee_rgb), axis=0)
    # points_show = new_ee_points
    # rgb = ee_rgb


    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # pick_points(pcd)

    print('# of masked points:', len(rgb))
    o3d.visualization.draw_geometries(
        # [pcd, kinect_frame, ee_frame]
        [pcd, ee_frame, kinect_frame, shapes]
        # [pcd, ee_magic_frame, ee_frame]
        # [pcd]
    )