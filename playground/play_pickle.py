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
    # else:
    #     points, rgb, labels, _, pose = data

    arm_idx = np.where(labels == 1)[0]
    # ee_idx = get_ee_idx(points, pose, switch_w=True, arm_idx=arm_idx) # switch_w=False in dataloader
    # # labels[ee_idx] = 2
    # pose_w_first = transformation.switch_w(pose)
    # icp_m = icp.get_point2point_matcher()
    # pose = icp_m(points[ee_idx], pose_w_first)
    # pose = np.concatenate((pose[:3], pose[4:], [pose[3]]))

    ee_idx = get_ee_idx(points, pose, switch_w=True, arm_idx=arm_idx) # switch_w=False in dataloader
    labels[ee_idx] = 2

    data_json = {
        "points": points,
        "rgb": rgb,
        "labels": labels,
        "instance_labels": labels,
        "pose": pose,
        "joint_angles": None
    }
    # ipdb.set_trace()
    # with open(sys.argv[1], 'wb') as fp:
    #     pickle.dump(data_json, fp)

    labels = np.array(labels, dtype=int)

    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as fp:
            limits = json.load(fp)
    else:
        limits = {"any": dict()}

    arm_idx = labels == 1

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

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print('# of masked points:', len(rgb))

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, ee_frame, kinect_frame], key_to_callback
    )
