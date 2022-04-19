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
from utils.data import get_roi_mask



if __name__ == "__main__":
    position = "p2_1"

    (
        points,
        rgb,
        labels,
        instance_label,
        pose,
    ), semantic_pred = file_utils.load_alive_file(sys.argv[1])

    with open(sys.argv[2], 'r') as fp:
        limits = json.load(fp)

    pred = [0] * 7

    pred = [
       -0.0562,
        0.0785,
        0.5898,
        0.5027,
        -0.3632,
        0.6574,
        0.5241
    ]
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

    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

    ee_frame = copy.deepcopy(frame)
    ee_frame.translate(ee_position)

    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    min_point = - np.array([0.2, 0.15, 0])
    max_point = + np.array([0.2, 0.15, 0.2])

    # bbox = o3d.geometry.AxisAlignedBoundingBox(
    #     np.array(min_point).reshape((3, 1)),
    #     np.array(max_point).reshape((3, 1)),
    # )
    # obbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
    # # obbox = ee_frame.get_oriented_bounding_box()
    #
    # obbox.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))
    # obbox.translate(ee_position)
    # ipdb.set_trace()
    offset = frame.get_rotation_matrix_from_quaternion(ee_orientation) @ np.array([0, 0, 0.03])
    obbox = o3d.geometry.OrientedBoundingBox(
        ee_position,
        frame.get_rotation_matrix_from_quaternion(ee_orientation),
        np.array([0.15, 0.27, 0.18])
    )
    obbox.color = [1, 0, 0]
    obbox.center = obbox.get_center() + offset

    # ipdb.set_trace()

    roi_mask = get_roi_mask(points, **limits[position])

    points = points[roi_mask]
    rgb = rgb[roi_mask]
    if rgb.min() < 0:
        # WRONG approach, tries to shit from data prep code.
        rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
        rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
        rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    obbox_points = np.asarray(obbox.get_box_points())
    obbox_points_min = obbox_points.min(axis=0)
    obbox_points_max = obbox_points.max(axis=0)

    bbox2 = o3d.geometry.AxisAlignedBoundingBox(
        obbox_points_min.reshape((3, 1)),
        obbox_points_max.reshape((3, 1)),
    )
    obbox2 = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox2)
    obbox2.color = [0, 0, 1]
    # ipdb.set_trace()

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    obbox_mask = obbox.get_point_indices_within_bounding_box(pcd.points)
    # ipdb.set_trace()

    pcd.points = o3d.utility.Vector3dVector(points[obbox_mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb[obbox_mask])
    # pcd = pcd.crop(obbox)
    # ipdb.set_trace()


    print('# of masked points:', len(rgb))

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, ee_frame, kinect_frame, obbox], key_to_callback
    )
