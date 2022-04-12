import sys
import os
import copy
import math

import ipdb

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

if __name__ == "__main__":
    (
        points,
        rgb,
        labels,
        instance_label,
        pose,
    ), semantic_pred = file_utils.load_alive_file(sys.argv[1])

    pred = [0] * 7

    pred = [
        -0.2567,
        -0.0931,
        0.779,
        0.5247,
        -0.3852,
        0.3667,
        0.3976
    ]
    arm_idx = labels == 1

    print('# of points:', len(rgb))
    print('# of arm points:', arm_idx.sum())

    roll_x, pitch_y, yaw_z = euler_from_quaternion(*pose[3:])
    pred_roll_x, pred_pitch_y, pred_yaw_z = euler_from_quaternion(pred[4], pred[5], pred[6], pred[3])

    print(f'GT euler: roll_x: {roll_x}, pitch_y: {pitch_y}, yaw_z: {yaw_z}')
    print(f'PRED euler: roll_x: {pred_roll_x}, pitch_y: {pred_pitch_y}, yaw_z: {pred_yaw_z}')

    min_point = points[arm_idx].min(axis=0)
    max_point = points[arm_idx].max(axis=0)

    # ipdb.set_trace()

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        np.array(min_point).reshape((3, 1)),
        np.array(max_point).reshape((3, 1)),
    )
    obbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
    obbox.color = [1, 0, 0]

    # points = points[arm_idx]
    # rgb = rgb[arm_idx]
    # labels = [arm_idx]

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]
    ee_frame = copy.deepcopy(frame).translate(ee_position)
    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))
    ee_position_pred = pred[:3]
    ee_orientation_pred = pred[3:]
    # ee_orientation_pred = ee_orientation_pred[-1:] + ee_orientation_pred[:-1]
    ee_frame_pred = copy.deepcopy(frame).translate(ee_position_pred)
    ee_frame_pred.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation_pred))
    kinect_frame = copy.deepcopy(frame).translate([0, 0, 0])
    kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

    # obbox.translate(ee_position).rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    roi_mask = points[:, 0] > -500
    # roi_mask = np.logical_and(points <= max_point, points >= min_point).sum(axis=1) == 3

    # p2_1
    roi_mask = np.logical_and(points[:, 0] < 0.39, roi_mask)  # x
    roi_mask = np.logical_and(points[:, 0] > -0.52, roi_mask)
    roi_mask = np.logical_and(points[:, 1] < 0.27, roi_mask)  # y
    # # # roi_mask = np.logical_and(points[:, 1] > -0.02, roi_mask)
    roi_mask = np.logical_and(points[:, 2] < 1.15, roi_mask)  # z
    roi_mask = np.logical_and(points[:, 2] > 0.3, roi_mask)

    points = points[roi_mask]
    rgb = rgb[roi_mask]

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, ee_frame, kinect_frame, ee_frame_pred, obbox], key_to_callback
    )
