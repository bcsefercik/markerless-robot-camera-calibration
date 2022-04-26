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


def get_frame_from_pose(base_frame, pose, switch_w=True):
    frame = copy.deepcopy(base_frame)

    if not isinstance(pose, list):
        pose = pose.tolist()

    ee_position = pose[:3]
    ee_orientation = pose[3:]
    if switch_w:
        ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]
    ee_frame = frame.translate(ee_position)
    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    return ee_frame


if __name__ == "__main__":
    position = "c2_p2_1_full_i5"

    data, semantic_pred = file_utils.load_alive_file(sys.argv[1])

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
    else:
        points, rgb, labels, _, pose = data


    with open(sys.argv[2], 'r') as fp:
        limits = json.load(fp)

    pred = [0] * 7

    # pred = [
    #     3.8387,
    #     -1.1463,
    #     -0.3983,
    #     0.6027, -0.4181,  0.5901,  0.3373
    # ]

    # for checking only angle
    # pred[:3] = pose[:3]

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

    ee_frame = get_frame_from_pose(frame, pose, switch_w=True)
    ee_frame_pred = get_frame_from_pose(frame, pred, switch_w=False)
    kinect_frame = get_frame_from_pose(frame, [0] * 7)

    # obbox.translate(ee_position).rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))
    roi_mask = get_roi_mask(points, **limits[position])

    points = points[roi_mask]
    rgb = rgb[roi_mask]
    if rgb.min() < 0:
        # WRONG approach, tries to shit from data prep code.
        rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
        rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
        rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print('# of masked points:', len(rgb))

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, ee_frame, kinect_frame, ee_frame_pred, obbox], key_to_callback
    )
