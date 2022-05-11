import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line
from utils.preprocess import center_at_origin

import ipdb


def normalize_color(
    color, is_color_in_range_0_255: bool = False
):
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.

    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()


def get_roi_mask(points, min_x=-500, max_x=500, min_y=-500, max_y=500, min_z=-500, max_z=500, offset=0.0):
    max_x += offset
    max_y += offset
    max_z += offset
    min_x -= offset
    min_y -= offset
    min_z -= offset

    roi_mask = points[:, 0] > -500

    roi_mask = np.logical_and(points[:, 0] < max_x, roi_mask)  # x
    roi_mask = np.logical_and(points[:, 0] > min_x, roi_mask)
    roi_mask = np.logical_and(points[:, 1] < max_y, roi_mask)  # y
    roi_mask = np.logical_and(points[:, 1] > min_y, roi_mask)
    roi_mask = np.logical_and(points[:, 2] < max_z, roi_mask)  # z
    roi_mask = np.logical_and(points[:, 2] > min_z, roi_mask)

    return roi_mask


def get_ee_idx(points, pose, switch_w=True, ee_dim=None):  # in training switch_w = False
    if not isinstance(ee_dim, dict):
        ee_dim = {
            'min_z': -0,
            'max_z': 0.12,
            'min_x': -0.03,
            'max_x': 0.03,
            'min_y': -0.11,
            'max_y': 0.11
        }

    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)

    ee_points = points - pose[:3]
    new_points = (rot_mat.T @ ee_points.reshape((-1, 3, 1))).reshape((-1, 3))
    ee_mask = get_roi_mask(new_points, **ee_dim)

    return np.where(ee_mask)[0]


def get_ee_cross_section_idx(ee_points, pose, count=32, cutoff=0.004, switch_w=True):  # switch_w=False in dataloader
    new_ee_points = np.array(ee_points, copy=True)
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)

    new_ee_points -= pose[:3]

    new_ee_points = (rot_mat.T @ new_ee_points.reshape((-1, 3, 1))).reshape((-1, 3))

    closest_points_dists, closest_points_idx = select_closest_points_to_line(
        new_ee_points,
        np.array([-0.05, 0, 0]),
        np.array([0.05, 0, 0]),
        count=count,
        cutoff=cutoff
    )

    return closest_points_dists, closest_points_idx


def get_key_points(ee_points, pose, switch_w=True):  # switch_w=False in dataloader
    new_ee_points = np.array(ee_points, copy=True)
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)
    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    new_ee_pos = new_ee_points[-1:]
    new_ee_points = new_ee_points[:-1]
    new_ee_pose_points, ee_pose_offset = center_at_origin(new_ee_pos)
    new_ee_points -= ee_pose_offset

    key_points = np.array([
        [0.022, 0.09, 0],
        [0.022, -0.09, 0],
        [0.014, 0.095, 0.07],
        [0.014, -0.095, 0.07],
        [0, 0.048, 0.12],  # gripper
        [0, -0.048, 0.12],  # gripper
        [-0.022, 0.09, 0],
        [-0.022, -0.09, 0],
        [-0.014, 0.095, 0.07],
        [-0.014, -0.095, 0.07]
    ])

    key_points += ee_pose_offset
    key_points = (rot_mat @  key_points.reshape((-1, 3, 1))).reshape((-1, 3))

    return key_points
