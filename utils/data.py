import os
from select import select
import sys
from webbrowser import get

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line
from utils.preprocess import center_at_origin

import ipdb


def get_farthest_point_sample_idx(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return centroids.astype(np.int32)


def get_farthest_point_sample(point, npoint):
    idx = get_farthest_point_sample_idx(point, npoint)
    return point[idx]


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


def get_ee_idx(points, pose, switch_w=True, ee_dim=None, arm_idx=None):  # in training switch_w = False
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

    ee_idx = np.where(ee_mask)[0]

    if arm_idx is not None:
        # remove ee idx which is not arm idx too
        ee_arm_match_idx = np.isin(ee_idx, arm_idx, assume_unique=True)
        ee_idx = ee_idx[ee_arm_match_idx]

    return ee_idx


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


def get_closest_point(p, points, maximize_dim=None):
    if len(points) < 1:
        return None

    if maximize_dim is not None:
        p = np.array(p, copy=True)
        p[maximize_dim] = points.max(axis=0)[maximize_dim]

    norms = np.linalg.norm(points - p, axis=1, ord=2)

    min_idx = norms.argmin()
    min_point, min_dist = points[min_idx], norms.min()

    return min_idx, min_point, min_dist


def get_key_points(ee_points, pose, switch_w=True, euclidean_threshold=0.018, ignore_label=-100):  # switch_w=False in dataloader
    new_ee_points = np.array(ee_points, copy=True)
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)
    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    new_ee_pos = new_ee_points[-1:]
    new_ee_points = new_ee_points[:-1]
    new_ee_pose_points, ee_pose_offset = center_at_origin(new_ee_pos)
    new_ee_points -= ee_pose_offset

    key_points = np.array([
        [0.02, 0.09, 0],
        [0.02, -0.09, 0],
        [0.014, 0.095, 0.07],
        [0.014, -0.095, 0.07],
        [0, 0.048, 0.12],  # gripper
        [0, -0.048, 0.12],  # gripper
        [-0.022, 0.09, 0],
        [-0.022, -0.09, 0],
        [-0.014, 0.095, 0.07],
        [-0.014, -0.095, 0.07]
    ])

    key_points_idx = np.zeros(len(key_points), dtype=np.long) + ignore_label

    front_side_mask = new_ee_points[:, 0] > 0.005
    front_side_idx = np.where(front_side_mask)[0]
    p1_idx, p1_closest, dist = get_closest_point(key_points[0], new_ee_points[front_side_mask])
    if p1_closest is not None and dist < euclidean_threshold:
        key_points[0] = p1_closest
        key_points_idx[0] = front_side_idx[p1_idx]
        key_points[6] = p1_closest + [-0.04, 0, 0]

    p2_idx, p2_closest, dist = get_closest_point(key_points[1], new_ee_points[front_side_mask])
    if p2_closest is not None and dist < euclidean_threshold:
        key_points[1] = p2_closest
        key_points_idx[1] = front_side_idx[p2_idx]
        key_points[7] = p2_closest + [-0.04, 0, 0]

    p3_idx, p3_closest, dist = get_closest_point(key_points[2], new_ee_points[front_side_mask])
    if p3_closest is not None and dist < euclidean_threshold:
        key_points[2] = p3_closest
        key_points_idx[2] = front_side_idx[p3_idx]
        key_points[8] = p3_closest + [-0.03, 0, 0]

    p4_idx, p4_closest, dist = get_closest_point(key_points[3], new_ee_points[front_side_mask])
    if p4_closest is not None and dist < euclidean_threshold:
        key_points[3] = p4_closest
        key_points_idx[3] = front_side_idx[p4_idx]
        key_points[9] = p4_closest + [-0.03, 0, 0]

    back_side_mask = new_ee_points[:, 0] < -0.01
    back_side_idx = np.where(back_side_mask)[0]
    if sum(back_side_mask) > 0:
        p7_idx, p7_closest, dist = get_closest_point(key_points[6], new_ee_points[back_side_mask])
        if p7_closest is not None and dist < euclidean_threshold:
            key_points_idx[6] = back_side_idx[p7_idx]
            key_points[6] = p7_closest

        p8_idx, p8_closest, dist = get_closest_point(key_points[7], new_ee_points[back_side_mask])
        if p8_closest is not None and dist < euclidean_threshold:
            key_points_idx[7] = back_side_idx[p8_idx]
            key_points[7] = p8_closest

        p9_idx, p9_closest, dist = get_closest_point(key_points[8], new_ee_points[back_side_mask])
        if p9_closest is not None and dist < euclidean_threshold:
            key_points_idx[8] = back_side_idx[p9_idx]
            key_points[8] = p9_closest

        p10_idx, p10_closest, dist = get_closest_point(key_points[9], new_ee_points[back_side_mask])
        if p10_closest is not None and dist < euclidean_threshold:
            key_points_idx[9] = back_side_idx[p10_idx]
            key_points[9] = p10_closest

    gripper_mask = new_ee_points[:, 2] > 0.08
    gripper_idx = np.where(gripper_mask)[0]
    gripper_selection = new_ee_points[gripper_mask]
    # P5 left gripper
    p5_closest = None
    if len(gripper_selection[gripper_selection[:, 1] > 0]) > 0:
        p5_idx, p5_closest, dist = get_closest_point(
            [0, 0.01, 0.1],
            gripper_selection[gripper_selection[:, 1] > 0],
            maximize_dim=2
        )
        if p5_closest is not None:
            key_points[4] = p5_closest
            key_points_idx[4] = gripper_idx[p5_idx]

    # P6 right gripper
    p6_closest = None
    if len(gripper_selection[gripper_selection[:, 1] < 0]) > 0:
        p6_idx, p6_closest, dist = get_closest_point(
            [0, -0.01, 0.1],
            gripper_selection[gripper_selection[:, 1] < 0],
            maximize_dim=2
        )
        if p6_closest is not None:
            key_points[5] = p6_closest
            key_points_idx[5] = gripper_idx[p6_idx]

    if p5_closest is None and p6_closest is not None:
        key_points[4] = (p6_closest * [1, -1, 1])
    elif p5_closest is not None and p6_closest is None:
        key_points[5] = (p5_closest * [1, -1, 1])

    key_points[4][2] = max(key_points[4][2], key_points[5][2])
    key_points[5][2] = key_points[4][2]

    key_points += ee_pose_offset
    key_points = (rot_mat @  key_points.reshape((-1, 3, 1))).reshape((-1, 3))

    return key_points, key_points_idx


def get_6_key_points(ee_points, pose, switch_w=True, euclidean_threshold=0.02, ignore_label=-100):
    new_ee_points = np.array(ee_points, copy=True)
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)
    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    new_ee_pos = new_ee_points[-1:]
    new_ee_points = new_ee_points[:-1]
    new_ee_pose_points, ee_pose_offset = center_at_origin(new_ee_pos)
    new_ee_points -= ee_pose_offset

    key_points = np.array([
        [0.02, 0.09, 0],  # P1: top left
        [0.02, -0.09, 0],  # P2: top right
        [0.014, 0.095, 0.07],  # P3: bottom left
        [0.014, -0.095, 0.07],  # P4: bottom right
        [0, 0.048, 0.12],  # gripper
        [0, -0.048, 0.12],  # gripper
    ])

    point_idx = np.ones(len(key_points), dtype=np.int) * ignore_label

    ee_mask = (new_ee_points[:, 0] > -0.005) * (new_ee_points[:, 2] < 0.09)
    ee_idx = np.where(ee_mask)[0]
    ee_selection = new_ee_points[ee_mask]

    ee_bbox = np.array([
        [0.24, 0.32, -0.2],  # P1: top left
        [0.24, -0.32, -0.2],  # P2: top right
        [0.24, 0.32, 0.2],  # P3: bottom left
        [0.24, -0.32, 0.2],  # P4: bottom right
    ])

    front_pidx = np.linalg.norm(ee_bbox.reshape((-1, 1, 3)) - ee_selection, axis=2).argmin(axis=1)
    front_kp_candidates = new_ee_points[ee_idx[front_pidx]]
    front_point_idx_candidates = ee_idx[front_pidx]
    dists_candidates = np.linalg.norm(key_points[:4] - front_kp_candidates, axis=1) < euclidean_threshold
    key_points[:4][dists_candidates] = front_kp_candidates[dists_candidates]
    point_idx[:4][dists_candidates] = front_point_idx_candidates[dists_candidates]

    gripper_mask = new_ee_points[:, 2] > 0.08
    gripper_idx = np.where(gripper_mask)[0]
    gripper_selection = new_ee_points[gripper_mask]
    # P5 left gripper
    p5_closest = None
    if len(gripper_selection[gripper_selection[:, 1] > 0]) > 0:
        p5_idx, p5_closest, dist = get_closest_point(
            [0, 0.01, 0.1],
            gripper_selection[gripper_selection[:, 1] > 0],
            maximize_dim=2
        )
        if p5_closest is not None:
            key_points[4] = p5_closest
            point_idx[4] = gripper_idx[p5_idx]

    # P6 right gripper
    p6_closest = None
    if len(gripper_selection[gripper_selection[:, 1] < 0]) > 0:
        p6_idx, p6_closest, dist = get_closest_point(
            [0, -0.01, 0.1],
            gripper_selection[gripper_selection[:, 1] < 0],
            maximize_dim=2
        )
        if p6_closest is not None:
            key_points[5] = p6_closest
            point_idx[5] = gripper_idx[p6_idx]

    if p5_closest is None and p6_closest is not None:
        key_points[4] = (p6_closest * [1, -1, 1])
    elif p5_closest  is not None and p6_closest is None:
        key_points[5] = (p5_closest * [1, -1, 1])

    key_points[4][2] = max(key_points[4][2], key_points[5][2])
    key_points[5][2] = key_points[4][2]

    key_points += ee_pose_offset
    key_points = (rot_mat @  key_points.reshape((-1, 3, 1))).reshape((-1, 3))

    return key_points, point_idx


def collect_closest_points(idx, points, euclidean_threshold=0.006):
    norms = np.linalg.norm(points[idx].reshape(-1,1,3) - points, axis=2)
    pcls_idx, p_idx = np.where(norms < euclidean_threshold)

    return pcls_idx, p_idx
