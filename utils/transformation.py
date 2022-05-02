import torch
import numpy as np

import ipdb


def get_quaternion_rotation_matrix(Q_init, switch_w=True):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q

    if switch_w:
        Q = np.insert(Q_init[:3], 0, Q_init[-1])  # put w to first place
    else:
        Q = Q_init  # w already at the first place

    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def get_quaternion_rotation_matrix_torch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Taken from: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def compute_vec_dist_to_line(p, lp1, lp2):
    return compute_dists_to_line(p.reshape((-1, 1)), lp1, lp2)[0]


def compute_dists_to_line(p, lp1, lp2):
    # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    # https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d
    d = (lp1 - lp2) / np.linalg.norm(lp1 - lp2)
    v = p - lp1
    t = np.dot(v, d)
    t = t.reshape((-1, 1))
    P = lp1 + t * d
    dists = np.linalg.norm(P - p, axis=1)
    return dists


def select_closest_points_to_line(points, lp1, lp2, count=0, cutoff=0.008):
    count = min(count, len(points)) if count > 0 else len(points)

    dists = compute_dists_to_line(points, lp2, lp1)
    dists_args_sorted = np.argsort(dists)

    cutoff_mask = dists[dists_args_sorted[:count]] < cutoff

    final_idx = dists_args_sorted[:count][cutoff_mask]

    return dists[final_idx], final_idx
