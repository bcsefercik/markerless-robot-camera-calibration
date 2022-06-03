import torch
import numpy as np
from scipy.spatial.transform import Rotation
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


def get_transformation_matrix(pose, switch_w=False):
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)
    trans_mat = np.concatenate((rot_mat, pose[:3].reshape((3,1))), axis=1)
    trans_mat = np.concatenate((trans_mat, np.array([[0, 0, 0, 1]])), axis=0)

    return trans_mat


def get_transformation_matrix_inverse(trans_mat):
    response = np.array(trans_mat, copy=True)

    response[:3, :3] = trans_mat[:3, :3].T
    response[:3, 3] = (-response[:3, :3]) @ trans_mat[:3, 3]

    return response


def get_q_from_matrix(rot_mat):
    rot = Rotation.from_matrix(rot_mat).as_quat()
    rot = np.insert(rot[:3], 0, rot[-1])
    return rot  # w, x, y, z


def get_pose_from_matrix(trans_mat):
    translation = trans_mat[:3, 3]
    rotation = get_q_from_matrix(np.array(trans_mat[:3, :3], copy=True))

    pose = np.concatenate((translation, rotation))

    return pose  # x,y,z qw, qx, qy, qz


def get_quaternion_rotation_matrix_torch(quaternions: torch.Tensor) -> torch.Tensor:  # Input: WXYZ
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


def get_affine_transformation(inp, out):
    inp_len = len(inp)
    B = np.vstack([np.transpose(inp), np.ones(inp_len)])
    D = 1.0 / np.linalg.det(B)

    def entry(r, d):
        return np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))

    M = [[(-1) ** i * D * entry(R, i) for i in range(inp_len)] for R in np.transpose(out)]
    A, t = np.hsplit(np.array(M), [inp_len - 1])
    t = np.transpose(t)[0]

    return A, t


def get_rigid_transform_3D(reference, target):
    A = reference.T
    B = target.T

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t.reshape(-1)


if __name__ == '__main__':
    inp = [[1, 1, 2], [2, 3, 0], [3, 2, -2], [-2, 2, 3]]  # <- points
    out = [[0, 2, 1], [1, 2, 2], [-2, -1, 6], [4, 1, -3]]  # <- mapped to
    # calculations

    A, t = get_affine_transformation(inp, out)
    # output
    print("Affine transformation matrix:\n", A)
    print("Affine transformation translation vector:\n", t)
    # unittests
    print("TESTING:")
    for p, P in zip(np.array(inp), np.array(out)):
        image_p = np.dot(A, p) + t
        result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
        print(p, " mapped to: ", image_p, " ; expected: ", P, result)
