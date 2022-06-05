from turtle import pos
import ipdb

import os
import sys

import numpy as np
import numpy.matlib as npmat
from scipy.special import erfcinv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import metrics


def get_outliers(y: np.array, m=2.0):
    # MATLAB method, did not work. ?
    # sqrt_2 = np.sqrt(2)
    # erfcinv_15 = erfcinv(3/2)
    # is_outlier = np.abs(y) > (-3 / (sqrt_2 * erfcinv_15) * np.median(np.abs(y - np.median(y))))

    d = np.abs(y - np.median(y))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    is_outlier = s > m

    return is_outlier, y[is_outlier]


def remove_outliers(y: np.array):
    is_outlier, _ = get_outliers(y)
    return np.array(y[np.logical_not(is_outlier)], copy=True)


def get_pose_outliers(poses: np.array) -> np.array:
    '''
    poses: Nx7 or Nx3
    '''
    ref = np.array([0, 0, 0, 1., 0, 0, 0], dtype=np.float32)

    outliers = np.zeros(len(poses), dtype=bool)

    for i in range(3):
        outliers = outliers + get_outliers(poses[:, i])[0]

    if poses.shape[1] == 7:
        angle_diffs = np.array(
            [metrics.compute_pose_metrics(ref, poses[i, :])['angle_diff'] for i in range(len(poses))]
        )

        outliers = outliers + get_outliers(angle_diffs, m=4)[0]

    return outliers, poses[outliers, :]


def remove_pose_outliers(poses: np.array) -> np.array:
    '''
    poses: Nx7 or Nx3
    '''
    is_outlier, _ = get_pose_outliers(poses)
    return np.array(poses[np.logical_not(is_outlier), :], copy=True)



# average_quaternions and average_quaternions_weighted are taken from:
# https://github.com/christophhagen/averaging-quaternions
# Paper: https://ntrs.nasa.gov/citations/20070017872

# Average multiple quaternions with specific weights
def compute_quaternions_weighted_average(Q, w):
    '''
    Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
    The quaternions are arranged as (w,x,y,z), with w being the scalar
    The weight vector w must be of the same length as the number of rows in the
    quaternion maxtrix Q
    '''
    # Number of quaternions to average
    M = Q.shape[0]
    A = npmat.zeros(shape=(4, 4))
    weightSum = np.sum(w)

    for i in range(0, M):
        q = Q[i, :]
        A = w[i] * np.outer(q, q) + A

    # scale
    A = (1.0 / weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].A1)


def compute_quaternions_average(Q):
    '''
    Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
    The quaternions are arranged as (w,x,y,z), with w being the scalar
    The result will be the average quaternion of the input. Note that the signs
    of the output quaternion can be reversed, since q and -q describe the same orientation
    '''
    return compute_quaternions_weighted_average(Q, np.ones(Q.shape[0]))


def compute_translations_average(t, weights=None):
    if weights is None:
        weights = np.ones(len(t))

    weights_sum = np.sum(weights)

    return np.sum(t * weights.reshape(-1, 1), axis=0) / weights_sum


def compute_poses_average(poses, weights=None):
    '''
    poses: Nx7; x, y, z, qw, qx, qy, qz
    weights: N
    '''

    if weights is None or len(weights) != len(poses):
        weights = np.ones(len(poses))

    if len(poses.shape) != 2:
        poses = np.array(poses.reshape(-1, 7), copy=True)

    if len(poses) == 1:
        return poses[0]

    pose_avg = np.zeros(7)

    pose_avg[:3] = compute_translations_average(poses[:, :3], weights=weights)
    pose_avg[3:] = compute_quaternions_weighted_average(poses[:, 3:], weights)

    return pose_avg
