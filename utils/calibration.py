import os
import sys

import numpy as np
from scipy.special import erfcinv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_outliers(y: np.array):
    sqrt_2 = np.sqrt(2)
    erfcinv_15 = erfcinv(3/2)

    is_outlier = np.abs(y) > (-3 / (sqrt_2 * erfcinv_15) * np.median(np.abs(y - np.median(y))))

    return is_outlier, y[is_outlier]


def remove_outliers(y: np.array):
    is_outlier, _ = get_outliers(y)
    return np.array(y[np.logical_not(is_outlier)], copy=True)


# average_quaternions and average_quaternions_weighted are taken from:
# https://github.com/christophhagen/averaging-quaternions
# Paper: https://ntrs.nasa.gov/citations/20070017872

# Average multiple quaternions with specific weights
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
def average_quaternions_weighted(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.matlib.zeros(shape=(4, 4))
    weightSum = np.sum(w)

    for i in range(0, M):
        q = Q[i, :]
        A = w[i] * np.outer(q, q) + A

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].A1)


# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def average_quaternions(Q):
    return average_quaternions_weighted(Q, np.ones(Q.shape[0]))
