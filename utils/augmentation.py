import math

import numpy as np
import scipy
from scipy.stats import special_ortho_group


# Elastic distortion
def distort_elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32)//gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


def add_noise(x, sigma=0.0016, clip=0.005):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise


def transform_random(pc):
    tr = np.random.rand() * 0.04
    rot = special_ortho_group.rvs(3)
    pc = pc @ rot
    pc += np.array([[tr, 0, 0]])
    pc = pc @ rot.T

    return pc


def flip_random(pc):
    m = np.eye(3)
    m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
    return np.matmul(pc, m)


def rotate_along_gravity(pc):
    angle = np.random.rand() * 2 * np.pi
    rot = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    pc = (rot @ pc.T).T

    return pc


def augment(
    points,
    probability=0.2,
    copy=False,
    elastic=False,
    noise=False,
    transform=False,
    flip=False,
    gravity=False
):
    points = np.array(points, copy=copy)

    if elastic and np.random.rand() < probability:
        points = distort_elastic(points, 1, 4)

    if noise and np.random.rand() < probability:
        points = add_noise(points)

    if transform and np.random.rand() < probability:
        points = transform_random(points)

    if flip and np.random.rand() < probability:
        points = flip_random(points)

    if gravity and np.random.rand() < probability:
        points = rotate_along_gravity(points)

    return points
