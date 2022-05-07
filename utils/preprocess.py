import ipdb

import numpy as np
import sklearn.preprocessing as preprocessing


def center_at_origin(points: np.array):
    origin_offset = (points.max(axis=0) + points.min(axis=0)) / 2
    points - origin_offset

    return points - origin_offset, origin_offset


def normalize_colors(rgb_input: np.array, is_color_in_range_0_255: bool = False):
    rgb = np.array(rgb_input, copy=True)

    if is_color_in_range_0_255:
        rgb /= 255

    if rgb.min() < 0:
        # WRONG approach, tries to get rid of trouble from data prep code.
        rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
        rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
        rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    if rgb.min() > (-1e-6) and rgb.max() < (1+1e-6):
        rgb -= 0.5

    return rgb
