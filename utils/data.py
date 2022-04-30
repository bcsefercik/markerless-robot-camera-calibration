import numpy as np

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
