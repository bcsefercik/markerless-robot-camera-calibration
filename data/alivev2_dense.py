import json
import os
import pickle
import glob
from random import sample
import ipdb

import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from data.alivev2 import AliveV2Dataset
from model.pointnet2_utils import farthest_point_sample

from utils import file_utils, logger, config
from utils.data import get_ee_idx, get_roi_mask, get_key_points, get_6_key_points, collect_closest_points, get_farthest_point_sample_idx
from utils.preprocess import center_at_origin
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line


_config = config.Config()
_logger = logger.Logger().get()


class AliveV2DenseDataset(AliveV2Dataset):

    def __getitem__(self, i):
        points, rgb, labels, _, pose, _, other = self.load_generic_data(i)

        sample_idx = get_farthest_point_sample_idx(
            points,
            _config.DATA.num_of_dense_input_points
        )

        points = points[sample_idx]
        rgb = rgb[sample_idx]
        labels = labels[sample_idx]

        if _config.DATA.keypoints_enabled:
            labels = self.load_key_points(i, points, pose, labels, p2p_label=False)

        points, pose, other = self.conduct_post_point_normalization(points, pose, other)

        if _config.DATA.use_coordinates_as_features:
            rgb = np.array(points, copy=True)
            if not _config.DATA.center_at_origin:
                rgb, rgb_origin_offset = center_at_origin(rgb)
            rgb /= rgb.max(axis=0)  # bw [-1, 1]

        return points, rgb, labels, pose, other


def collate(data):
    data = [d for d in data if d is not None]
    coords, feats, labels, poses, others = list(
        zip(*data)
    )  # same size as getitem's return's

    coords_batch = torch.from_numpy(np.stack(coords)).to(dtype=torch.float32)
    feats_batch = torch.from_numpy(np.stack(feats)).to(dtype=torch.float32)
    labels_batch = torch.from_numpy(np.stack(labels)).long()
    poses_batch = torch.from_numpy(np.stack(poses)).to(dtype=torch.float32)

    start_offset = 0
    for i, o in enumerate(others):
        if not o.get('position'):
            others[i]["position"] = o["filename"].split("/")[-3]
        others[i]["filename"] = o["filename"].split("/")[-1]

        end_offset = start_offset + len(labels[i])
        others[i]["offset"] = (start_offset, end_offset)
        start_offset = end_offset

    return coords_batch, feats_batch, labels_batch, poses_batch, others
