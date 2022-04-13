import os
import pickle
import glob

import ipdb
import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset
import MinkowskiEngine as ME

from utils import file_utils, logger, config


_config = config.Config()
_logger = logger.Logger().get()


class AliveV2Dataset(Dataset):
    def __init__(self, set_name="train", augment=False, file_names=list()):
        self.dataset = _config.DATA.folder
        self.dataset = os.path.join(self.dataset, set_name)

        self.filename_suffix = _config.DATA.suffix

        self.batch_size = _config.DATA.batch_size
        self.train_workers = _config.DATA.workers
        self.val_workers = _config.DATA.workers

        self.full_scale = _config.DATA.full_scale
        self.scale = _config.DATA.scale
        self.max_npoint = _config.DATA.max_npoint
        self.mode = _config.DATA.mode
        self.quantization_size = _config()["DATA"].get(
            "quantization_size", 1 / _config.DATA.scale
        )

        self.test_split = _config.TEST.split
        self.test_workers = _config.TEST.workers
        self.batch_size = _config.TEST.batch_size

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.file_names = file_names
        self.load_file_names()

    def __getitem__(self, i):
        # TODO: extract instance, semantic here
        (
            (xyz_origin, rgb, labels, instance_label, pose),
            semantic_pred,
            file_name,
        ) = self.load_data_file(i)

        xyz_origin = xyz_origin.astype(np.float32)
        rgb = rgb.astype(np.float32)
        labels = labels.astype(np.float32)
        arm_idx = labels == 1
        labels = np.reshape(labels, (-1, 1))
        pose = np.array(pose, dtype=np.float32)  # xyzw
        pose = np.insert(pose[:6], 3, pose[-1])  # wxyz
        pose = np.reshape(pose, (1, -1))
        other = {"filename": file_name}
        if isinstance(self.file_names[i], dict):
            other.update(self.file_names[i])

        if _config.DATA.data_type == "gt_seg":
            xyz_origin = xyz_origin[arm_idx]
            rgb = rgb[arm_idx]
            labels = labels[arm_idx]
        elif _config.DATA.data_type == "gt_bbox":
            min_point = xyz_origin[arm_idx].min(axis=0)
            max_point = xyz_origin[arm_idx].max(axis=0)
            arm_bbox = (
                np.logical_and(xyz_origin <= max_point, xyz_origin >= min_point).sum(
                    axis=1
                )
                == 3
            )
            xyz_origin = xyz_origin[arm_bbox]
            rgb = rgb[arm_bbox]
            labels = labels[arm_bbox]

        if rgb.min() < 0:
            # WRONG approach, tries to shit from data prep code.
            rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
            rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
            rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=xyz_origin,
            features=rgb,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100,
        )

        # ipdb.set_trace()

        return discrete_coords, unique_feats, unique_labels, pose, other

    def __len__(self):
        return len(self.file_names)

    def filter_file(self, file):
        filepath = file["filepath"] if isinstance(file, dict) else file
        filename = filepath.split("/")[-1]
        result = True

        result = result and filename[-16::] != "_semantic.pickle"
        result = result and "dark" not in filename

        if _config.DATA.prefix:
            result = result and filename.startswith(_config.DATA.prefix)

        if _config()["DATA"].get("arm_point_count_threshold", None):
            result = result and file['arm_point_count'] >= _config()["DATA"]["arm_point_count_threshold"]

        return result

    def load_file_names(self):
        if not self.file_names:
            self.file_names = glob.glob(
                os.path.join(self.dataset, "*" + self.filename_suffix)
            )
        self.file_names = [
            fn
            for fn in self.file_names
            if self.filter_file(fn)
        ]
        self.file_names.sort(key=lambda fn: fn["filepath"] if isinstance(fn, dict) else fn )

    def load_data_file(self, i, semantic_enabled=False):
        fn = self.file_names[i]
        curr_file_name = fn["filepath"] if isinstance(fn, dict) else fn
        x, semantic_pred = file_utils.load_alive_file(
            curr_file_name, semantic_enabled=semantic_enabled
        )

        return x, semantic_pred, curr_file_name


def collate(data):
    coords, feats, labels, poses, others = list(
        zip(*data)
    )  # same size as getitem's return's

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).to(dtype=torch.float32)
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).to(dtype=torch.int32)
    poses_batch = torch.from_numpy(np.concatenate(poses, 0)).to(dtype=torch.float32)

    for i, o in enumerate(others):
        if not o.get('position'):
            others[i]["position"] = o["filename"].split("/")[-3]
        others[i]["filename"] = o["filename"].split("/")[-1]

    # ipdb.set_trace()

    return coords_batch, feats_batch, labels_batch, poses_batch, others
