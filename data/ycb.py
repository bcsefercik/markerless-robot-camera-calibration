import os
import pickle
import glob

import ipdb
import torch
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import open3d as o3d

from utils import logger, config
from utils.data import normalize_color


_config = config.Config()
_logger = logger.Logger().get()


class YCBDataset(Dataset):
    def __init__(self, set_name="train", augment=False):
        self.dataset = _config.DATA.folder
        self.dataset = os.path.join(self.dataset, set_name)

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

        self.file_names = None
        self.load_file_names()

    def __getitem__(self, i):
        coords, colors, labels, file_name = self.load_data_file(i)

        other = {"filename": file_name}

        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=coords,
            features=colors,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100,
        )

        return discrete_coords, unique_feats, unique_labels, other

    def __len__(self):
        return len(self.file_names)

    def filter_filename(self, filepath):
        filename = filepath.split("/")[-1]
        result = True

        if _config.DATA.prefix:
            result = result and filename.startswith(_config.DATA.prefix)

        if _config.DATA.suffix:
            result = result and filename.endswith(_config.DATA.suffix)

        return result

    def load_file_names(self):
        self.file_names = glob.glob(os.path.join(self.dataset, "*"))

        self.file_names = [fn for fn in self.file_names if self.filter_filename(fn)]
        # self.file_names.sort()

    def load_data_file(self, i):
        curr_file_path = self.file_names[i]
        curr_file_name = curr_file_path.split("/")[-1]
        pcd = o3d.io.read_point_cloud(curr_file_path)
        coords = np.array(pcd.points)
        if not len(coords):
            print(curr_file_path,'CORT')
            os.remove(curr_file_path)
        colors = np.array(pcd.colors)
        labels = int(curr_file_name[:3]) * np.ones(
            (coords.shape[0], 1), dtype=np.int32
        )

        return coords, colors, labels, curr_file_path


def collate(data):
    coords, colors, labels, others = list(zip(*data))  # same size as getitem's return's

    coords_batch = ME.utils.batched_coordinates(coords)
    colors_batch = torch.from_numpy(np.concatenate(colors, 0)).to(dtype=torch.float32)
    colors_batch = normalize_color(colors_batch)
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).to(dtype=torch.int32)
    others = {"filenames": [o["filename"].split("/")[-1] for o in others]}

    return coords_batch, colors_batch, labels_batch, others
