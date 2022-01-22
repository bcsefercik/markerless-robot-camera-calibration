import os
import pickle
import glob

import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME

from utils import logger, config


_config = config.Config()
_logger = logger.Logger().get()


class AliveV1Dataset(Dataset):
    def __init__(self, set_name="train", quantization_size=0.02, augment=False):
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
        self.quantization_size = quantization_size

        self.test_split = _config.TEST.split
        self.test_workers = _config.TEST.workers
        self.batch_size = _config.TEST.batch_size

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.file_names = None
        self.load_file_names()

    def __getitem__(self, i):
        # TODO: extract instance, semantic here
        (xyz_origin, rgb, labels, instance_label, pose), semantic_pred, file_name = self.load_data_file(i)
        xyz_origin = xyz_origin.astype(np.float32)
        rgb = rgb.astype(np.float32)
        labels = labels.astype(np.float32)
        pose = np.array(pose, dtype=np.float32)

        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=xyz_origin,
            features=rgb,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100,
        )

        return discrete_coords, unique_feats, unique_labels, pose

    def __len__(self):
        return len(self.file_names)

    def filter_filename(self, filepath):
        filename = filepath.split("/")[-1]
        result = True

        result = result and filename[-16::] != "_semantic.pickle"
        result = result and "dark" not in filename

        if _config.DATA.prefix:
            result = result and filename.startswith(_config.DATA.prefix)

        return result

    def load_file_names(self):
        self.file_names = glob.glob(
            os.path.join(self.dataset, "*" + self.filename_suffix)
        )
        self.file_names = [fn for fn in self.file_names if self.filter_filename(fn)]
        self.file_names.sort()

    def load_data_file(self, i, semantic_enabled=False):
        x, semantic_pred = None, None

        curr_file_name = self.file_names[i]
        with open(curr_file_name, "rb") as filehandler:
            x = pickle.load(filehandler, encoding="bytes")

        if semantic_enabled:
            with open(
                curr_file_name.replace(".pickle", "_semantic.pickle"), "rb"
            ) as fp:
                semantic_pred = pickle.load(fp, encoding="bytes")

        return x, semantic_pred, curr_file_name
