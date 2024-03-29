import json
import os
import glob
import time
import ipdb
from datetime import timedelta


import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from tqdm import tqdm, trange

from utils import file_utils, logger, config
from utils.data import get_ee_idx, get_roi_mask, get_ee_cross_section_idx, get_key_points, get_6_key_points, collect_closest_points
from utils.preprocess import center_at_origin
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line
from utils import augmentation as aug

_config = config.Config()
_logger = logger.Logger().get()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class AliveV2Dataset(Dataset):
    def __init__(self, set_name="train", augment=False, file_names=list(), quantization_enabled=True):
        self.augment = augment or (set_name=='train' and len(_config.DATA.augmentation) > 0)

        self.dataset = _config()["DATA"].get("folder", "")
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
        self.quantization_enabled = quantization_enabled

        self.test_split = _config.TEST.split
        self.test_workers = _config.TEST.workers
        self.batch_size = _config.TEST.batch_size

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.file_names = file_names
        self.load_file_names()

        self.ee_idx = [None] * len(self.file_names)

        self.key_points = [None] * len(self.file_names)
        if _config.DATA.keypoints_enabled:
            self.key_points_generator = get_6_key_points if _config.DATA.num_of_keypoints == 6 else get_key_points

        self.voting_enabled = _config()["DATA"].get("voting_enabled", False)
        self.ee_closest_points_idx = [None] * len(self.file_names)

        self.roi = None

        if _config()['DATA'].get('roi') is not None:
            self.roi = dict()
            roi_files = _config()['DATA']['roi']
            for rf in roi_files:
                with open(rf, 'r') as fp:
                    self.roi.update(json.load(fp))

            for k, v in self.roi.items():
                for kk in v:
                    if kk.startswith('max'):
                        self.roi[k][kk] += _config()['DATA'].get('roi_offset', 0)
                    else:
                        self.roi[k][kk] -= _config()['DATA'].get('roi_offset', 0)

        self.sample_idx_memo = [None] * len(self.file_names)

        self.file_idx_to_skip = set()

        # self.file_names = self.file_names[:128]

        # loadd caches for fast fetch, can't do in get item due to multiprocessing
        if _config.DATA.load_cache_at_start:
            _logger.info(f"Loading dataset caches ({set_name})")
            s = time.time()
            for i in trange(len(self.file_names)):
                self.__getitem__(i)
            _logger.info(f"Successfully loaded caches in {timedelta(seconds=time.time() - s)} ({set_name})")

            self.file_names = [v for i, v in enumerate(self.file_names) if i not in self.file_idx_to_skip]
            self.sample_idx_memo = [v for i, v in enumerate(self.sample_idx_memo) if i not in self.file_idx_to_skip]
            self.ee_idx = [v for i, v in enumerate(self.ee_idx) if i not in self.file_idx_to_skip]
            self.key_points = [v for i, v in enumerate(self.key_points) if i not in self.file_idx_to_skip]
            self.ee_closest_points_idx = [v for i, v in enumerate(self.ee_closest_points_idx) if i not in self.file_idx_to_skip]

    def load_generic_data(self, i):
        # TODO: extract instance, semantic here
        x, semantic_pred, file_name = self.load_data_file(i)
        joint_angles = None
        if isinstance(x, dict):
            points = x['points']
            rgb = x['rgb']
            labels = x['labels']
            instance_labels = x['instance_labels']
            pose = x['pose']
            joint_angles = x['joint_angles']
        else:
            points, rgb, labels, instance_labels, pose = x

        points = points.astype(np.float32)
        rgb = rgb.astype(np.float32)
        labels = labels.astype(np.float32)
        pose = np.array(pose, dtype=np.float32)  # xyzw
        pose = np.insert(pose[:6], 3, pose[-1])  # WXYZ

        other = {
            "filename": file_name,
            "joint_angles": joint_angles
        }
        if isinstance(self.file_names[i], dict):
            other.update(self.file_names[i])

        arm_idx = np.where(labels == 1)[0]

        if _config.DATA.ee_segmentation_enabled or _config.DATA.data_type == "ee_seg":
            if self.ee_idx[i] is None:
                if not (labels == 2).any():
                    self.ee_idx[i] = get_ee_idx(
                        points,
                        pose,
                        ee_dim={
                            'min_z': -0,
                            'max_z': 0.13,
                            'min_x': -0.05,
                            'max_x': 0.05,
                            'min_y': -0.14,
                            'max_y': 0.14
                        },  # leave big margin for bbox since we remove non arm points
                        arm_idx=arm_idx,
                        switch_w=False)
                else:
                    self.ee_idx[i] = np.where(labels == 2)[0]

            labels[self.ee_idx[i]] = 2

        labels = np.reshape(labels, (-1, 1))
        pose = np.reshape(pose, (1, -1))

        if _config.DATA.data_type == "gt_seg":
            points = points[arm_idx]
            rgb = rgb[arm_idx]
            labels = labels[arm_idx]
        elif _config.DATA.data_type == "ee_seg":
            if len(self.ee_idx[i]) < 1:
                return None

            points = points[self.ee_idx[i]]
            rgb = rgb[self.ee_idx[i]]
            labels = labels[self.ee_idx[i]]

        if self.roi is not None:
            instance_roi = get_roi_mask(
                points,
                **self.roi[other['position']]
            )
            points = points[instance_roi]
            rgb = rgb[instance_roi]
            labels = labels[instance_roi]

        if len(rgb) > 0:
            if rgb.min() < 0:
                # WRONG approach, tries to get rid of trouble from data prep code.
                rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
                rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
                rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

            if rgb.min() > (-1e-6) and rgb.max() < (1+1e-6):
                rgb -= 0.5

        return points, rgb, labels, instance_labels, pose, joint_angles, other

    def conduct_post_point_ops(self, points, pose, other):
        if  _config.DATA.data_type == "ee_seg" and _config.DATA.move_ee_to_origin:
            rot_mat = get_quaternion_rotation_matrix(pose[0, 3:], switch_w=False)  # switch_w=False in dataloader
            points = (rot_mat.T @ np.concatenate((points, pose[0, :3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
            pose[0, :3] = np.array(points[-1], copy=True)
            points = points[:-1]

        if _config.DATA.center_at_origin:
            points, origin_offset = center_at_origin(points)
            pose[:, :3] -= origin_offset
            other['origin_offset'] = origin_offset

        elif _config.DATA.base_at_origin:
            origin_base_offset = points.min(axis=0)
            points -= origin_base_offset
            pose[:, :3] -= origin_base_offset
            other['origin_base_offset'] = origin_base_offset

        return points, pose, other

    def load_key_points(self, i, points, pose, labels, p2p_label=True):
        labels *= 0
        labels += _config.DATA.ignore_label

        if self.key_points[i] is None:
            key_points, kp_idx = self.key_points_generator(
                points,
                pose[0],
                ignore_label=_config.DATA.ignore_label,
                switch_w=False  # switch_w=False in dataloader
            )

            if not p2p_label:
                return kp_idx

            kp_real = kp_idx > -1
            kp_classes_real = np.arange(len(kp_idx), dtype=np.int)[kp_real]
            kp_idx_real = kp_idx[kp_real]
            pcls_idx, kp_idx = collect_closest_points(kp_idx_real, points)
            kp_classes = kp_classes_real[pcls_idx]

            self.key_points[i] = (kp_classes, kp_idx)

        kp_classes, kp_idx = self.key_points[i]
        labels[kp_idx] = kp_classes.reshape(-1, 1)

        return labels

    def __getitem__(self, i):
        data = self.load_generic_data(i)
        if data is None:
            self.file_idx_to_skip.add(i)
            return None

        points, rgb, labels, instance_labels, pose, joint_angles, other = data

        if _config()["DATA"].get("voxelize_position", False):
            pose[0, :3] /= self.quantization_size
        # ipdb.set_trace()

        if self.voting_enabled:
            if _config.DATA.keypoints_enabled:
                raise AttributeError("Voting and keypoint cannot be simultaneously enabled.")

            if self.ee_closest_points_idx[i] is None:
                _, self.ee_closest_points_idx[i] = get_ee_cross_section_idx(
                    points,  # ee points
                    pose[0],
                    count=32,
                    cutoff=0.004,
                    switch_w=False
                )  # switch_w=False in dataloader

            if _config.DATA.data_type == "ee_seg":
                labels *= 0

            labels[self.ee_closest_points_idx[i], :] = (1 if _config.DATA.data_type == "ee_seg" else 3)

        if _config.DATA.keypoints_enabled:
            labels = self.load_key_points(i, points, pose, labels)

        if self.augment:  # TODO: add augmentation for pose
            points = aug.augment_segmentation(
                points,
                scale=_config.DATA.scale,
                probability=_config.DATA.augmentation_probability,
                **{k: True for k in _config.DATA.augmentation}
            )

        points, pose, other = self.conduct_post_point_ops(points, pose, other)

        if _config.DATA.use_coordinates_as_features:
            rgb = np.array(points, copy=True)
            if not _config.DATA.center_at_origin:
                rgb, rgb_origin_offset = center_at_origin(rgb)
            rgb /= rgb.max(axis=0)  # bw [-1, 1]

        if self.quantization_enabled:
            discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
                coordinates=points,
                features=rgb,
                labels=labels,
                quantization_size=self.quantization_size,
                ignore_label=_config.DATA.ignore_label,
            )
        else:
            discrete_coords, unique_feats, unique_labels = points, rgb, labels
            # discrete_coords /= self.quantization_size

        return discrete_coords, unique_feats, unique_labels, pose, other

    def __len__(self):
        return len(self.file_names)

    def filter_file(file):
        filepath = file["filepath"] if isinstance(file, dict) else file
        filename = filepath.split("/")[-1]

        result = True

        result = result and (not filename.endswith("_semantic.pickle"))
        result = result and (not filename.endswith("_eemask.pickle"))
        result = result and "dark" not in filename

        if _config.DATA.prefix:
            result = result and filename.startswith(_config.DATA.prefix)

        if _config().get("DATA", dict()).get("position_eligibility_enabled"):
            result = result and file.get('position_eligibility', False)

        if _config().get("DATA", dict()).get("orientation_eligibility_enabled"):
            result = result and file.get('orientation_eligibility', False)

        if _config().get("DATA", dict()).get("arm_point_count_threshold"):
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
            if AliveV2Dataset.filter_file(fn)
        ]
        self.file_names.sort(key=lambda fn: fn["filepath"] if isinstance(fn, dict) else fn )

    def load_data_file(self, i, semantic_enabled=False):
        fn = self.file_names[i]
        curr_file_name = fn["filepath"] if isinstance(fn, dict) else fn
        if not os.path.isabs(curr_file_name):
            curr_file_name = os.path.join(
                os.path.dirname(BASE_PATH),
                curr_file_name
            )

        x, semantic_pred = file_utils.load_alive_file(
            curr_file_name, semantic_enabled=semantic_enabled
        )

        return x, semantic_pred, curr_file_name


def collate(data):
    data = [d for d in data if d is not None]
    coords, feats, labels, poses, others = list(
        zip(*data)
    )  # same size as getitem's return's
    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).to(dtype=torch.float32)
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).long()
    poses_batch = torch.from_numpy(np.concatenate(poses, 0)).to(dtype=torch.float32)

    start_offset = 0
    for i, o in enumerate(others):
        if not o.get('position'):
            others[i]["position"] = o["filename"].split("/")[-3]
        others[i]["filename"] = o["filename"].split("/")[-1]

        end_offset = start_offset + len(labels[i])
        others[i]["offset"] = (start_offset, end_offset)
        start_offset = end_offset

        if _config.STRUCTURE.use_joint_angles:
            others[i]['joint_angles'] = torch.from_numpy(
                others[i]['joint_angles'].reshape((1, -1))
            ).to(dtype=torch.float32)

    return coords_batch, feats_batch, labels_batch, poses_batch, others


def collate_sparse(data):
    data = [d for d in data if d is not None]
    coords, feats, labels, poses, others = list(
        zip(*data)
    )  # same size as getitem's return's
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(
        coords,
        feats,
        labels,
        dtype=torch.float32,
    )
    poses_batch = torch.from_numpy(np.concatenate(poses, 0)).to(dtype=torch.float32)

    start_offset = 0
    for i, o in enumerate(others):
        if not o.get('position'):
            others[i]["position"] = o["filename"].split("/")[-3]
        others[i]["filename"] = o["filename"].split("/")[-1]

        end_offset = start_offset + len(labels[i])
        others[i]["offset"] = (start_offset, end_offset)
        start_offset = end_offset

        if _config.STRUCTURE.use_joint_angles:
            others[i]['joint_angles'] = torch.from_numpy(
                others[i]['joint_angles'].reshape((1, -1))
            ).to(dtype=torch.float32)

    return coords_batch, feats_batch, labels_batch, poses_batch, others


def collate_tupled(data):
    data = [d for d in data if d is not None]
    coords, feats, labels, poses, others = list(
        zip(*data)
    )  # same size as getitem's return's

    coords_batch = torch.from_numpy(np.concatenate(coords, 0)).to(dtype=torch.float32)
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).to(dtype=torch.float32)
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).long()
    poses_batch = torch.from_numpy(np.concatenate(poses, 0)).to(dtype=torch.float32)

    start_offset = 0
    for i, o in enumerate(others):
        if not o.get('position'):
            others[i]["position"] = o["filename"].split("/")[-3]
        others[i]["filename"] = o["filename"].split("/")[-1]

        end_offset = start_offset + len(labels[i])
        others[i]["offset"] = (start_offset, end_offset)
        start_offset = end_offset

    return coords_batch, feats_batch, labels_batch, poses_batch, others
