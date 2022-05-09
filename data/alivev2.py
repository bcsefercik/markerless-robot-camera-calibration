import json
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
from utils.data import get_ee_idx, get_roi_mask, get_ee_cross_section_idx
from utils.preprocess import center_at_origin
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line


_config = config.Config()
_logger = logger.Logger().get()


class AliveV2Dataset(Dataset):
    def __init__(self, set_name="train", augment=False, file_names=list(), quantization_enabled=True):
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
        self.ee_segmentation_enabled = _config()["DATA"].get('ee_segmentation_enabled', False)
        if self.ee_segmentation_enabled:
            self.ee_idx = dict()

        self.voting_enabled = _config()["DATA"].get("voting_enabled", False)
        if self.voting_enabled:
            self.ee_closest_points_idx = dict()

        self.test_split = _config.TEST.split
        self.test_workers = _config.TEST.workers
        self.batch_size = _config.TEST.batch_size

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.file_names = file_names
        self.load_file_names()

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

    def __getitem__(self, i):
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

        if self.ee_segmentation_enabled or _config.DATA.data_type == "ee_seg":
            if i not in self.ee_idx:
                self.ee_idx[i] = get_ee_idx(
                    points,
                    pose,
                    ee_dim={
                        'min_z': -0,
                        'max_z': 0.13,
                        'min_x': -0.04,
                        'max_x': 0.04,
                        'min_y': -0.13,
                        'max_y': 0.13
                    },  # leave big margin for bbox since we remove non arm points
                    switch_w=False)

                # remove ee idx which is not arm idx too
                ee_arm_match_idx = np.isin(self.ee_idx[i], arm_idx, assume_unique=True)
                self.ee_idx[i] = self.ee_idx[i][ee_arm_match_idx]

            labels[self.ee_idx[i]] = 2


        labels = np.reshape(labels, (-1, 1))
        pose = np.reshape(pose, (1, -1))
        # ipdb.set_trace()

        if _config.DATA.data_type == "gt_seg":
            points = points[arm_idx]
            rgb = rgb[arm_idx]
            labels = labels[arm_idx]
        elif _config.DATA.data_type == "gt_bbox":
            min_point = points[arm_idx].min(axis=0)
            max_point = points[arm_idx].max(axis=0)
            arm_bbox = (
                np.logical_and(points <= max_point, points >= min_point).sum(
                    axis=1
                )
                == 3
            )
            points = points[arm_bbox]
            rgb = rgb[arm_bbox]
            labels = labels[arm_bbox]
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

        if _config()["DATA"].get("voxelize_position", False):
            pose[0, :3] /= self.quantization_size

        if len(rgb) > 0:
            if rgb.min() < 0:
                # WRONG approach, tries to get rid of trouble from data prep code.
                rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
                rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
                rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

            if rgb.min() > (-1e-6) and rgb.max() < (1+1e-6):
                rgb -= 0.5
        # ipdb.set_trace()

        if self.voting_enabled:
            if i not in self.ee_closest_points_idx:
                closest_points_dists, self.ee_closest_points_idx[i] = get_ee_cross_section_idx(
                    points,  # ee points
                    pose[0],
                    count=32,
                    cutoff=0.004,
                    switch_w=False
                )  # switch_w=False in dataloader

            if _config.DATA.data_type == "ee_seg":
                labels *= 0

            labels[self.ee_closest_points_idx[i], :] = (1 if _config.DATA.data_type == "ee_seg" else 3)

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


def collate_non_quantized(data):
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

    # ipdb.set_trace()

    return coords_batch, feats_batch, labels_batch, poses_batch, others
