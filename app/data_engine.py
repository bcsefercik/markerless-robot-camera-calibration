import ipdb

import os
import sys
import abc
import json
import time
from itertools import cycle
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils
from utils.data import get_ee_idx
from utils.transformation import switch_w

from dto import PointCloudDTO, RawDTO


class DataEngineInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_data_source")
            and callable(subclass.load_data_source)
            and hasattr(subclass, "extract_text")
            and callable(subclass.extract_text)
            or NotImplemented
        )

    @abc.abstractmethod
    def get(self) -> PointCloudDTO:
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> None:
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def exit(self) -> None:
        """Load in the data set"""
        raise NotImplementedError


class PickleDataEngine(DataEngineInterface):
    def __init__(self, data_path, split="test", cyclic=True) -> None:
        self.data = {split: []}
        with open(data_path, "r") as fp:
            self.data.update(json.load(fp))
        self.data[split].sort(
            key=lambda x: (
                x["position"],
                int(x["filepath"].split("/")[-1].split(".")[0]),
            )
        )
        self.data_pool = cycle(self.data[split]) if cyclic else iter(self.data[split])

    def get(self) -> PointCloudDTO:
        # time.sleep(0.1)
        data_ins = next(self.data_pool)
        data_ins = self.data['test'][52]
        data, _ = file_utils.load_alive_file(data_ins["filepath"])

        ee2base_pose = None
        gt_pose = None  # TODO: remove gt_pose
        if isinstance(data, dict):
            points = data["points"]
            rgb = data["rgb"]
            gt_pose = data["pose"]  # TODO: remove gt_pose
            ee2base_pose = data.get("robot2ee_pose")
        else:
            points, rgb, _, _, gt_pose = data  # TODO: remove gt_pose

        if gt_pose is not None:  # TODO: remove gt_pose
            gt_pose = switch_w(gt_pose)  # WXYZ

        if ee2base_pose is not None:
            ee2base_pose = switch_w(ee2base_pose)  # WXYZ

        return PointCloudDTO(
            points=points,
            rgb=rgb,
            ee2base_pose=ee2base_pose,
            timestamp=datetime.utcnow(),
            gt_pose=gt_pose  # TODO: remove this line
        )

    def get_raw(self) -> RawDTO:
        try:
            data_ins = next(self.data_pool)
        except StopIteration:
            return None
        data, _ = file_utils.load_alive_file(data_ins["filepath"])

        ee2base_pose = None
        if isinstance(data, dict):
            points = data["points"]
            rgb = data["rgb"]
            labels = data["labels"]
            pose = data["pose"]
            ee2base_pose = data.get("robot2ee_pose")
        else:
            points, rgb, labels, _, pose = data

        points = points.astype(np.float32)
        rgb = rgb.astype(np.float32)
        labels = labels.astype(np.int)
        pose = switch_w(pose)  # WXYZ

        if ee2base_pose is not None:
            ee2base_pose = switch_w(ee2base_pose)  # WXYZ

        other = {"filepath": data_ins["filepath"], "position": data_ins["position"]}

        arm_idx = np.where(labels == 1)[0]
        ee_idx = get_ee_idx(
            points,
            pose,
            ee_dim={
                "min_z": -0,
                "max_z": 0.13,
                "min_x": -0.04,
                "max_x": 0.04,
                "min_y": -0.13,
                "max_y": 0.13,
            },  # leave big margin for bbox since we remove non arm points
            arm_idx=arm_idx,
            switch_w=False,
        )

        labels[ee_idx] = 2

        return RawDTO(points, rgb, pose, labels, ee2base_pose=ee2base_pose, other=other)

    def run(self):
        return None

    def exit(self):
        return None

    def __len__(self):
        return len(self.data_pool)
