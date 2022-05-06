import ipdb

import os
import sys
import abc
import json
from itertools import cycle
from dataclasses import dataclass
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils


@dataclass
class PointCloutDTO:
    points: np.array
    rgb: np.array
    timestamp: datetime = datetime.utcnow()
    joint_angles: np.array = None


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
    def get_frame(self) -> PointCloutDTO:
        """Load in the data set"""
        raise NotImplementedError


class PickleDataEngine(DataEngineInterface):
    def __init__(self, data_path, split='test') -> None:
        self.data = {split: []}
        with open(data_path, 'r') as fp:
            self.data.update(json.load(fp))
        self.data[split].sort(key=lambda x: (x['position'], int(x['filepath'].split("/")[-1].split(".")[0])))
        self.data_pool = cycle(self.data[split])

    def get_frame(self) -> PointCloutDTO:
        data_ins = next(self.data_pool)
        data, _ = file_utils.load_alive_file(data_ins['filepath'])

        if isinstance(data, dict):
            points = data['points']
            rgb = data['rgb']
        else:
            points, rgb, _, _, _ = data

        return PointCloutDTO(points=points, rgb=rgb)

    def __len__(self):
        return len(self.data_pool)


class FreenectDataEngine(DataEngineInterface):
    def __init__(self) -> None:
        pass

    def get_frame(self) -> PointCloutDTO:
        pass
