import time
import ipdb

import os
import sys
import abc
import json
from itertools import cycle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils

from dto import PointCloudDTO


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
    def __init__(self, data_path, split='test') -> None:
        self.data = {split: []}
        with open(data_path, 'r') as fp:
            self.data.update(json.load(fp))
        self.data[split].sort(key=lambda x: (x['position'], int(x['filepath'].split("/")[-1].split(".")[0])))
        self.data_pool = cycle(self.data[split])

    def get(self) -> PointCloudDTO:
        # time.sleep(0.1)
        data_ins = next(self.data_pool)
        # data_ins = self.data['test'][63]
        data, _ = file_utils.load_alive_file(data_ins['filepath'])

        if isinstance(data, dict):
            points = data['points']
            rgb = data['rgb']
        else:
            points, rgb, _, _, _ = data

        return PointCloudDTO(points=points, rgb=rgb, timestamp=datetime.utcnow())

    def run(self):
        return None

    def exit(self):
        return None

    def __len__(self):
        return len(self.data_pool)
