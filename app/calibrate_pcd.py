import typing
import ipdb

import os
import sys
import json
import pickle
import random
import statistics
from collections import defaultdict

import torch
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Side, Border, Font, PatternFill

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MinkowskiEngine as ME

from utils.transformation import get_base2cam_pose, transform_pose2pose
from utils import config, logger, preprocess, utils, metrics
from utils import calibration as calib_util
from utils.data import get_6_key_points

import data_engine
from inference_engine import InferenceEngine
from dto import TestResultDTO, PointCloudDTO, CalibrationResultDTO


_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class CalibrationApp:
    def __init__(self, data_source=_config.INFERENCE.CALIBRATION.pcd_source) -> None:
        self._data_source = data_engine.PCDDataEngine(data_source, cyclic=False, step=10)

        self._inference_engine = InferenceEngine()

        self.instance_results = None

        self.calibration: CalibrationResultDTO = None

        self.unit_multipliers = [1, 1]
        if _config.TEST.units[0] == "cm":
            self.unit_multipliers[0] = 100
        if _config.TEST.units[1] == "degree":
            self.unit_multipliers[1] = 57.2958

        self.clear_results()

        random.seed(_config.TEST.seed)
        np.random.seed(_config.TEST.seed)
        torch.manual_seed(_config.TEST.seed)

    def clear_results(self):
        self.instance_results = [[]]
        self.calibration = None

    def run_calibration(self):
        self.clear_results()

        # Make predictions
        with torch.no_grad():
            while True:
                data: PointCloudDTO = self._data_source.get()

                if data is None:
                    break

                result = self._inference_engine.predict(data)

                if len(self.instance_results[-1]) >= 10:
                    self.instance_results.append(list())

                self.instance_results[-1].append(result)
                print(data.id)

        self.calibration = self._inference_engine.calibrate({i: v for i, v in enumerate(self.instance_results)})
        print(self.calibration)
        # ipdb.set_trace()

if __name__ == "__main__":
    random.seed(_config.TEST.seed)
    np.random.seed(_config.TEST.seed)
    torch.manual_seed(_config.TEST.seed)

    app = CalibrationApp()
    app.run_calibration()
    # app.export_to_xslx()

    # ipdb.set_trace()
