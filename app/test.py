import ipdb

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MinkowskiEngine as ME
import torch
from utils import config, logger, utils
from utils.visualization import (
    get_frame_from_pose,
    generate_colors,
    create_coordinate_frame,
    generate_key_point_shapes
)

import data_engine
from inference_engine import InferenceEngine

_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class TestApp:
    def __init__(self, data_source=_config.INFERENCE.data_source) -> None:
        if os.path.isfile(str(data_source)):
            self._data_source = data_engine.PickleDataEngine(data_source)
        else:
            import freenect_data_engine

            self._data_source = freenect_data_engine.FreenectDataEngine()

        self._inference_engine = InferenceEngine()


if __name__ == "__main__":
    app = TestApp()
    ipdb.set_trace()
