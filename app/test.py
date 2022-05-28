import ipdb

import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MinkowskiEngine as ME


from utils import config, logger, preprocess, utils, metrics
from utils.visualization import (
    get_frame_from_pose,
    generate_colors,
    create_coordinate_frame,
    generate_key_point_shapes,
)

import data_engine
from inference_engine import InferenceEngine

_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class TestApp:
    def __init__(self, data_source=_config.TEST.data_source) -> None:
        self._data_source = data_engine.PickleDataEngine(data_source, cyclic=False)

        self._inference_engine = InferenceEngine()

        self.latest_results = dict()

    def clear_results(self):
        self.latest_results = {"segmentation": {"instances": dict(), "overall": dict()}}

    def run_tests(self):
        self.clear_results()

        with torch.no_grad():
            while True:
                data = self._data_source.get_raw()
                data_key = (
                    f"{data.other['position']}/{data.other['filepath'].split('/')[-1]}"
                )

                if data is None:
                    break

                rgb = preprocess.normalize_colors(data.rgb)  # never use data.rgb below

                seg_results = data.segmentation

                if _config.TEST.SEGMENTATION.evaluate:
                    seg_results = self._inference_engine.predict_segmentation(
                        data.points, data.rgb
                    )
                    segmentation_metrics = metrics.compute_segmentation_metrics(
                        data.segmentation,
                        seg_results,
                        classes=_config.INFERENCE.SEGMENTATION.classes,
                    )

                ee_idx = np.where(seg_results == 2)[0]
                ee_raw_points = data.points[ee_idx]  # no origin offset
                ee_raw_rgb = torch.from_numpy(rgb[ee_idx]).to(dtype=torch.float32)

                rot_result = self._inference_engine.predict_rotation(
                    ee_raw_points, ee_raw_rgb
                )

                pos_result, _ = self._inference_engine.predict_translation(
                    ee_raw_points, ee_raw_rgb, q=rot_result
                )

                nn_pose = np.concatenate((pos_result, rot_result))

                kp_coords, kp_classes, kp_probs = self._inference_engine.predict_key_points(
                    ee_raw_points, ee_raw_rgb
                )
                kp_pose = self._inference_engine.predict_pose_from_kp(
                    kp_coords, kp_classes
                )

                (
                    nn_dist_position,
                    nn_angle_diff,
                ) = metrics.compute_pose_metrics(data.pose, nn_pose)

                if kp_pose is not None:
                    (
                        kp_dist_position,
                        kp_angle_diff,
                    ) = metrics.compute_pose_metrics(data.pose, kp_pose)
                print('-------')
                # ipdb.set_trace()


if __name__ == "__main__":
    app = TestApp()
    app.run_tests()
    ipdb.set_trace()
