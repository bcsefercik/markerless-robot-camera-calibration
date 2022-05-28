import ipdb

import os
import sys
import json
from collections import defaultdict

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MinkowskiEngine as ME


from utils import config, logger, preprocess, utils, metrics
from utils.data import get_6_key_points

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

        self.instance_results = defaultdict(dict)
        self.clear_results()

    def clear_results(self):
        self.instance_results = defaultdict(dict)

    def run_tests(self):
        self.clear_results()

        with torch.no_grad():
            while True:
                data = self._data_source.get_raw()
                data_key = (
                    f"{data.other['position']}/{data.other['filepath'].split('/')[-1]}"
                )
                self.instance_results[data_key]['position'] = data.other['position']

                if data is None:
                    break

                rgb = preprocess.normalize_colors(data.rgb)  # never use data.rgb below

                seg_results = data.segmentation

                if _config.TEST.SEGMENTATION.evaluate:
                    seg_results = self._inference_engine.predict_segmentation(
                        data.points, rgb
                    )
                    segmentation_metrics = metrics.compute_segmentation_metrics(
                        data.segmentation,
                        seg_results,
                        classes=_config.INFERENCE.SEGMENTATION.classes,
                    )
                    self.instance_results[data_key]['segmentation'] = segmentation_metrics

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

                nn_pose_metrics = metrics.compute_pose_metrics(data.pose, nn_pose)

                nn_dist_position = nn_pose_metrics['dist_position']
                nn_angle_diff = nn_pose_metrics['angle_diff']

                self.instance_results[data_key]['dist_position'] = {'nn': nn_dist_position}
                self.instance_results[data_key]['angle_diff'] = {'nn': nn_angle_diff}

                kp_gt_coords, kp_gt_idx = get_6_key_points(ee_raw_points, data.pose, switch_w=False)
                kp_coords, kp_classes, kp_confs = self._inference_engine.predict_key_points(
                    ee_raw_points, ee_raw_rgb,
                )
                mean_kp_error = metrics.compute_kp_error(kp_gt_coords, kp_coords, kp_classes)
                self.instance_results[data_key]['mean_kp_error'] = mean_kp_error

                if len(kp_classes) > 3:
                    kp_pose = self._inference_engine.predict_pose_from_kp(
                        kp_coords, kp_classes
                    )
                    kp_pose_metrics = metrics.compute_pose_metrics(data.pose, kp_pose)

                    kp_dist_position = kp_pose_metrics['dist_position']
                    kp_angle_diff = kp_pose_metrics['angle_diff']

                    self.instance_results[data_key]['dist_position']['kp'] = kp_dist_position
                    self.instance_results[data_key]['angle_diff']['kp'] = kp_angle_diff

                print('---------')

                ipdb.set_trace()


if __name__ == "__main__":
    app = TestApp()
    app.run_tests()
    ipdb.set_trace()
