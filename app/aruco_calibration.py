import os
import statistics
import sys
import ipdb
from collections import defaultdict

import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config, logger, aruco, utils, metrics
from utils.transformation import get_base2cam_pose, transform_pose2pose
from utils.data import get_ee_idx

import data_engine
from inference_engine import InferenceEngine
from dto import TestResultDTO, RawDTO, CalibrationResultDTO
from app.test import TestApp

_config = config.Config()
_logger = logger.Logger().get()


class ArucoTestApp(TestApp):
    def run_tests(self):
        self.clear_results()

        while True:
            data: RawDTO = self._data_source.get_raw()

            if data is None:
                break

            data_key = (
                f"{data.other['position']}/{data.other['filepath'].split('/')[-1]}"
            )

            ee_pose = aruco.compute_ee_pose(data.points, data.rgb, t_tag2ee=_config.TEST.t_tag2ee)
            result_dto = TestResultDTO(
                segmentation=None,
                ee_pose=ee_pose,
                key_points_pose=ee_pose,
                is_confident=False
            )

            if ee_pose is not None:
                if _config.INFERENCE.icp_enabled:
                    ee_idx = get_ee_idx(
                        data.points,
                        ee_pose,
                        switch_w=False,
                        ee_dim={
                            'min_z': -0.02,
                            'max_z': 0.07,
                            'min_x': -0.05,
                            'max_x': 0.05,
                            'min_y': -0.15,
                            'max_y': 0.15
                        }
                    )

                    ee_raw_points = data.points[ee_idx]

                    result_dto.ee_pose = self._inference_engine.match_icp(ee_raw_points, ee_pose)

                result_dto.is_confident = True
                self.instance_results[data_key]['position'] = data.other['position']
                nn_pose_metrics = metrics.compute_pose_metrics(data.pose, result_dto.ee_pose)
                self.instance_results[data_key]['dist_position'] = {'nn': nn_pose_metrics['dist_position']}
                self.instance_results[data_key]['angle_diff'] = {'nn': nn_pose_metrics['angle_diff']}

                base_pose = get_base2cam_pose(result_dto.ee_pose, data.ee2base_pose)
                result_dto.base_pose = base_pose
                result_dto.key_points_base_pose = base_pose

                if self._inference_engine.camera_link_transformation_pose is not None:
                    base_pose = transform_pose2pose(result_dto.base_pose, self._inference_engine.camera_link_transformation_pose)
                    result_dto.base_pose_camera_link = base_pose

                base_pose_metrics = metrics.compute_pose_metrics(
                    self._gt_base_to_cam_pose,
                    base_pose
                )
                self.instance_results[data_key]['base2cam'] = {
                    'dist_position': base_pose_metrics['dist_position'],
                    'angle_diff': base_pose_metrics['angle_diff']
                }

                self.predictions[data.other['position']].append(result_dto)

            _logger.info(f'{data_key}{"" if result_dto.is_confident else ", ignored"}')

        print("Calibration Results:")
        self.calibration = self._inference_engine.calibrate(self.predictions)

        # print(self.calibration)

        position_results_raw = defaultdict(list)
        for ir in self.instance_results.values():
            position_results_raw[ir['position']].append(ir)

        for pos, irs in position_results_raw.items():
            self.position_results[pos]['base2cam_dist_position'] = [ir['base2cam']['dist_position'] for ir in irs]
            self.position_results[pos]['base2cam_angle_diff'] = [ir['base2cam']['angle_diff'] for ir in irs]
            self.position_results[pos]['angle_diff_nn'] = [ir['angle_diff']['nn'] for ir in irs]
            self.position_results[pos]['dist_position_nn'] = [ir['dist_position']['nn'] for ir in irs]

        for prs in self.position_results.values():
            for k in prs:
                if len(prs[k]) > 0:
                    self.overall_results[k].append(statistics.mean(prs[k]))

        self.overall_results["calibration_angle_diff"] = -100
        self.overall_results["calibration_dist_position"] = -100
        if self.calibration.pose_camera_link is not None:
            calibration_metrics = metrics.compute_pose_metrics(self.calibration.pose_camera_link, self._gt_base_to_cam_pose)
            self.overall_results["calibration_angle_diff"] = calibration_metrics["angle_diff"]
            self.overall_results["calibration_dist_position"] = calibration_metrics["dist_position"]

        print("Translation error:", self.overall_results["calibration_dist_position"] * self.unit_multipliers[0])
        print("Rotation error:", self.overall_results["calibration_angle_diff"] * self.unit_multipliers[1])

        # print(self.instance_results)
        # self.export_to_xslx()


if __name__ == "__main__":
    np.random.seed(_config.TEST.seed)

    app = ArucoTestApp(calibration_only=True)
    app.run_tests()
    # app.export_to_xslx()

    # ipdb.set_trace()
