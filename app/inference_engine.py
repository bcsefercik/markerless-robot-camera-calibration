from collections import defaultdict
import concurrent.futures
import json
import time
import ipdb

import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.alivev2 import AliveV2Dataset, collate_non_quantized
from utils import config, logger, utils, preprocess, output
from model.backbone import minkunet
from model.robotnet_vote import RobotNetVote
from model.robotnet_encode import RobotNetEncode
from model.robotnet import RobotNet
from dto import ResultDTO, PointCloudDTO

import MinkowskiEngine as ME


_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class InferenceEngine:
    def __init__(self) -> None:
        self.cluster_util = output.ClusterUtil()

        self.models = defaultdict(lambda: minkunet.MinkUNet18D)
        self.models["minkunet101"] = minkunet.MinkUNet101
        self.models["minkunet34C"] = minkunet.MinkUNet34C
        self.models["minkunet14A"] = minkunet.MinkUNet14A
        self.models["minkunet"] = minkunet.MinkUNet18D
        self.models["translation"] = RobotNetVote
        self.models["robotnet_encode"] = RobotNetEncode
        self.models["robotnet"] = RobotNet

        self._segmentation_model = self.models[_config.INFERENCE.SEGMENTATION.backbone](
            in_channels=_config.DATA.input_channel,
            out_channels=len(_config.INFERENCE.SEGMENTATION.classes),
        )
        utils.checkpoint_restore(
            self._segmentation_model,
            f=_config.INFERENCE.SEGMENTATION.checkpoint,
            use_cuda=_use_cuda,
        )
        self._segmentation_model.eval()

        self._translation_model = self.models["translation"](
            in_channels=_config.DATA.input_channel,
            num_classes=2,  # binary; ee point or not
        )
        utils.checkpoint_restore(
            self._translation_model,
            f=_config.INFERENCE.TRANSLATION.checkpoint,
            use_cuda=_use_cuda,
        )
        self._translation_model.eval()

        compute_confidence = _config()["STRUCTURE"].get("compute_confidence", False)
        self._rotation_model = self.models[
            f'robotnet{"_encode" if _config.INFERENCE.ROTATION.encode_only else ""}'
        ](
            in_channels=_config.DATA.input_channel,
            out_channels=(10 if compute_confidence else 7),
        )
        utils.checkpoint_restore(
            self._rotation_model,
            f=_config.INFERENCE.ROTATION.checkpoint,
            use_cuda=_use_cuda,
        )
        self._rotation_model.eval()

    def predict(self, data: PointCloudDTO):
        with torch.no_grad():
            rgb = preprocess.normalize_colors(data.rgb)  # never use data.rgb below

            if _config.INFERENCE.SEGMENTATION.center_at_origin:
                seg_points, seg_origin_offset = preprocess.center_at_origin(data.points)
            else:
                seg_points = data.points
                seg_origin_offset = np.array([0.0, 0.0, 0.0])

            seg_rgb = torch.from_numpy(rgb).to(dtype=torch.float32)
            seg_points = torch.from_numpy(seg_points).to(dtype=torch.float32)

            seg_in_field = ME.TensorField(
                features=seg_rgb,
                coordinates=ME.utils.batched_coordinates(
                    [seg_points * _config.INFERENCE.SEGMENTATION.scale], dtype=torch.float32
                ),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=_device,
            )
            seg_input = seg_in_field.sparse()
            seg_output = self._segmentation_model(seg_input)
            seg_out_field = seg_output.slice(seg_in_field)

            seg_results, seg_conf = output.get_segmentations_from_tensor_field(seg_out_field)
            ee_mask = seg_results == 2
            ee_idx = np.where(seg_results == 2)[0]
            seg_results[ee_idx] = 1  # initially, set all ee pred to arm

            ee_idx_inside = self.cluster_util.get_largest_cluster(seg_points[ee_mask])
            seg_results[ee_idx[ee_idx_inside]] = 2  # set ee classes within largest linkage cluster
            result_dto = ResultDTO(segmentation=seg_results)

            ee_raw_points = data.points[ee_idx[ee_idx_inside]]  # no origin offset
            ee_raw_rgb = rgb[ee_idx[ee_idx_inside]]
            ee_rgb = torch.from_numpy(ee_raw_rgb).to(dtype=torch.float32)

            # TODO: run rot and trans in parallel!

            # Rotation estimation
            rot_result = self.predict_rotation(ee_raw_points, ee_rgb)
            result_dto.ee_pose[3:] = rot_result

            # Translation estimation
            pos_model_out, _ = self.predict_translation(ee_raw_points, ee_rgb)
            pos_result = output.get_pred_center(pos_model_out, ee_raw_points, q=rot_result)
            result_dto.ee_pose[:3] = pos_result

            return result_dto

    def predict_rotation(self, ee_raw_points, ee_rgb):
        with torch.no_grad():
            if _config.INFERENCE.ROTATION.center_at_origin:
                ee_rot_points, _ = preprocess.center_at_origin(ee_raw_points)
            else:
                ee_rot_points = ee_raw_points

            rot_points = torch.from_numpy(ee_rot_points).to(dtype=torch.float32)

            rot_input = ME.TensorField(
                features=ee_rgb,
                coordinates=ME.utils.batched_coordinates(
                    [rot_points * _config.INFERENCE.ROTATION.scale], dtype=torch.float32
                ),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=_device,
            ).sparse()
            rot_output = self._rotation_model(rot_input)

            return rot_output[0][3:].cpu().numpy()

    def predict_translation(self, ee_raw_points, ee_rgb):
        with torch.no_grad():
            if _config.INFERENCE.TRANSLATION.center_at_origin:
                ee_pos_points, pos_origin_offset = preprocess.center_at_origin(ee_raw_points)
            else:
                ee_pos_points = ee_raw_points
                pos_origin_offset = np.array([0.0, 0.0, 0.0])

            pos_points = torch.from_numpy(ee_pos_points).to(dtype=torch.float32)

            pos_in_field = ME.TensorField(
                features=ee_rgb,
                coordinates=ME.utils.batched_coordinates(
                    [pos_points * _config.INFERENCE.TRANSLATION.scale], dtype=torch.float32
                ),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=_device,
            )
            pos_input = pos_in_field.sparse()
            pos_output = self._translation_model(pos_input)
            pos_out_field = pos_output.slice(pos_in_field)

        return pos_out_field.features, pos_origin_offset
