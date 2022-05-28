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

from data.alivev2 import AliveV2Dataset, collate_tupled
from utils import config, logger, utils, preprocess
from utils import output as out_utils
from utils.transformation import get_q_from_matrix, get_quaternion_rotation_matrix, get_rigid_transform_3D
from utils.data import get_farthest_point_sample_idx
from model.backbone import minkunet
from model.robotnet_vote import RobotNetVote
from model.robotnet_segmentation import RobotNetSegmentation
from model.robotnet_encode import RobotNetEncode
from model.robotnet import RobotNet
from dto import ResultDTO, PointCloudDTO
from model.pointnet2 import PointNet2SSG

import MinkowskiEngine as ME


_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class InferenceEngine:
    def __init__(self) -> None:
        self.cluster_util = out_utils.ClusterUtil()

        self.models = defaultdict(lambda: minkunet.MinkUNet18D)
        self.models["minkunet101"] = minkunet.MinkUNet101
        self.models["minkunet34C"] = minkunet.MinkUNet34C
        self.models["minkunet14A"] = minkunet.MinkUNet14A
        self.models["minkunet"] = minkunet.MinkUNet18D
        self.models["translation"] = RobotNetVote
        self.models["robotnet_encode"] = RobotNetEncode
        self.models["robotnet"] = RobotNet
        self.models["robotnet_segmentation"] = RobotNetSegmentation
        self.models["pointnet2"] = PointNet2SSG

        # self._segmentation_model = self.models[_config.INFERENCE.SEGMENTATION.backbone](
        #     in_channels=_config.DATA.input_channel,
        #     out_channels=len(_config.INFERENCE.SEGMENTATION.classes),
        # )
        self._segmentation_model = self.models[_config.INFERENCE.SEGMENTATION.backbone](
            in_channels=_config.DATA.input_channel,
            num_classes=len(_config.INFERENCE.SEGMENTATION.classes),
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

        if _config.INFERENCE.KEY_POINTS.backbone == "pointnet2":
            in_channels = (
                6 if _config.INFERENCE.KEY_POINTS.use_coordinates_as_features else 9
            )
            self._key_points_model = self.models[_config.INFERENCE.KEY_POINTS.backbone](
                in_channels=in_channels,
                num_classes=_config.INFERENCE.KEY_POINTS.num_of_keypoints,  # num of key points
            )
        else:
            self._key_points_model = self.models[_config.INFERENCE.KEY_POINTS.backbone](
                in_channels=_config.DATA.input_channel,
                num_classes=10,  # num of key points, TODO: get from _config.INFERENCE.KEY_POINTS.backbone
            )
        utils.checkpoint_restore(
            self._key_points_model,
            f=_config.INFERENCE.KEY_POINTS.checkpoint,
            use_cuda=_use_cuda,
        )
        self._key_points_model.eval()

        self.reference_key_points = np.array([
            [ 0.01982731,  0.08085986,  0.00321919],
            [ 0.02171595, -0.08986182,  0.00388430],
            [ 0.01288678,  0.09103118,  0.06127814],
            [ 0.02079032, -0.09790908,  0.05609143],
            [-0.00185802,  0.04654205,  0.11564558],
            [ 0.00241113, -0.04262756,  0.11564558]
       ])

    def check_sanity(self, data: PointCloudDTO, result: ResultDTO):
        num_of_ee_points = (result.segmentation == 2).sum()
        if num_of_ee_points < _config.INFERENCE.SANITY.min_num_of_ee_points:
            print("fail min # points")
            return False


        # ipdb.set_trace()
        return True

    def predict(self, data: PointCloudDTO):
        with torch.no_grad():
            rgb = preprocess.normalize_colors(data.rgb)  # never use data.rgb below

            seg_results = self.predict_segmentation(data.points, rgb)

            result_dto = ResultDTO(segmentation=seg_results)

            ee_idx = np.where(seg_results == 2)[0]

            ee_raw_points = data.points[ee_idx]  # no origin offset
            ee_raw_rgb = torch.from_numpy(rgb[ee_idx]).to(dtype=torch.float32)

            # TODO: run rot and trans in parallel!

            # Rotation estimation
            rot_result = self.predict_rotation(ee_raw_points, ee_raw_rgb)
            result_dto.ee_pose[3:] = rot_result

            # Translation estimation
            pos_result, _ = self.predict_translation(
                ee_raw_points, ee_raw_rgb, q=rot_result
            )
            result_dto.ee_pose[:3] = pos_result

            # Key Points estimation
            kp_coords, kp_classes = self.predict_key_points(ee_raw_points, ee_raw_rgb)
            result_dto.key_points = list(zip(kp_classes, kp_coords))

            result_dto.key_points_pose = self.predict_pose_from_kp(kp_coords, kp_classes)

            result_dto.is_confident = self.check_sanity(data, result_dto)

            return result_dto

    def predict_pose_from_kp(self, kp_coords, kp_classes):
        if len(kp_classes) < 4:
            return None

        kp_rot_mat, kp_translation = get_rigid_transform_3D(
            self.reference_key_points[kp_classes],
            kp_coords
        )
        kp_q = get_q_from_matrix(kp_rot_mat)

        return np.concatenate((kp_translation, kp_q))

    def predict_segmentation(self, points, rgb):
        if _config.INFERENCE.SEGMENTATION.center_at_origin:
            seg_points, seg_origin_offset = preprocess.center_at_origin(points)
        else:
            seg_points = points
            seg_origin_offset = np.array([0.0, 0.0, 0.0])

        seg_rgb = torch.from_numpy(rgb).to(dtype=torch.float32)
        seg_points = torch.from_numpy(points).to(dtype=torch.float32)

        seg_in_field = ME.TensorField(
            features=seg_rgb,
            coordinates=ME.utils.batched_coordinates(
                [seg_points * _config.INFERENCE.SEGMENTATION.scale],
                dtype=torch.float32,
            ),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=_device,
        )
        seg_input = seg_in_field.sparse()
        seg_output = self._segmentation_model(seg_input)
        seg_out_field = seg_output.slice(seg_in_field)

        seg_results, seg_conf = out_utils.get_segmentations_from_tensor_field(
            seg_out_field
        )
        ee_mask = seg_results == 2
        ee_idx = np.where(seg_results == 2)[0]
        seg_results[ee_idx] = 1  # initially, set all ee pred to arm

        if len(ee_idx) < _config.INFERENCE.ee_point_counts_threshold:
            return result_dto

        ee_idx_inside = self.cluster_util.get_largest_cluster(seg_points[ee_mask])
        seg_results[
            ee_idx[ee_idx_inside]
        ] = 2  # set ee classes within largest linkage cluster

        return seg_results

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

    def predict_translation(self, ee_raw_points, ee_rgb, q=None):
        with torch.no_grad():
            ee_points = np.array(ee_raw_points, copy=True)
            if (
                _config.INFERENCE.TRANSLATION.move_ee_to_origin
                or _config.INFERENCE.TRANSLATION.magic_enabled
            ) and q is not None:
                rot_mat = get_quaternion_rotation_matrix(
                    q, switch_w=False
                )  # switch_w=False in inference
                ee_points = (rot_mat.T @ ee_raw_points.reshape((-1, 3, 1))).reshape(
                    (-1, 3)
                )

            if (
                _config.INFERENCE.TRANSLATION.center_at_origin
                or _config.INFERENCE.TRANSLATION.magic_enabled
            ):
                ee_pos_points, pos_origin_offset = preprocess.center_at_origin(
                    ee_points
                )
            else:
                ee_pos_points = ee_points
                pos_origin_offset = np.array([0.0, 0.0, 0.0])

            if _config.INFERENCE.TRANSLATION.magic_enabled:
                min_z = ee_pos_points.min(axis=0)[2]
                ee_pos_magic = np.array([-0.01, 0.0, min_z])
                ee_pos_magic_reverse = ee_pos_magic + pos_origin_offset
                pos_result = rot_mat @ ee_pos_magic_reverse
            else:
                pos_points = torch.from_numpy(ee_pos_points).to(dtype=torch.float32)

                pos_in_field = ME.TensorField(
                    features=ee_rgb,
                    coordinates=ME.utils.batched_coordinates(
                        [pos_points * _config.INFERENCE.TRANSLATION.scale],
                        dtype=torch.float32,
                    ),
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                    device=_device,
                )
                pos_input = pos_in_field.sparse()
                pos_output = self._translation_model(pos_input)
                pos_out_field = pos_output.slice(pos_in_field)

                pos_result = out_utils.get_pred_center(
                    pos_out_field.features, ee_raw_points, q=q
                )

        return pos_result, pos_origin_offset

    def predict_key_points(self, raw_points, rgb):
        points = np.array(raw_points, copy=True)

        if _config.INFERENCE.KEY_POINTS.center_at_origin:
            points, origin_offset = preprocess.center_at_origin(points)
        else:
            origin_offset = np.array([0.0, 0.0, 0.0])

        if _config.INFERENCE.KEY_POINTS.use_coordinates_as_features:
            # rgb = np.array(points, copy=True)
            # if not _config.INFERENCE.KEY_POINTS.center_at_origin:
            #     rgb, rgb_origin_offset = preprocess.center_at_origin(rgb)
            # rgb /= rgb.max(axis=0)  # bw [-1, 1]
            rgb = preprocess.normalize_points(points)

        points = torch.from_numpy(points).to(dtype=torch.float32)

        if not torch.is_tensor(rgb):
            rgb = torch.from_numpy(rgb).to(dtype=torch.float32)

        if _config.INFERENCE.KEY_POINTS.backbone == "pointnet2":
            if len(points) < _config.INFERENCE.num_of_dense_input_points:
                return [], []
            if _config.INFERENCE.KEY_POINTS.pointcloud_sampling_method == 'uniform':
                sample_idx = np.random.choice(len(points), _config.INFERENCE.num_of_dense_input_points, replace=False)
            else:
                sample_idx = get_farthest_point_sample_idx(
                    points.cpu().numpy(), _config.INFERENCE.num_of_dense_input_points
                )
            inp = torch.cat(
                (points[sample_idx], rgb[sample_idx]),
                dim=-1
            ).view(1, _config.DATA.num_of_dense_input_points, -1).transpose(2, 1).to(device=_device)

            out = self._key_points_model(inp)[0].view( _config.DATA.num_of_dense_input_points, -1)
            kp_idx, kp_classes, probs = out_utils.get_key_point_predictions(
                out, conf_th=_config.INFERENCE.KEY_POINTS.conf_threshold
            )
            kp_idx = sample_idx[kp_idx]

        else:
            in_field = ME.TensorField(
                features=rgb,
                coordinates=ME.utils.batched_coordinates(
                    [points * _config.INFERENCE.KEY_POINTS.scale], dtype=torch.float32
                ),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=_device,
            )
            inp = in_field.sparse()
            out = self._key_points_model(inp)
            output = out.slice(in_field).features

            kp_idx, kp_classes, _ = out_utils.get_key_point_predictions(
                output, conf_th=_config.INFERENCE.KEY_POINTS.conf_threshold
            )

        kp_coords = raw_points[kp_idx]

        return kp_coords, kp_classes
