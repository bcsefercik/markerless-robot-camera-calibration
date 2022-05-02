from enum import Enum
import ipdb

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from utils.quaternion import qeuler
from utils.metrics import compute_pose_dist
from utils import config


_config = config.Config()

_backbone = _config()['STRUCTURE'].get('backbone')
if _backbone == 'minkunet':
    from model.backbone.minkunet import MinkUNet18D as UNet
elif _backbone == 'minkunet101':
    from model.backbone.minkunet import MinkUNet101 as UNet
elif _backbone == 'minkunet34C':
    from model.backbone.minkunet import MinkUNet34C as UNet
elif _backbone == 'minkunet14A':
    from model.backbone.minkunet import MinkUNet14A as UNet
else:
    from model.backbone.aliveunet import AliveUNet as UNet

M = _config.STRUCTURE.m

EPS = 1e-6


class RobotNet(UNet):
    name = "robotnet"

    def __init__(self, in_channels, out_channels, D=3):
        UNet.__init__(self, in_channels, out_channels, D)
        # self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.global_pool = ME.MinkowskiGlobalMaxPooling()
        self.leaky_relu = ME.MinkowskiLeakyReLU(inplace=False)
        self.final_bn = ME.MinkowskiBatchNorm(out_channels)

        self.output_layer = nn.Sequential(
            ME.MinkowskiBatchNorm(self.PLANES[-1] * self.BLOCK.expansion),
            self.relu
        )

        self.pose_regression_input_size = self.PLANES[-1] * self.BLOCK.expansion
        if _config.STRUCTURE.use_joint_angles:
            self.pose_regression_input_size += 9

        self.pose_regression = nn.Sequential(
            nn.Linear(self.pose_regression_input_size, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, out_channels)
        )

    def forward(self, x):  # WXYZ
        if isinstance(x, tuple):
            x, joint_angles = x
        else:
            x = x

        output = self.forward_except_final(x)
        output = self.output_layer(output)
        output = self.global_pool(output)

        if _config.STRUCTURE.use_joint_angles:
            regression_input = torch.cat((output.features, joint_angles), dim=1)
        else:
            regression_input = output.features

        output = self.pose_regression(regression_input)

        output[:, 7:] = torch.sigmoid(output[:, 7:])  # confidences

        if not self.training:
            output[:, 3:7] = F.normalize(output[:, 3:7], p=2, dim=1)
        return output


class LossType(Enum):
    MSE = "mse"
    COS = "cos"
    ANGLE = "angle"
    COS2 = "cos2"
    WGEODESIC = "wgeodesic"
    SMOOTHL1 = "smoothl1"


def get_criterion(device="cuda", loss_type=LossType.ANGLE, reduction="mean"):
    regression_criterion = nn.MSELoss(reduction=reduction).to(device)
    cos_regression_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    confidence_criterion = nn.BCELoss(reduction=reduction)
    smooth_l1_criterion = nn.SmoothL1Loss(reduction=reduction).to(device)

    confidence_enabled = _config()['STRUCTURE'].get('compute_confidence', False)

    gamma = 50
    gamma2 = 1

    def compute_angle_loss(q_expected, q_pred, reduction=reduction):
        expected_euler = qeuler(q_expected, order="zyx", epsilon=1e-6)
        predicted_euler = qeuler(q_pred, order="zyx", epsilon=1e-6)
        angle_distance = (
            torch.remainder(predicted_euler - expected_euler + np.pi, 2 * np.pi) - np.pi
        )

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(torch.abs(angle_distance))

    def compute_cos_loss(y, y_pred, reduction=reduction):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_rot = 1.0 - cos_regression_criterion(y[:, :3], y_pred[:, :3])

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(loss_rot) + loss_coor

    def compute_loss(y, y_pred, reduction=reduction):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_quaternion = compute_angle_loss(y[:, 3:7], y_pred[:, 3:7])

        loss = gamma * loss_coor + gamma2 * loss_quaternion

        return loss

    def compute_cos2_loss(y, y_pred, reduction=reduction):
        reduction_func = torch.sum if reduction == "sum" else torch.mean

        gamma_cos = 2

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            if not _config()['STRUCTURE'].get('disable_position', False):
                loss_rot = 1.0 - cos_regression_criterion(y[:, :7], y_pred[:, :7])
                loss_rot = reduction_func(loss_rot)
            else:
                loss_rot = regression_criterion(y[:, 3:7], y_pred[:, 3:7])
            loss_rot *= gamma_cos

        loss_confidence = 0
        if confidence_enabled:
            _, dist_position, _, angle_diff = compute_pose_dist(y, y_pred[:, :7])
            position_confidence_idx = (dist_position < _config.STRUCTURE.position_threshold) + (dist_position > _config.STRUCTURE.position_ignore_threshold)
            position_confidence = (dist_position < _config.STRUCTURE.position_threshold).float()
            loss_confidence += confidence_criterion(
                y_pred[:, 7][position_confidence_idx],
                position_confidence[position_confidence_idx]
            )

            orientation_confidence_idx = (angle_diff < _config.STRUCTURE.angle_diff_threshold) + (angle_diff > _config.STRUCTURE.angle_diff_ignore_threshold)
            orientation_confidence = (angle_diff < _config.STRUCTURE.angle_diff_threshold).float()
            loss_confidence += confidence_criterion(
                y_pred[:, 8][orientation_confidence_idx],
                orientation_confidence[orientation_confidence_idx]
            )

            overall_confidence_idx = position_confidence_idx * orientation_confidence_idx
            overall_confidence = position_confidence * orientation_confidence
            loss_confidence += confidence_criterion(
                y_pred[:, 9][overall_confidence_idx],
                overall_confidence[overall_confidence_idx]
            )

        return loss_rot + loss_coor + loss_confidence

    def compute_with_geodesic_loss(y, y_pred, reduction=reduction):
        reduction_func = torch.sum if reduction == "sum" else torch.mean

        gamma_cos = 1

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            y_normalized = F.normalize(y[:, 3:7], p=2, dim=1)
            y_pred_normalized = F.normalize(y_pred[:, 3:7], p=2, dim=1)

            loss_rot = torch.acos(
                            (torch.sum(y_normalized * y_pred_normalized, dim=1) - 1) * 0.5,
                        )
            loss_rot = reduction_func(loss_rot)
            loss_rot *= gamma_cos

        loss_confidence = 0

        return loss_rot + loss_coor + loss_confidence

    def compute_smooth_l1_loss(y, y_pred, reduction=reduction):
        reduction_func = torch.sum if reduction == "sum" else torch.mean

        gamma_cos = 1

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = smooth_l1_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            y_normalized = F.normalize(y[:, 3:7], p=2, dim=1)
            y_pred_normalized = F.normalize(y_pred[:, 3:7], p=2, dim=1)

            loss_rot = torch.acos(
                            (torch.sum(y_normalized * y_pred_normalized, dim=1) - 1) * 0.5,
                        )
            loss_rot = reduction_func(loss_rot)
            loss_rot *= gamma_cos

        loss_confidence = 0

        return loss_rot + loss_coor + loss_confidence

    if loss_type == LossType.COS:
        return compute_cos_loss
    elif loss_type == LossType.MSE:
        return regression_criterion
    elif loss_type == LossType.COS2:
        return compute_cos2_loss
    elif loss_type == LossType.WGEODESIC:
        return compute_with_geodesic_loss
    elif loss_type == LossType.SMOOTHL1:
        return compute_smooth_l1_loss
    else:
        return compute_loss
