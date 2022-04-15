from enum import Enum
import ipdb

import torch
import numpy as np
import torch.nn as nn
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

        # self.pose_regression = ME.MinkowskiOps.MinkowskiLinear(
        #     self.PLANES[-1] * self.BLOCK.expansion,
        #     out_channels
        # )

        self.pose_regression = nn.Sequential(
            ME.MinkowskiOps.MinkowskiLinear(
                self.PLANES[-1] * self.BLOCK.expansion,
                2048
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiOps.MinkowskiLinear(
                2048,
                out_channels
            )
        )

    def forward(self, x):  # WXYZ
        output = self.forward_except_final(x)
        output = self.output_layer(output)
        output = self.global_pool(output)
        output = self.pose_regression(output)
        output = output.features
        output[:, 7:] = torch.sigmoid(output[:, 7:])  # confidences
        return output


class LossType(Enum):
    MSE = "mse"
    COS = "cos"
    ANGLE = "angle"
    COS2 = "cos2"


def get_criterion(device="cuda", loss_type=LossType.ANGLE, reduction="mean"):
    regression_criterion = nn.MSELoss(reduction=reduction).to(device)
    cos_regression_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    confidence_criterion = nn.BCELoss(reduction=reduction)

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
        cos_dist = 1.0 - cos_regression_criterion(y[:, :3], y_pred[:, :3])

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(cos_dist) + loss_coor

    def compute_loss(y, y_pred, reduction=reduction):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_quaternion = compute_angle_loss(y[:, 3:7], y_pred[:, 3:7])

        loss = gamma * loss_coor + gamma2 * loss_quaternion

        return loss

    def compute_cos2_loss(y, y_pred, reduction=reduction):
        gamma_cos = 2
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        cos_dist = 1.0 - cos_regression_criterion(y[:, :7], y_pred[:, :7])
        cos_dist *= gamma_cos

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

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(cos_dist) + loss_coor + loss_confidence

    if loss_type == LossType.COS:
        return compute_cos_loss
    elif loss_type == LossType.MSE:
        return regression_criterion
    elif loss_type == LossType.COS2:
        return compute_cos2_loss
    else:
        return compute_loss
