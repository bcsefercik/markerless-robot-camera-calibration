from enum import Enum

import torch
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME

from model.backbone.minkunet import MinkUNet18D as UNet
from utils.quaternion import qeuler


class RobotNet(UNet):
    name = "robotnet"

    def __init__(self, in_channels, out_channels, D=3):
        UNet.__init__(self, in_channels, out_channels, D)

        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.leaky_relu = ME.MinkowskiLeakyReLU(inplace=True)
        self.final_bn = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):  # WXYZ
        unet_output = super().forward(x)
        unet_output = self.final_bn(unet_output)
        unet_output = self.leaky_relu(unet_output)
        return self.global_pool(unet_output)


class LossType(Enum):
    MSE = 'mse'
    COS = 'cos'
    ANGLE = 'angle'


def get_criterion(device="cuda", loss_type=LossType.ANGLE):
    regression_criterion = nn.MSELoss(reduction="mean").to(device)
    cos_regression_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    gamma = 50
    gamma2 = 1

    def compute_angle_loss(q_expected, q_pred, reduction="mean"):
        expected_euler = qeuler(q_expected, order='zyx', epsilon=1e-6)
        predicted_euler = qeuler(q_pred, order='zyx', epsilon=1e-6)
        angle_distance = torch.remainder(predicted_euler - expected_euler + np.pi, 2*np.pi) - np.pi

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(torch.abs(angle_distance))

    def compute_cos_loss(y, y_pred, reduction="mean"):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        cos_dist = 1. - cos_regression_criterion(y, y_pred)

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(cos_dist) + loss_coor

    def compute_loss(y, y_pred):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_quaternion = compute_angle_loss(y[:, 3:], y_pred[:, 3:])

        loss = gamma * loss_coor + gamma2 * loss_quaternion

        return loss

    if loss_type == LossType.COS:
        return compute_cos_loss
    elif loss_type == LossType.MSE:
        return regression_criterion
    else:
        return compute_loss
