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


def get_criterion(device="cuda"):
    regression_criterion = nn.MSELoss(reduction="mean").to(device)
    cos_regression_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    gamma = 1
    gamma2 = 1

    def compute_angle_loss(q_expected, q_pred, reduction="mean"):
        expected_euler = qeuler(q_expected, order='zyx', epsilon=1e-6)
        predicted_euler = qeuler(q_pred, order='zyx', epsilon=1e-6)
        angle_distance = torch.remainder(predicted_euler - expected_euler + np.pi, 2*np.pi) - np.pi

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(torch.abs(angle_distance))

    def compute_cos_loss(q_expected, q_pred, reduction="mean"):
        cos_dist = 1. - cos_regression_criterion(q_expected, q_pred))

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(cos_dist)

    def compute_loss(y, y_pred):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_quaternion = compute_cos_loss(y, y_pred)

        loss = gamma * loss_coor + gamma2 * loss_quaternion

        return loss

    return compute_loss
