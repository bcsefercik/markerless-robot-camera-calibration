
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
if _config.MODE == "inference":
    _backbone = _config.INFERENCE.ROTATION.backbone

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
