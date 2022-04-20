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


class RobotNetEncode(UNet):
    name = "robotnet"

    def __init__(self, in_channels, out_channels, D=3):
        UNet.__init__(self, in_channels, out_channels, D)
        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        # self.global_pool = ME.MinkowskiGlobalMaxPooling()
        self.leaky_relu = ME.MinkowskiLeakyReLU(inplace=False)
        self.final_bn = ME.MinkowskiBatchNorm(out_channels)

        # self.pose_regression = nn.Sequential(
        #     ME.MinkowskiOps.MinkowskiLinear(
        #         self.PLANES[3] * self.BLOCK.expansion,
        #         out_channels
        #     )
        # )

        self.pose_regression = nn.Sequential(
            ME.MinkowskiOps.MinkowskiLinear(
                self.PLANES[3] * self.BLOCK.expansion,
                2048
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiOps.MinkowskiLinear(
                2048,
                out_channels
            )
        )

        # self.pose_regression = nn.Sequential(
        #     ME.MinkowskiOps.MinkowskiLinear(
        #         self.PLANES[3] * self.BLOCK.expansion,
        #         2048
        #     ),
        #     ME.MinkowskiReLU(),
        #     ME.MinkowskiOps.MinkowskiLinear(
        #         2048,
        #         2048
        #     ),
        #     ME.MinkowskiReLU(),
        #     ME.MinkowskiOps.MinkowskiLinear(
        #         2048,
        #         out_channels
        #     )
        # )

    def forward(self, x):  # WXYZ
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        output = self.block4(out)

        # tensor_stride=8
        # out = self.convtr4p16s2(out)
        # out = self.bntr4(out)
        # output = self.relu(out)

        output = self.global_pool(output)
        output = self.pose_regression(output)
        output = output.features
        output[:, 7:] = torch.sigmoid(output[:, 7:])  # confidences
        return output
