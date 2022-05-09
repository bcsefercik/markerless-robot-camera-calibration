
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
    _backbone = _config.INFERENCE.SEGMENTATION.backbone

if _backbone == 'minkunet101':
    from model.backbone.minkunet import MinkUNet101 as UNet
elif _backbone == 'minkunet34C':
    from model.backbone.minkunet import MinkUNet34C as UNet
elif _backbone == 'minkunet14A':
    from model.backbone.minkunet import MinkUNet14A as UNet
else:
    from model.backbone.minkunet import MinkUNet18D as UNet

M = _config.STRUCTURE.m

EPS = 1e-6


class RobotNetSegmentation(UNet):
    name = "robotnet"

    def __init__(self, in_channels, out_channels=256, D=3, num_classes=_config.DATA.classes):
        UNet.__init__(self, in_channels, out_channels, D)
        # self.leaky_relu = nn.LeakyReLU()
        self.leaky_relu = ME.MinkowskiLeakyReLU()

        self.regression = nn.Sequential(
            ME.MinkowskiOps.MinkowskiLinear(256, 1024),
            ME.MinkowskiLeakyReLU(),
            ME.MinkowskiOps.MinkowskiLinear(
                1024,
                num_classes
            )
        )

        # self.sigm = nn.Sigmoid()
        self.sigm = ME.MinkowskiSigmoid()

    def forward(self, x):
        if isinstance(x, tuple):
            x, joint_angles = x

        output = super().forward(x)
        output = self.leaky_relu(output)

        output = self.regression(output)
        # output = self.sigm(output)
        return output
