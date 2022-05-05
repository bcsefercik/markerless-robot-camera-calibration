from enum import Enum
from ntpath import join
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


class RobotNetVote(UNet):
    name = "robotnet"

    def __init__(self, in_channels, out_channels=256, D=3, num_classes=(2 if _config.DATA.data_type == "ee_seg" else 4)):
        UNet.__init__(self, in_channels, out_channels, D)
        # self.leaky_relu = nn.LeakyReLU()
        self.leaky_relu = ME.MinkowskiLeakyReLU()

        # self.regression = nn.Sequential(
        #     nn.Linear(256, 1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024, 1)
        # )

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


def get_criterion():
    # return nn.BCEWithLogitsLoss(reduction=_config()["TRAIN"].get("loss_reduction", "mean"))
    return torch.nn.CrossEntropyLoss(
        reduction=_config()["TRAIN"].get("loss_reduction", "mean"),
        ignore_index=_config.DATA.ignore_label
    )
