from enum import Enum

import torch
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME

from pytorch_metric_learning import miners, losses

# from model.backbone.minkunet import MinkUNet18 as UNet
from model.backbone.minkunet import MinkUNet34A as UNet


class FeatureNet(UNet):
    name = "featurenet"
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
    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss()

    return loss_func, miner
