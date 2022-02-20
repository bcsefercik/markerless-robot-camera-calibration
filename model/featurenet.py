from enum import Enum

import torch
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME

from pytorch_metric_learning import miners, losses

from model.backbone.minkunet import MinkUNet18 as UNet


class FeatureNet(UNet):
    name = "featurenet"


def get_criterion(device="cuda"):
    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss()

    return loss_func, miner
