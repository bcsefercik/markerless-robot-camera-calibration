import torch.nn as nn
import MinkowskiEngine as ME

from model.backbone.minkunet import MinkUNet34A as UNet


class RobotNet(UNet):
    name = 'robotnet'

    def __init__(self, in_channels, out_channels, D=3):
        UNet.__init__(self, in_channels, out_channels, D)

        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.leaky_relu = ME.MinkowskiLeakyReLU(inplace=True)
        self.final_bn = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        unet_output = super().forward(x)
        unet_output = self.final_bn(unet_output)
        unet_output = self.leaky_relu(unet_output)
        return self.global_pool(unet_output)


def get_criterion(device="cuda"):
    regression_criterion = nn.MSELoss(reduction='sum').to(device)
    cos_regression_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    gamma = 2
    gamma2 = 2

    def compute_loss(y, y_pred):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_coor *= gamma2
        loss_quaternion = (gamma * (1. - cos_regression_criterion(y[:, 3:], y_pred[:, 3:]))).sum()

        loss = loss_coor + loss_quaternion

        return loss

    return compute_loss
