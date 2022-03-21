import torch
import torch.nn as nn
import MinkowskiEngine as ME


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = nn.Sequential(
                nn.Identity()
            )
        else:
            self.i_branch = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, bias=False)
            )