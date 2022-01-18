import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from tensorboardX import SummaryWriter

from utils import config, logger, utils


import ipdb


_config = config.Config()
_logger = logger.Logger().get()
_tensorboard_writer = SummaryWriter(_config.exp_path)

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")

def normalize_color(color: torch.Tensor, is_color_in_range_0_255: bool = False) -> torch.Tensor:
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.

    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()

def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd

if __name__ == '__main__':
    _logger.info('Starting new training.')
    _logger.info(f'CONFIG: {_config()}')

    _logger.info(f"Setting seed: {_config.GENERAL.seed}")
    random.seed(_config.GENERAL.seed)
    np.random.seed(_config.GENERAL.seed)
    torch.manual_seed(_config.GENERAL.seed)
    torch.cuda.manual_seed_all(_config.GENERAL.seed)

    from model.robotnet import RobotNet

    voxel_size = 0.02
    N_labels = 4

    criterion = nn.MSELoss()
    net = RobotNet(in_channels=3, out_channels=N_labels, D=3)
    print(net)

    net = net.to(_device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    coords, colors, pcd = load_file("1.ply")
    coords = torch.from_numpy(coords)
    # # Get new data
    # coordinates = ME.utils.batched_coordinates(
    #     [coords[:10000]],
    #     dtype=torch.float32,
    #     device=_device
    # )

    # colors = torch.from_numpy(colors[:10000]).to(_device, dtype=coordinates.dtype)
    # features = colors
    # # features = torch.cat((colors, colors), dim=0)

    # dummy_label = torch.randint(0, N_labels, (2,), device=_device)

    # input = ME.SparseTensor(features, coordinates, device=_device)

    # # Forward
    # output = net(input)
    # ipdb.set_trace()
    # in_field = ME.TensorField(
    #     features=normalize_color(torch.from_numpy(colors)),
    #     coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
    #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    #     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
    #     device=_device,
    #     requires_grad=True
    # )
    # # Convert to a sparse tensor
    # sinput = in_field.sparse()
    # # Output sparse tensor
    # soutput = net(sinput)
    # # get the prediction on the input tensor field
    # out_field = soutput.slice(in_field)
    # logits = out_field.F

    # Loss
    # loss = criterion(output.F, dummy_label)

    # ipdb.set_trace()
    for i in range(500):
        optimizer.zero_grad()


        in_field = ME.TensorField(
            features=normalize_color(torch.from_numpy(colors)),
            coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=_device,
            requires_grad=True
        )
        # Convert to a sparse tensor
        sinput = in_field.sparse()
        # Output sparse tensor
        soutput = net(sinput)
        out_field = soutput.slice(in_field)
        dummy_label = torch.randint(0, 1, out_field.F.size(), device=_device, dtype=torch.float32)
        # Loss
        loss = criterion(out_field.F, dummy_label)
        print("Iteration: ", i, ", Loss: ", loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    print('no error')
