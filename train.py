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

    criterion = nn.CrossEntropyLoss()
    net = RobotNet(in_channels=3, out_channels=N_labels, D=3)
    print(net)


    net = net.to(_device)
    optimizer = SGD(net.parameters(), lr=1e-4)

    coords, colors, pcd = load_file("1.ply")
    coords = torch.from_numpy(coords)
    # Get new data
    coordinates = ME.utils.batched_coordinates(
        [coords, coords],
        dtype=torch.float32,
        device=_device
    )

    colors = torch.from_numpy(colors)
    features = torch.cat((colors, colors), dim=0).to(_device, dtype=coordinates.dtype)


    dummy_label = torch.randint(0, N_labels, (2,), device=_device)
    for i in range(500):
        optimizer.zero_grad()

        input = ME.SparseTensor(features, coordinates, device=_device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, dummy_label)
        print("Iteration: ", i, ", Loss: ", loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    print('no error')
