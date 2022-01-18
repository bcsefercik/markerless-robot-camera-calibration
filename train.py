import random
from collections import defaultdict

import torch
import torch.optim as optim
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

    if _use_cuda:
        torch.cuda.manual_seed_all(_config.GENERAL.seed)
        torch.cuda.empty_cache()

    from model.robotnet import RobotNet, get_criterion

    voxel_size = 0.02
    N_labels = 7

    # criterion = nn.MSELoss()
    criterion = get_criterion(device=_device)
    model = RobotNet(in_channels=3, out_channels=N_labels, D=3)
    print(model)

    model = model.to(_device)

    if _config.TRAIN.optim == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=_config.TRAIN.lr
        )
    elif _config.TRAIN.optim == 'SGD':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=_config.TRAIN.lr,
            momentum=_config.TRAIN.momentum,
            weight_decay=_config.TRAIN.weight_decay
        )

    coords, colors, pcd = load_file("1.ply")
    coords = torch.from_numpy(coords)
    # # Get new data
    # coordinates = ME.utils.batched_coordinates(
    #     [coords[:10000]],
    #     dtype=torch.float32,
    #     device=_device
    # )

    colors = torch.from_numpy(colors).to(_device, dtype=torch.float32)
    colors = normalize_color(colors)
    features = colors
    # features = torch.cat((colors[:100], colors[:100]), dim=0)

    # dummy_label = torch.randint(0, N_labels, (2,), device=_device)

    # input = ME.SparseTensor(features, coordinates, device=_device)

    # # Forward
    # output = model(input)
    # ipdb.set_trace()
    # in_field = ME.TensorField(
    #     features=normalize_color(features),
    #     coordinates=ME.utils.batched_coordinates([coords / voxel_size, coords], dtype=torch.float32),
    #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    #     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
    #     device=_device
    # )
    # # Convert to a sparse tensor
    # sinput = in_field.sparse()
    # # Output sparse tensor
    # soutput = model(sinput)
    # # get the prediction on the input tensor field
    # out_field = soutput.slice(in_field)
    # # logits = out_field.F
    # ipdb.set_trace()
    # Loss
    # loss = criterion(output.F, dummy_label)

    # ipdb.set_trace()
    for i in range(500):
        optimizer.zero_grad()


        in_field = ME.TensorField(
            features=features,
            coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=_device,
            requires_grad=True
        )
        # Convert to a sparse tensor
        sinput = in_field.sparse()
        # Output sparse tensor
        soutput = model(sinput)
        if i == 130:
            ipdb.set_trace()
        # out_field = soutput.slice(in_field)
        dummy_label = torch.randint(0, 1, soutput.F.size(), device=_device, dtype=torch.float32)
        # ipdb.set_trace()
        # Loss
        loss = criterion(soutput.F, dummy_label)

        print("Iteration: ", i, ", Loss: ", loss.item())

        # Gradient
        loss.backward()
        optimizer.step()



    print('no error')
