import random
import time
import os
import json
import traceback
import statistics
import datetime
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from tensorboardX import SummaryWriter

from utils import config, logger, utils, metrics


import ipdb


_config = config.Config()
_logger = logger.Logger().get()
_tensorboard_writer = SummaryWriter(_config.exp_path)

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")

torch.set_printoptions(precision=_config.TEST.print_precision, sci_mode=False)


def test(model, criterion, data_loader, output_filename="results.txt"):
    data_iter = iter(data_loader)
    model = model.eval()
    with torch.no_grad():
        start = time.time()

        overall_results = defaultdict(list)
        individual_results = {
            "p1": defaultdict(list),
            "p2": defaultdict(list),
            "p3": defaultdict(list),
        }

        for i, batch in enumerate(data_iter):
            coords, feats, _, poses, others = batch
            poses = poses.to(device=_device)

            model_input = ME.SparseTensor(feats, coordinates=coords, device=_device)
            out = model(model_input)

            loss = criterion(out.F, poses).item()

            (
                dist,
                dist_position,
                dist_orientation,
                angle_diff,
            ) = metrics.compute_pose_dist(poses, out.features)

            ipdb.set_trace()


if __name__ == "__main__":
    if _use_cuda:
        torch.cuda.empty_cache()

    from model.robotnet import RobotNet, get_criterion
    from data.alivev1 import AliveV1Dataset, collate

    criterion = get_criterion(device=_device)
    model = RobotNet(in_channels=3, out_channels=7, D=3)

    start_epoch = utils.checkpoint_restore(
        model,
        f=os.path.join(_config.exp_path, _config.TEST.checkpoint),
        use_cuda=_use_cuda,
    )

    print("Loaded model.")

    # train_dataset = AliveV1Dataset(set_name="train")
    # train_data_loader = DataLoader(
    #     train_dataset,
    #     batch_size=_config.TEST.batch_size,
    #     collate_fn=collate,
    #     num_workers=_config.TEST.workers,
    #     shuffle=False,
    #     drop_last=False,
    #     pin_memory=True,
    # )
    val_dataset = AliveV1Dataset(set_name="val")
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=_config.TEST.batch_size,
        collate_fn=collate,
        num_workers=_config.TEST.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    # test_dataset = AliveV1Dataset(set_name="test")
    # test_data_loader = DataLoader(
    #     test_dataset,
    #     batch_size=_config.TEST.batch_size,
    #     collate_fn=collate,
    #     num_workers=_config.TEST.workers,
    #     shuffle=False,
    #     drop_last=False,
    #     pin_memory=True,
    # )

    test(
        model,
        criterion,
        val_data_loader,
        output_filename=os.path.join(
            _config.exp_path,
            f"{utils.remove_suffix(_config.TEST.checkpoint, '.pth')}_val.txt",
        ),
    )

    ipdb.set_trace()

    print("DONE!")
