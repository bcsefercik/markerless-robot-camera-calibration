import os
import time
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

from utils.data import normalize_color


_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


def eval(data_loader, model, criterion, miner):
    data_iter = iter(data_loader)
    model.eval()

    with open("features_val.tsv", "w") as ffp, open("metadata_val.tsv", "w") as mfp:
        mfp.write('label\tname\n')

    with torch.no_grad():
        start = time.time()

        overall_results = defaultdict(list)
        individual_results = defaultdict(lambda: defaultdict(list))

        for i, batch in enumerate(data_iter):
            # try:
            coords, rgb, labels, others = batch
            labels = labels.tolist()
            model_input = ME.SparseTensor(rgb, coordinates=coords, device=_device)
            out = model(model_input)

            with open("features_val.tsv", "a") as ffp, open("metadata_val.tsv", "a") as mfp:
                for j, embed in enumerate(out.features.tolist()):
                    print(f'{others[j]["object_name"]}/{others[j]["filename"]}')

                    mfp.write(f'{labels[j]}\t{others[j]["object_name"]}\n')
                    ffp.write("\t".join([str(e) for e in embed]))
                    ffp.write('\n')
                # labels.extend(labels.tolist())


                # except Exception:
                #     ipdb.set_trace()
                #     continue


if __name__ == "__main__":
    if _use_cuda:
        torch.cuda.manual_seed_all(_config.GENERAL.seed)
        torch.cuda.empty_cache()

    from model.featurenet import FeatureNet, get_criterion
    from data.ycbv2 import YCBDataset, collate

    criterion, miner = get_criterion(device=_device)
    model = FeatureNet(
        in_channels=3, out_channels=_config.STRUCTURE.embedding_size, D=3
    )
    if _use_cuda:
        model.cuda()

    model.eval()
    _logger.info(f"Model: {str(model)}")

    start_epoch = utils.checkpoint_restore(
        model,
        f=os.path.join(_config.exp_path, _config.TEST.checkpoint),
        use_cuda=_use_cuda,
    )

    file_names = defaultdict(list)
    file_names_path = _config()["DATA"].get("file_names")
    if file_names_path:
        with open(file_names_path, "r") as fp:
            file_names = json.load(fp)

    # for dtype in ["train", "val", "test"]:
    for dtype in ["val"]:

        dataset = YCBDataset(set_name=dtype, file_names=file_names[dtype])
        data_loader = DataLoader(
            dataset,
            batch_size=_config.TEST.batch_size,
            collate_fn=collate,
            num_workers=_config.TEST.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=utils.seed_worker,
            generator=utils.torch_generator,
        )

        eval(data_loader, model, criterion, miner)
    ipdb.set_trace()

    _logger.info("DONE!")
