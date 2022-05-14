import pickle
import random
import time
import os
import json
import traceback
import statistics
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d

from utils import config, logger, utils, metrics
# from train_segmentation import compute_accuracies
from utils.output import ClusterUtil, get_key_points

import ipdb


_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")

_ee_cluster = ClusterUtil()

torch.set_printoptions(precision=_config.TEST.print_precision, sci_mode=False)


def test(model, criterion, data_loader, output_filename="results.txt"):
    data_iter = iter(data_loader)
    model = model.eval()
    with torch.no_grad():
        start = time.time()

        overall_results = defaultdict(list)
        individual_results = defaultdict(lambda: defaultdict(list))
        results_json = {}

        for i, batch in enumerate(data_iter):
            try:
                coords, feats, labels, _, others = batch
                labels = labels.to(device=_device)
                # model_input = ME.SparseTensor(feats, coordinates=coords, device=_device)
                # out = model(model_input)

                # loss = criterion(out.features, labels)
                # accuracies = compute_accuracies(out, labels, others)

                for fi, other_info in enumerate(others):
                    start = other_info["offset"][0]
                    end = other_info["offset"][1]


                    in_field = ME.TensorField(
                        features=feats[start:end],
                        coordinates=ME.utils.batched_coordinates([coords[start:end]], dtype=torch.float32),
                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                        device=_device,
                    )

                    sinput = in_field.sparse()
                    soutput = model(sinput)
                    out_field = soutput.slice(in_field)
                    logits = out_field.F

                    key_points_idx, key_points_classes = get_key_points(logits)
                    key_points_pred = (coords[start:end][key_points_idx] * data_loader.dataset.quantization_size) + other_info['origin_offset']
                    key_points_gt = other_info['key_points'][key_points_classes]
                    dists = np.linalg.norm(key_points_gt - key_points_pred.cpu().numpy(), axis=1)
                    # if you need to cluster/filter ee output use following line
                    # biggest_ee_idx = _ee_cluster.get_largest_cluster(coords[start:end][labels_cpu == 2])

                    fname = other_info["filename"]
                    position = other_info["position"]

                    print(f"{position}/{fname}")


                    # ipdb.set_trace()
            except Exception as e:
                print(e)
                _logger.exception(f"Filenames: {json.dumps([o['filepath'] for o in others])}")
                raise e

        with open(output_filename.replace('.txt', '.pickle'), "wb") as fp:
            pickle.dump(results_json, fp)

        for pos in individual_results:
            for k in individual_results[pos]:
                individual_results[pos][k] = round(
                    statistics.mean(individual_results[pos][k]), 4
                )
                overall_results[k].append(individual_results[pos][k])

        for k in overall_results:
            overall_results[k] = round(statistics.mean(overall_results[k]), 4)

        with open(output_filename, "a") as fp:
            fp.write("\n---------- SUMMARY ----------\n")

            for pos in individual_results:
                fp.write(f"{pos}: {json.dumps(individual_results[pos], indent=4)}\n")

            fp.write(f"Overall: {json.dumps(overall_results, indent=4)}\n")


if __name__ == "__main__":
    print(f"CONFIG: {_config()}")

    if _use_cuda:
        torch.cuda.empty_cache()

    if _config()["STRUCTURE"].get("encode_only", False):
        from model.robotnet_encode_keypoint import RobotNetEncode as RobotNet
    else:
        from model.robotnet_segmentation import RobotNetSegmentation as RobotNet

    from data.alivev2 import AliveV2Dataset, collate_non_batched

    criterion = nn.CrossEntropyLoss(
        ignore_index=_config.DATA.ignore_label,
        reduction=_config()["TRAIN"].get("loss_reduction", "mean"),
    ).to(_device)

    model = RobotNet(in_channels=3, num_classes=10, D=3)  # out: # of kps

    start_epoch = utils.checkpoint_restore(
        model,
        f=os.path.join(_config.exp_path, _config.TEST.checkpoint),
        use_cuda=_use_cuda,
    )

    print("Loaded model.")

    dataset_name = ""

    file_names = defaultdict(list)
    file_names_path = _config()['DATA'].get('file_names')
    if file_names_path:
        file_names_path = file_names_path.split(',')

        dataset_name = utils.remove_suffix(file_names_path[0].split('/')[-1], '.json')

        with open(file_names_path[0], 'r') as fp:
            file_names = json.load(fp)

        for fnp in file_names_path[1:]:
            with open(fnp, 'r') as fp:
                new_file_names = json.load(fp)

                for k in new_file_names:
                    if k in file_names:
                        file_names[k].extend(new_file_names[k])

    for dt in ("val", "test", "train"):
        print("Dataset:", dt)

        if not file_names[dt]:
            print(f"Dataset {dt} split is empty.")
            continue

        dataset = AliveV2Dataset(set_name=dt, file_names=file_names[dt], quantization_enabled=False)
        data_loader = DataLoader(
            dataset,
            batch_size=_config.TEST.batch_size,
            collate_fn=collate_non_batched,
            num_workers=_config.TEST.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        test(
            model,
            criterion,
            data_loader,
            output_filename=os.path.join(
                _config.exp_path,
                f"{utils.remove_suffix(_config.TEST.checkpoint, '.pth')}_results_{dataset_name}_{dt}.txt",
            ),
        )

    # ipdb.set_trace()

    print("DONE!")
