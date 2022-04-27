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
from tensorboardX import SummaryWriter

from utils import config, logger, utils, metrics
from train_vote import compute_accuracies, compute_center_dists

import ipdb


_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")

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
            # try:
            coords, feats, labels, pose, others = batch
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
                    coordinates=ME.utils.batched_coordinates([coords[start:end] / data_loader.dataset.quantization_size], dtype=torch.float32),
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                    device=_device,
                )

                sinput = in_field.sparse()
                soutput = model(sinput)

                out_field = soutput.slice(in_field)
                logits = out_field.F

                center_pred = coords[logits.argmax()].tolist()

                center_dist = np.linalg.norm(np.array(center_pred) - np.array(other_info['ee_center']))
                center_dist = round(center_dist, 4)

                _, preds = logits.max(1)
                preds = preds.cpu().numpy()
                labels_cpu = labels[start:end].cpu().numpy().reshape((-1))
                accuracy = (preds == labels_cpu).sum() / len(labels_cpu)

                fname = other_info["filename"]
                position = other_info["position"]

                print(f"{position}/{fname}")
                # preds_fi = [round(p, 4) for p in out[fi].tolist()]
                result = {
                    "center_pred_conf": logits.max().item(),
                    "center_dist": center_dist,
                    "center_pred": center_pred,
                    "center_gt": other_info['ee_center']
                }

                # individual_results[position]["accuracy"].append(
                #     result["accuracy"]
                # )
                individual_results[position]["center_dist"].append(
                    result["center_dist"]
                )
                individual_results[position]["pred_conf"].append(
                    result["center_pred_conf"]
                )
                overall_results["center_dist"].append(
                    result["center_dist"]
                )
                overall_results["pred_conf"].append(
                    result["center_pred_conf"]
                )
                results_json[f"{position}/{fname}"] = result

                with open(output_filename, "a") as fp:
                    fp.write(
                        f"{position}/{fname}: {json.dumps(result, sort_keys=True, indent=4)}\n"
                    )
                # ipdb.set_trace()
            # except Exception as e:
            #     print(e)
                # _logger.exception(f"Filenames: {json.dumps(others)}")

        with open(output_filename.replace('.txt', '.json'), "a") as fp:
            json.dump(results_json, fp)

        for k in overall_results:
            overall_results[k] = round(statistics.mean(overall_results[k]), 4)

        individual_results_to_print = dict()

        for pos in individual_results:
            individual_results_to_print[pos] = dict()
            for k in individual_results[pos]:
                individual_results_to_print[pos][f'{k}_max'] = round(max(individual_results[pos][k]), 4)
                individual_results_to_print[pos][f'{k}_min'] = round(min(individual_results[pos][k]), 4)
                individual_results_to_print[pos][f'{k}_mean'] = round(
                    statistics.mean(individual_results[pos][k]), 4
                )
                # ipdb.set_trace()
        with open(output_filename, "a") as fp:
            fp.write("\n---------- SUMMARY ----------\n")

            for pos in individual_results_to_print:
                fp.write(f"{pos}: {json.dumps(individual_results_to_print[pos], indent=4)}\n")

            fp.write(f"Overall: {json.dumps(overall_results, indent=4)}\n")


if __name__ == "__main__":
    print(f"CONFIG: {_config()}")

    if _use_cuda:
        torch.cuda.empty_cache()

    from model.robotnet_vote import RobotNetVote

    from data.alivev2 import AliveV2Dataset, collate_non_quantized

    criterion = nn.BCELoss(
        reduction=_config()["TRAIN"].get("loss_reduction", "mean")
    ).to(_device)

    model = RobotNetVote(in_channels=_config.DATA.input_channel)

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
            collate_fn=collate_non_quantized,
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
