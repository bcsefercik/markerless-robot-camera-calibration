from collections import defaultdict
import json
import time
import ipdb

import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.alivev2 import AliveV2Dataset, collate_non_quantized
from utils import config, logger, utils
from model.backbone import minkunet
from model.robotnet_vote import RobotNetVote
from model.robotnet_encode import RobotNetEncode
from model.robotnet import RobotNet

import MinkowskiEngine as ME


_config = config.Config()
_config.save()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")


class InferenceEngine:
    def __init__(self) -> None:
        self.models = defaultdict(lambda: minkunet.MinkUNet18D)
        self.models["minkunet101"] = minkunet.MinkUNet101
        self.models["minkunet34C"] = minkunet.MinkUNet34C
        self.models["minkunet14A"] = minkunet.MinkUNet14A
        self.models["minkunet"] = minkunet.MinkUNet18D
        self.models["translation"] = RobotNetVote
        self.models["robotnet_encode"] = RobotNetEncode
        self.models["robotnet"] = RobotNet

        self.segmentation_model = self.models[_config.INFERENCE.SEGMENTATION.backbone](
            in_channels=_config.DATA.input_channel,
            out_channels=_config.INFERENCE.SEGMENTATION.classes
        )
        utils.checkpoint_restore(
            self.segmentation_model,
            f=_config.INFERENCE.SEGMENTATION.checkpoint,
            use_cuda=_use_cuda,
        )
        self.segmentation_model.eval()

        self.translation_model = self.models["translation"](
            in_channels=_config.DATA.input_channel,
            num_classes=len(_config.INFERENCE.TRANSLATION.classes)
        )
        utils.checkpoint_restore(
            self.translation_model,
            f=_config.INFERENCE.TRANSLATION.checkpoint,
            use_cuda=_use_cuda,
        )
        self.segmentation_model.eval()

        compute_confidence = _config()['STRUCTURE'].get('compute_confidence', False)
        self.rotation_model = self.models[f'robotnet{"_encode" if _config.INFERENCE.ROTATION.encode_only else ""}'](
            in_channels=_config.DATA.input_channel,
            out_channels=(10 if compute_confidence else 7)
        )
        utils.checkpoint_restore(
            self.rotation_model,
            f=_config.INFERENCE.ROTATION.checkpoint,
            use_cuda=_use_cuda,
        )
        self.rotation_model.eval()



if __name__ == "__main__":
    engine = InferenceEngine()

    # TODO: make data loading lose coupled
    dt = "test"
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
    dataset = AliveV2Dataset(set_name=dt, file_names=file_names[dt], quantization_enabled=False)
    data_loader = DataLoader(
        dataset,
        batch_size=_config.TEST.batch_size,
        collate_fn=collate_non_quantized,
        num_workers=_config.TEST.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    data_iter = iter(data_loader)

    with torch.no_grad():
        start = time.time()

        overall_results = defaultdict(list)
        individual_results = defaultdict(lambda: defaultdict(list))
        results_json = {}

        for i, batch in enumerate(data_iter):
            coords, feats, labels, _, others = batch
            for fi, other_info in enumerate(others):
                start_ins = time.time()
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
                soutput = engine.segmentation_model(sinput)
                soutput1 = engine.rotation_model(sinput)
                soutput2 = engine.translation_model(sinput)
                out_field = soutput.slice(in_field)
                logits = out_field.F

                _, preds = logits.max(1)
                preds = preds.cpu().numpy()
                labels_cpu = labels[start:end].cpu().numpy().reshape((-1))
                end_ins = time.time()
                dur = round(end_ins - start_ins, 2)
                print(f"{other_info['filename']} time: {dur} fps: {round(1/dur,2 )}")


                # ipdb.set_trace()
