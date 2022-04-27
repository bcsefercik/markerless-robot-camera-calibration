import os
import sys
import glob
import argparse
import random
import shutil
import ipdb
import json

sys.path.append("..")  # noqa
from utils import file_utils
from utils.visualization import get_ee_center_from_pose


def create_info(filepath):
    instance_parts = filepath.split("/")[-3].split("_")

    x, _ = file_utils.load_alive_file(filepath)

    if isinstance(x, dict):
        labels = x["labels"]
        pose = x['pose']
    else:
        (_, _, labels, _, pose) = x

    return {
        "filepath": filepath,
        "position": "_".join(instance_parts[:-1]) if len(instance_parts) > 1 else instance_parts[0],
        "light": instance_parts[-1],
        "arm_point_count": int((labels == 1).sum()),
        "ee_center": get_ee_center_from_pose(pose, switch_w=True).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alivev2")
    parser.add_argument("--infolder", type=str, default="alivev2/")
    parser.add_argument("--out", type=str, default="splits.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument(
        "--ratio",
        nargs="+",
        type=float,
        default=[0.9, 0.05, 0.05],
        help="train, val, test",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    class_folders = glob.glob(os.path.join(args.infolder, "*"))
    class_folders = [cf for cf in class_folders if os.path.isdir(cf)]
    data_types = {"train": list(), "val": list(), "test": list()}

    for cf in class_folders:
        print("Processing:", cf)
        pickles = glob.glob(os.path.join(cf, "labeled", "*.pickle"))

        if args.temporal:
            pickles.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        else:
            random.shuffle(pickles)

        findex = [int(r * len(pickles)) for r in args.ratio]
        findex.insert(0, 0)
        for i in range(1, len(findex)):
            findex[i] += findex[i - 1]
        findex[-1] = len(pickles)

        for i, (_, dt) in enumerate(data_types.items()):
            dt.extend([create_info(pf) for pf in pickles[findex[i] : findex[i + 1]]])

    with open(args.out, "w") as fp:
        json.dump(data_types, fp, indent=2)

    # ipdb.set_trace()
