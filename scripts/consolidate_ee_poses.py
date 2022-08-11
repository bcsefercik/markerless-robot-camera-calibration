import os
import sys
import glob
import pickle
import argparse

sys.path.append("..")  # noqa
from utils import file_utils

import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alive")
    parser.add_argument("--infolder", type=str, default="alive/")
    parser.add_argument("--out", type=str, default="out.pickle")
    args = parser.parse_args()

    ee_poses = list()
    if os.path.isfile(args.out):
        with open(args.out, "rb") as fp:
            ee_poses = pickle.load(fp, encoding="bytes")

    pickles = glob.glob(os.path.join(args.infolder, "labeled", "*.pickle"))

    new_ee_poses = [file_utils.load_alive_file(p)[0]['pose'] for p in pickles]

    ee_poses.extend(new_ee_poses)

    with open(args.out, "wb") as fp:
        pickle.dump(ee_poses, fp)
