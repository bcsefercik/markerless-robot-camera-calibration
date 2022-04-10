import os
import glob
import argparse
import random
import shutil
import ipdb
import json


def create_info(filepath):
    filename_parts = filepath.split('/')[-1].split('_')

    return {
            'filepath': filepath,
            'position': filename_parts[0],
            'light': filename_parts[1]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alivev1")
    parser.add_argument("--infolder", type=str, default="alivev1/")
    parser.add_argument("--out", type=str, default="alivev1_splits.json")
    args = parser.parse_args()

    class_folders = glob.glob(os.path.join(args.infolder, '*'))
    class_folders = [cf for cf in class_folders if os.path.isdir(cf)]
    data_types = {
        'train': list(),
        'val': list(),
        'test': list()
    }

    for dt in data_types:
        pickles = glob.glob(os.path.join(args.infolder, dt, '*.pickle'))
        pickles = [pf for pf in pickles if not pf.endswith('_semantic.pickle')]
        pickles = [pf for pf in pickles if 'dark' not in pf]
        data_types[dt].extend([create_info(pf) for pf in pickles])

    with open(args.out, 'w') as fp:
        json.dump(data_types, fp, indent=2)

    # ipdb.set_trace()
