import glob
import json
import os
import sys

import ipdb


if __name__ == '__main__':
    split_files = glob.glob(os.path.join(sys.argv[1], "*splits*.json"))

    for sf in split_files:
        print(sf)
        with open(sf, 'r') as fp:
            splits = json.load(fp)

        for ss in splits:
            for i, _ in enumerate(splits[ss]):
                if isinstance(splits[ss][i], dict):
                    splits[ss][i].pop('ee_center', None)

        with open(sf, 'w') as fp:
            json.dump(splits, fp, indent=4)
