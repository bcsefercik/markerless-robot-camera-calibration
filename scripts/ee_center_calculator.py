import sys
import json

import ipdb

sys.path.append("..")  # noqa
from utils.file_utils import load_alive_file
from utils.visualization import get_ee_center_from_pose


if __name__ == '__main__':
    splits = dict()
    with open(sys.argv[1], "r") as fp:
        splits = json.load(fp)

    for sk in splits:
        for i, ins in enumerate(splits[sk]):
            print("Processing:", ins["filepath"])
            data, semantic_pred = load_alive_file(ins["filepath"])

            if isinstance(data, dict):
                pose = data['pose']
            else:
                points, rgb, labels, _, pose = data

            splits[sk][i]['ee_center'] = get_ee_center_from_pose(pose, switch_w=True).tolist()

    with open(sys.argv[1], "w") as fp:
        json.dump(splits, fp)

    print('Successfully computed.')
