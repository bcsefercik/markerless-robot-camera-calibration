import os
import glob
import argparse
import random
import shutil
import ipdb
import json
import functools


def filter_folder(path):
    nok_list = [
        'airplane',
        'lego_duplo',
        '_cups',
        '_marbles',
        # 'colored_wood_blocks'
    ]

    result = True

    result = result and (not path.endswith("zip"))
    result = result and os.path.isdir(path)
    result = result and os.path.isdir(os.path.join(path, 'clouds'))
    result = result and functools.reduce(
        lambda a, b: a and b,
        [nl not in path for nl in nok_list]
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split ycb")
    parser.add_argument("--infolder", type=str, default="ycb/")
    parser.add_argument("--out", type=str, default="splits.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ratio", nargs="+", type=float, default=[0.9, 0.05, 0.05], help='train, val, test')
    args = parser.parse_args()

    random.seed(args.seed)

    class_folders = glob.glob(os.path.join(args.infolder, '*'))
    class_folders = [cf for cf in class_folders if filter_folder(cf)]
    data_types = {
        'train': list(),
        'val': list(),
        'test': list()
    }


    for cf in class_folders:
        print('Processing:', cf)
        class_id = int(cf.split('/')[-1].split('_')[0].split('-')[0])
        pickles = glob.glob(os.path.join(cf, 'clouds', '*.pcd'))
        random.shuffle(pickles)

        findex = [int(r * len(pickles)) for r in args.ratio]
        findex.insert(0, 0)
        for i in range(1, len(findex)):
            findex[i] += findex[i - 1]
        findex[-1] = len(pickles)


        for i, (_, dt) in enumerate(data_types.items()):
            dt.extend(
                (class_id, pickle) for pickle in pickles[findex[i]:findex[i + 1]]
            )

    with open(args.out, 'w') as fp:
        json.dump(data_types, fp, indent=2)

    # ipdb.set_trace()
