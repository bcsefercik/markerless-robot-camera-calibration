import argparse
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np

import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram of errors.")
    parser.add_argument("--splits", type=str, default="splits.json")
    parser.add_argument("--results", type=str, default="results.json")
    args = parser.parse_args()

    with open(args.splits, 'r') as fp:
        splits_raw = json.load(fp)

    splits = dict()
    for k, split in splits_raw.items():
        splits.update({f'{s["position"]}/{s["filepath"].split("/")[-1]}': s for s in split})

    with open(args.results, 'r') as fp:
        results = json.load(fp)

    bins = (1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000)

    categories = ('dist_position', 'dist_orientation', 'angle_diff')

    category_results = {
        c: {b: list() for b in bins} for c in categories
    }

    for kr, result in results.items():
        result_bin = [b for b in bins if b - splits[kr]['arm_point_count'] > 0]
        result_bin = min(result_bin) if result_bin else bins[-1]

        for c in categories:
            category_results[c][result_bin].append(result[c])

    plots = dict()

    for c in categories:
        plots[c] = [statistics.mean(l) if l else 0 for l in category_results[c].values()]

    # plt.figure(figsize=(9, 3))

    long_x = [splits[k]['arm_point_count'] for k, _ in results.items()]
    order = np.argsort(long_x)
    long_x = np.array(long_x)[order]

    for i, c in enumerate(categories):
        plt.subplot(int(f'1{len(categories)}{i+1}'))
        plt.plot(bins, plots[c])
        # long_y = [res[c] for k, res in results.items()]
        # lon_y = np.array(long_y)[order]
        # plt.plot(long_x, long_y)
        plt.xlabel('# arm points')
        plt.ylabel('metric')
        plt.title(c)
        plt.grid(True)
    # plt.title('dist_position')
    # plt.subplot(2)
    # plt.plot(bins, plots['dist_orientation'])
    # plt.title('dist_orientation')
    # plt.subplot(3)
    # plt.plot(bins, plots['angle_diff'])
    # plt.title('angle_diff')
    plt.grid(True)
    plt.show()

    # ipdb.set_trace()
