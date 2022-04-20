import argparse
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np

import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram of errors.")
    parser.add_argument("--results", type=str, default="results.json")
    args = parser.parse_args()

    with open(args.results, 'r') as fp:
        results = json.load(fp)

    series = {
        "dist_position": list(),
        "dist_orientation": list(),
        "angle_diff": list(),
        "dist": list(),
        "position_confidence": list(),
        "orientation_confidence": list(),
        "confidence": list()
    }

    for rk, r in results.items():
        if not rk.startswith('c1'):
            continue
        for sk in series:
            series[sk].append(r[sk])

    pairs = (
        ('position_confidence', 'dist_position'),
        ('orientation_confidence', 'dist_orientation'),
        ('orientation_confidence', 'angle_diff'),
        ('confidence', 'dist')
    )

    for i, (xk, yk) in enumerate(pairs):
        x = series[xk].copy()
        y = series[yk].copy()
        x, y = zip(*sorted(zip(x, y)))
        plt.subplot(2, 2, i+1)
        plt.plot(x, y, 'b.')
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--")
        plt.xlabel(xk)
        plt.ylabel(yk)
        plt.grid(True)
    plt.show()
    # for i, c in enumerate(categories):
    #     plt.subplot(int(f'1{len(categories)}{i+1}'))
    #     plt.plot(bins, plots[c])
    #     # long_y = [res[c] for k, res in results.items()]
    #     # lon_y = np.array(long_y)[order]
    #     # plt.plot(long_x, long_y)
    #     plt.xlabel('# arm points')
    #     plt.ylabel('metric')
    #     plt.title(c)
    #     plt.grid(True)


    # ipdb.set_trace()
