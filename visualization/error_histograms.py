import argparse
import json

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



    ipdb.set_trace()
