import os
import sys
import json
import argparse

import open3d as o3d
import sklearn.preprocessing as preprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils
from utils.data import get_roi_mask
from utils.visualisation import get_frame_from_pose

import ipdb


def fix_filepath(filepath):
    return filepath.replace('/kuacc/users/bsefercik/dataset/', '/Users/bugra.sefercik/workspace/datasets/')
    # return filepath.replace('/datasets/alive_ee/', '/Users/bugra.sefercik/workspace/datasets/')


def save_file(filename, data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)
    print('Saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split alivev2")
    parser.add_argument("--splits", type=str, default="alivev2_splits.json")
    parser.add_argument("--save_freq", type=int, default=16)
    args = parser.parse_args()

    with open(args.splits, 'r') as fp:
        splits = json.load(fp)

    key_to_callback = {ord("S"): lambda vis: save_file(args.splits, splits)}
    roi = {
        "min_x": -0.52,
        "max_x": 0.52,
        "max_y": 0.4,
        "min_z": 0,
        "max_z": 1.2
    }

    new_fields = ('position_eligibility', 'orientation_eligibility')
    # bad_camera_positions = ('p2h1l1', 'p2h1l2', 'p2h2l1', 'p2h2l2', 'p1h3l1', 'p1h3l2', 'p1h2l1', 'p1h2l2', 'p2h3l1', 'p2h3l2')
    bad_camera_positions = tuple()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    for s in splits:
        for i, ins in enumerate(splits[s]):
            if i % 5 != 0:
                continue

            try:
                print(ins['filepath'])

                if sum(1 for k in new_fields if k in ins) == len(new_fields):
                    print('Skipped')
                    continue

                if ins['position'] in bad_camera_positions:
                    print('Bad position, Skipped')
                    continue

                filepath = fix_filepath(ins['filepath'])

                (
                    points,
                    rgb,
                    labels,
                    instance_label,
                    pose,
                ), semantic_pred = file_utils.load_alive_file(filepath)

                splits[s][i]['arm_point_count'] = int((labels == 1).sum())

                if rgb.min() < 0:
                    # WRONG approach, tries to shit from data prep code.
                    rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
                    rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
                    rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

                ee_frame = get_frame_from_pose(frame, pose, switch_w=True)
                kinect_frame = get_frame_from_pose(frame, [0] * 7)

                pcd = o3d.geometry.PointCloud()

                roi_mask = get_roi_mask(points, **roi)
                points = points[roi_mask]
                rgb = rgb[roi_mask]

                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

                o3d.visualization.draw_geometries_with_key_callbacks(
                    [pcd, ee_frame, kinect_frame], key_to_callback
                )

                position_ok = input("Is position OK? [Y/n]: ")
                orientation_ok = input("Is orientation OK? [Y/n]: ")

                splits[s][i]['position_eligibility'] = position_ok.lower() in ('', 'yes', 'y')
                splits[s][i]['orientation_eligibility'] = orientation_ok.lower() in ('', 'yes', 'y')

                if i % args.save_freq == 0:
                    save_file(args.splits, splits)
                    print(f'{s}: %{round((i/len(splits[s])) * 100,1 )} done.')
            except KeyboardInterrupt:
                save_file(args.splits, splits)
                raise KeyboardInterrupt

        save_file(args.splits, splits)

    print("Done!")
