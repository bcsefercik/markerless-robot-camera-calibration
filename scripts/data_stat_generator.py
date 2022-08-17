import os
import statistics
import sys
import glob
import pickle
import argparse
import random

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")  # noqa
from utils import file_utils
from utils.visualization import create_coordinate_frame, get_kinect_mesh
from utils.data import get_ee_idx, get_roi_mask
from utils.transformation import get_pose_inverse, get_transformation_matrix, switch_w, transform_pose2pose


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="viz test data")
    parser.add_argument("--datasets", type=str, nargs="+")
    args = parser.parse_args()

    ee_dists = list()
    base_dists = list()
    ee_point_counts = list()
    arm_point_counts = list()

    instance_count = 0

    for cf in args.datasets:
        print("Processing:", cf)
        pickles = glob.glob(os.path.join(cf, "labeled", "*.pickle"))

        instance_count += len(pickles)

        for sp in pickles:
            data, _ = file_utils.load_alive_file(sp)

            points = data['points']
            rgb = data['rgb']
            pose = data['pose']
            pose = switch_w(pose)
            labels = data['labels']
            robot2ee_pose = data['robot2ee_pose']
            ee2robot_pose = get_pose_inverse(switch_w(robot2ee_pose))
            cam2base = transform_pose2pose(pose, ee2robot_pose)

            ee_dists.append(round(np.linalg.norm(pose[:3]), 4))
            base_dists.append(round(np.linalg.norm(cam2base[:3]), 2))

            arm_idx = np.where(labels == 1)[0]
            ee_idx = np.where(labels == 2)[0]
            if len(ee_idx) < 1:
                ee_idx = get_ee_idx(points, pose, switch_w=False, arm_idx=arm_idx, ee_dim={
                    'min_z': -0.0,
                    'max_z': 0.15,
                    'min_x': -0.08,
                    'max_x': 0.08,
                    'min_y': -0.15,
                    'max_y': 0.15
                })
                labels[ee_idx] = 2

            ee_point_counts.append(len(ee_idx))
            arm_point_counts.append(len(arm_idx))

    ee_dists_mean = round(statistics.mean(ee_dists), 2)
    ee_dists_std = round(statistics.stdev(ee_dists), 2)
    ee_dists_min = min(ee_dists)
    ee_dists_max = max(ee_dists)

    ee_point_counts_mean = int(statistics.mean(ee_point_counts))
    ee_point_counts_std = int(statistics.stdev(ee_point_counts))
    ee_point_counts_min = min(ee_point_counts)
    ee_point_counts_max = max(ee_point_counts)

    arm_point_counts_mean = int(statistics.mean(arm_point_counts))
    arm_point_counts_std = int(statistics.stdev(arm_point_counts))
    arm_point_counts_min = min(arm_point_counts)
    arm_point_counts_max = max(arm_point_counts)

    unique_base_dists = set(base_dists)
    unique_base_dists = [str(round(ubd, 2)) for ubd in unique_base_dists]

    print('\nEE Distances')
    print(f'Mean:\t\t{ee_dists_mean:.2f} m')
    print(f'Min:\t\t{ee_dists_min:.2f} m')
    print(f'Max:\t\t{ee_dists_max:.2f} m')
    print(f'Std:\t\t{ee_dists_std:.2f} m')

    print('\nEE Point Counts')
    print(f'Mean:\t\t{ee_point_counts_mean}')
    print(f'Min:\t\t{ee_point_counts_min}')
    print(f'Max:\t\t{ee_point_counts_max}')
    print(f'Std:\t\t{ee_point_counts_std}')

    print(f'\n# frames:\t{instance_count}')
    print(f'Base Distances:\t{", ".join(unique_base_dists)} (m)')


    # plt.hist(ee_dists, density=True, bins=32)  # density=False would make counts
    # plt.ylabel('%')
    # plt.xlabel('EE Dists (m)')
    # plt.grid()
    # plt.show()

    # plt.hist(ee_point_counts, density=True, bins=32)  # density=False would make counts
    # plt.ylabel('%')
    # plt.xlabel('EE Point Counts')
    # plt.grid()
    # plt.show()

    # ipdb.set_trace()
