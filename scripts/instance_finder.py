import os
import sys
import glob
import argparse
import shutil


import numpy as np

sys.path.append("..")  # noqa
from utils import file_utils
from utils.transformation import switch_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find instances for test/calib set")
    parser.add_argument("--infolder", type=str)
    parser.add_argument("--outfolder", type=str, default="fold/")
    args = parser.parse_args()

    pickles = glob.glob(os.path.join(args.infolder, "*.pickle"))
    pickles.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    last_instance_pose = np.array([0, 0, 0, 1, 0, 0, 0])
    last_instance_id = 0

    i = 0
    while i < (len(pickles) - 10):
        data, _ = file_utils.load_alive_file(pickles[i])
        ee2base_pose = data.get("robot2ee_pose")
        ee2base_pose = switch_w(ee2base_pose)

        data_end, _ = file_utils.load_alive_file(pickles[i + 9])
        ee2base_pose_end = data_end.get("robot2ee_pose")
        ee2base_pose_end = switch_w(ee2base_pose_end)

        dist_last = np.linalg.norm(last_instance_pose - ee2base_pose)
        dist_e2e = np.linalg.norm(ee2base_pose - ee2base_pose_end)

        if dist_last > 0.1 and dist_e2e < 0.05:
            last_instance_pose = ee2base_pose
            last_instance_id += 1
            start = i + 1
            end = start + 9
            print(f'i{last_instance_id}: {start} - {end}')

            i += 9

            out_folder = os.path.join(args.outfolder, f'i{last_instance_id}')
            os.mkdir(out_folder)
            out_folder = os.path.join(out_folder, 'labeled')
            os.mkdir(out_folder)

            for j in range(i, i + 10):
                shutil.copyfile(pickles[j], os.path.join(out_folder, f'{j+1}.pickle'))

        i += 1
