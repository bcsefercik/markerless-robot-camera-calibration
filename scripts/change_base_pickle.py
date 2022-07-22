import ipdb

import os
import sys
import pickle
import glob

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transformation import switch_w, transform_pose2pose
from utils import file_utils

if __name__ == "__main__":
    base_pose = np.array(
        [0.0618, 0.0996, 1.4652, -0.3177, -0.6542, -0.6263, 0.2807]
    )  # p2 test

    pickles = glob.glob(os.path.join(sys.argv[1], "*.pickle"))
    for fpath in pickles:
        # fpath = os.path.join(sys.argv[1], f"{i}.pickle")
        # if not os.path.isfile(fpath):
        #     break

        data, semantic_pred = file_utils.load_alive_file(fpath)
        ee2base_pose = data['robot2ee_pose']
        print(i)

        ee2base_pose_w_first = switch_w(ee2base_pose)

        ee_pose_w_first_transformed = transform_pose2pose(base_pose, ee2base_pose_w_first)

        ee_pose_w_last_transformed = np.concatenate((ee_pose_w_first_transformed[:3], ee_pose_w_first_transformed[4:], ee_pose_w_first_transformed[3:4]))

        data['pose'] = ee_pose_w_last_transformed

        with open(fpath, "wb") as fp:
            pickle.dump(data, fp)

    print('done')
