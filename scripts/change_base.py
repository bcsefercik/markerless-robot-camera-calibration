import ipdb

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transformation import switch_w, transform_pose2pose


if __name__ == "__main__":
    base_pose = base_pose = np.array(
        [
            -0.66030359,
            0.1169908,
            1.50719007,
            -0.47416406,
            -0.88011599,
            -0.01268632,
            -0.02008359,
        ]
    )  # w first

    i = 1

    while True:
        print(i)
        ee_pose_file = os.path.join(sys.argv[1], f"{i}.npy")
        ee2base_pose_file = os.path.join(sys.argv[1], f"{i}_robot2ee_pose.npy")

        if not os.path.isfile(ee_pose_file) or not os.path.isfile(ee2base_pose_file):
            break

        ee2base_pose = np.load(ee2base_pose_file, allow_pickle=True)

        ee2base_pose_w_first = switch_w(ee2base_pose)

        ee_pose_w_first_transformed = transform_pose2pose(base_pose, ee2base_pose_w_first)

        ee_pose_w_last_transformed = np.concatenate((ee_pose_w_first_transformed[:3], ee_pose_w_first_transformed[4:], ee_pose_w_first_transformed[3:4]))

        np.save(ee_pose_file, ee_pose_w_last_transformed)
        # ipdb.set_trace()

        i += 1

    print('done')
