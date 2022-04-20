import sys
import argparse
import ipdb
import json
import pickle

import numpy as np
import open3d as o3d

sys.path.append("..")  # noqa
from utils import config, file_utils
from data.alivev2 import AliveV2Dataset


_config = config.Config()


if __name__ == "__main__":
    with open(_config.EEMASK.splits, 'r') as fp:
        splits = json.load(fp)

    for k, cf in splits.items():
        pickles = [ins for ins in cf if AliveV2Dataset.filter_file(ins['filepath'])]
        for p in pickles:
            (points, rgb, labels, _, pose), _ = file_utils.load_alive_file(p['filepath'])
            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
            ee_position = pose[:3]
            ee_orientation = pose[3:].tolist()
            ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

            offset = frame.get_rotation_matrix_from_quaternion(ee_orientation) @ np.array([0, 0, 0.03])
            obbox = o3d.geometry.OrientedBoundingBox(
                ee_position,
                frame.get_rotation_matrix_from_quaternion(ee_orientation),
                np.array([0.15, 0.27, 0.18])
            )
            obbox.color = [1, 0, 0]
            obbox.center = obbox.get_center() + offset

            eemask = obbox.get_point_indices_within_bounding_box(pcd.points)

            with open(p['filepath'].replace('.pickle', '_eemask.pickle'), 'wb') as fp:
                pickle.dump(eemask, fp)

            print(p['filepath'])
            # ipdb.set_trace()

    print('All done.')

    # ipdb.set_trace()
