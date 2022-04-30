import sys
import os
import copy
import math
import json

import ipdb
import sklearn.preprocessing as preprocessing

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils
from utils.data import get_roi_mask
from utils.visualization import get_frame_from_pose, get_ee_center_from_pose, create_coordinate_frame
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line


if __name__ == "__main__":
    position = "c2_p2_1_full_i5"

    data, semantic_pred = file_utils.load_alive_file(sys.argv[1])

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
    else:
        points, rgb, labels, _, pose = data

    with open(sys.argv[2], 'r') as fp:
        limits = json.load(fp)
    arm_idx = labels == 1

    print('# of points:', len(rgb))
    print('# of arm points:', arm_idx.sum())



    pcd = o3d.geometry.PointCloud()

    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[-0.0103351, -0.0103351, -0.0103351])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # ipdb.set_trace()
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # print('initial frame center', frame.get_center())
    # frame.translate(pose[:3])
    # print('updated frame center', frame.get_center())
    # frame.rotate(frame.get_rotation_matrix_from_quaternion(pose[3:]))
    # print('updated frame center', frame.get_center())

    kinect_frame = get_frame_from_pose(frame, [0] * 7)
    print('ee_position:', pose[:3])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(pose[:3])
    sphere.paint_uniform_color([0.1, 0.1, 0.1])

    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)
    sphere2_diff = rot_mat @ np.array([0.08, 0, 0])
    closest_points_dists, closest_points_idx = select_closest_points_to_line(
        points,
        pose[:3],
        pose[:3] + sphere2_diff,
        count=32,
        cutoff=0.004
    )

    print("closest point count:", len(closest_points_dists))

    rgb[closest_points_idx] = [1, 1, 0]

    lineset_points = [
        pose[:3],
        pose[:3] + sphere2_diff
    ]
    lineset_lines = [[0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lineset_points),
        lines=o3d.utility.Vector2iVector(lineset_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([[0.9, 0.1, 0.1]])
    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere2.translate(pose[:3] + sphere2_diff)
    sphere2.paint_uniform_color([0.1, 0.1, 0.9])

    my_coordinate_frame = create_coordinate_frame(pose)

    # sphere.rotate(frame.get_rotation_matrix_from_quaternion(pose[3:]))
    print('sphere center:', sphere.get_center())



    # ipdb.set_trace()

    roi_mask = get_roi_mask(points, **limits[position])

    points = points[roi_mask]
    rgb = rgb[roi_mask]
    if rgb.min() < 0:
        # WRONG approach, tries to shit from data prep code.
        rgb[:, 0] = preprocessing.minmax_scale(rgb[:, 0], feature_range=(0, 1), axis=0)
        rgb[:, 1] = preprocessing.minmax_scale(rgb[:, 1], feature_range=(0, 1), axis=0)
        rgb[:, 2] = preprocessing.minmax_scale(rgb[:, 2], feature_range=(0, 1), axis=0)

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print('# of masked points:', len(rgb))

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, kinect_frame, line_set, sphere, sphere2], key_to_callback
    )
    # o3d.visualization.draw_geometries_with_key_callbacks(
    #     [pcd, kinect_frame, line_set, my_coordinate_frame], key_to_callback
    # )
