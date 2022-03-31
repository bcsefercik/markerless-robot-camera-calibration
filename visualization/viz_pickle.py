import sys
import os
import copy

import ipdb

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils


if __name__ == "__main__":
    (
        points,
        rgb,
        labels,
        instance_label,
        pose,
    ), semantic_pred = file_utils.load_alive_file(sys.argv[1])

    arm_idx = labels == 1

    min_point = points[arm_idx].min(axis=0)
    max_point = points[arm_idx].max(axis=0)

    # ipdb.set_trace()

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        np.array(min_point).reshape((3, 1)),
        np.array(max_point).reshape((3, 1)),
    )
    obbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
    obbox.color = [1, 0, 0]

    # points = points[arm_idx]
    # rgb = rgb[arm_idx]
    # labels = [arm_idx]

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]
    ee_frame = copy.deepcopy(frame).translate(ee_position)
    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))
    kinect_frame = copy.deepcopy(frame).translate([0, 0, 0])
    kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

    # obbox.translate(ee_position).rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    arm_mask = points[:, 0] > -500
    arm_mask = np.logical_and(points <= max_point, points >= min_point).sum(axis=1) == 3
    arm_mask = np.logical_and(points[:, 0] < 0.5, arm_mask)  # x
    arm_mask = np.logical_and(points[:, 0] > -0.5, arm_mask)
    arm_mask = np.logical_and(points[:, 1] < 0.27, arm_mask)  # y
    # # arm_mask = np.logical_and(points[:, 1] > 0.2, arm_mask)
    arm_mask = np.logical_and(points[:, 2] < 1.3, arm_mask)  # z
    # # # arm_mask = np.logical_and(points[:, 2] > 1.5, arm_mask)

    points = points[arm_mask]
    rgb = rgb[arm_mask]

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    key_to_callback = {ord("K"): switch_to_normal}
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd, ee_frame, kinect_frame, obbox], key_to_callback
    )
