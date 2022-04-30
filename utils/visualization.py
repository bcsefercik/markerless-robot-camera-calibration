import copy

import numpy as np
import open3d as o3d

from utils.transformation import get_quaternion_rotation_matrix

import ipdb


def get_frame_from_pose(base_frame, pose, switch_w=True):
    frame = copy.deepcopy(base_frame)

    if not isinstance(pose, list):
        pose = pose.tolist()

    ee_position = pose[:3]
    ee_orientation = pose[3:]
    if switch_w:  # move w from last place to first place
        ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

    ee_frame = frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))
    ee_frame = ee_frame.translate(ee_position)

    return ee_frame


def get_ee_center_from_pose(pose, switch_w=True):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    ee_frame = get_frame_from_pose(frame, pose, switch_w=switch_w)

    return ee_frame.get_center()


def create_coordinate_frame(pose, length=0.2, radius=0.0075,  switch_w=True):
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius*2.2)
    sphere.translate(pose[:3])
    sphere.paint_uniform_color([0.6, 0.6, 0.6])

    z_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length
    )
    z_cylinder.rotate(rot_mat)
    z_cylinder.translate(pose[:3] + rot_mat @ np.array([0, 0, length/2]))
    z_cylinder.paint_uniform_color([0.1, 0.1, 0.9])

    y_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length
    )
    y_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    y_cylinder.translate(pose[:3] + rot_mat @ np.array([0, length/2, 0]))
    y_cylinder.rotate([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    y_cylinder.rotate(rot_mat)

    x_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length
    )
    x_cylinder.paint_uniform_color([0.9, 0.1, 0.1])
    x_cylinder.translate(pose[:3] + rot_mat @ np.array([length/2, 0, 0]))
    x_cylinder.rotate([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    x_cylinder.rotate(rot_mat)

    return sphere + z_cylinder + y_cylinder + x_cylinder
