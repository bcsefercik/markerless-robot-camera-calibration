import os
import copy

import numpy as np
import open3d as o3d

from utils.transformation import get_quaternion_rotation_matrix, get_transformation_matrix


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


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


def create_coordinate_frame(pose, length=0.2, radius=0.0075, switch_w=True):
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=switch_w)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 2.2)
    sphere.translate(pose[:3])
    sphere.paint_uniform_color([0.6, 0.6, 0.6])

    z_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    z_cylinder.rotate(rot_mat)
    z_cylinder.translate(pose[:3] + rot_mat @ np.array([0, 0, length / 2]))
    z_cylinder.paint_uniform_color([0.1, 0.1, 0.9])

    y_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    y_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    y_cylinder.translate(pose[:3] + rot_mat @ np.array([0, length / 2, 0]))
    y_cylinder.rotate([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    y_cylinder.rotate(rot_mat)

    x_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    x_cylinder.paint_uniform_color([0.9, 0.1, 0.1])
    x_cylinder.translate(pose[:3] + rot_mat @ np.array([length / 2, 0, 0]))
    x_cylinder.rotate([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    x_cylinder.rotate(rot_mat)

    return sphere + z_cylinder + y_cylinder + x_cylinder


def create_sphere(pose, radius=0.0075):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 2.2)
    sphere.translate(pose[:3])
    sphere.paint_uniform_color([0.9, 0.1, 0.1])

    return sphere


def generate_colors(n):
    return np.random.rand(n, 3)


def get_key_point_colors():
    np.random.seed(5)
    return generate_colors(10)


def generate_key_point_shapes(
    key_points,
    colors=get_key_point_colors(),
    radius=0.01,
    shape='icosahedron',
):
    if shape == 'sphere':
        shape_generator = o3d.geometry.TriangleMesh.create_sphere
    elif shape == 'octahedron':
        shape_generator = o3d.geometry.TriangleMesh.create_octahedron
    else:
        shape_generator = o3d.geometry.TriangleMesh.create_icosahedron

    shapes = o3d.geometry.TriangleMesh()

    for cls, coor in key_points:
        shape = shape_generator(radius=radius)
        shape.translate(coor)
        shape.paint_uniform_color(colors[cls])

        shapes += shape

    return shapes


def get_kinect_mesh(pose, scale=0.001, coordinate_frame_enabled=False):
    tf_mat = get_transformation_matrix(pose)

    kinect_mesh = o3d.io.read_triangle_mesh(
        os.path.join(
            BASE_PATH, "..", "app", "hand_files", "kinect.obj"
        )  # seems to work better
    )
    kinect_mesh.paint_uniform_color(np.array([0.1, 0.1, 0.1]))
    kinect_mesh.scale(scale, np.array([0, 0, 0]))

    k_rot_x = np.matrix([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float64)

    k_rot_y = np.matrix([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ], dtype=np.float64)

    kinect_mesh.rotate(k_rot_y)
    kinect_mesh.rotate(k_rot_x)

    if coordinate_frame_enabled:
        kinect_frame = create_coordinate_frame([-0.00773637, -0.03,  0.01641659, 1, 0, 0, 0], switch_w=False, radius=0.002)
        kinect_mesh += kinect_frame

    kinect_mesh.transform(tf_mat)

    return kinect_mesh
