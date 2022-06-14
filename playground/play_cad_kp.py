import ipdb

import sys
import os

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import create_coordinate_frame, generate_colors, generate_key_point_shapes, get_key_point_colors
from utils.data import collect_closest_points, get_6_key_points, get_ee_cross_section_idx, get_ee_idx, get_key_points, get_roi_mask


textured_mesh = o3d.io.read_triangle_mesh("../app/hand_files/hand.obj")

pcd = textured_mesh.sample_points_uniformly(number_of_points=20000)
pcd = textured_mesh.sample_points_poisson_disk(number_of_points=12000, pcl=pcd)
# ipdb.set_trace()

pcd =  o3d.io.read_point_cloud("../app/hand_files/hand.obj")

pcd_points = np.asarray(pcd.points)
pcd_points = pcd_points[pcd_points[:, 0] > 0.01]
pcd_points = pcd_points[pcd_points[:, 2] > -0.01]

pcd.points = o3d.utility.Vector3dVector(pcd_points)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
kinect_frame = frame.translate([0, 0, 0])
kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

ref_key_points, ref_p_idx = get_6_key_points(pcd_points, np.array([0, 0, 0, 1, 0, 0, 0]), switch_w=False)

ref_shapes = generate_key_point_shapes(
    list(zip(list(range(len(ref_p_idx))), ref_key_points)),
    radius=0.016,
    shape='octahedron'
)

o3d.visualization.draw_geometries([kinect_frame, ref_shapes, pcd])