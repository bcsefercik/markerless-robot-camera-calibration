import ipdb

import numpy as np
import open3d as o3d

textured_mesh = o3d.io.read_triangle_mesh("../app/hand_files/hand.obj")

pcd = textured_mesh.sample_points_uniformly(number_of_points=8192)
pcd = textured_mesh.sample_points_poisson_disk(number_of_points=4096, pcl=pcd)
# ipdb.set_trace()


pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + 0.15)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
kinect_frame = frame.translate([0, 0, 0])
kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))


o3d.visualization.draw_geometries([textured_mesh, kinect_frame , pcd])