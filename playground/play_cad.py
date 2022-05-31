import open3d as o3d

textured_mesh = o3d.io.read_triangle_mesh("../others/hand_files/hand.obj")

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

kinect_frame = frame.translate([0, 0, 0])
kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))


o3d.visualization.draw_geometries([textured_mesh, kinect_frame])