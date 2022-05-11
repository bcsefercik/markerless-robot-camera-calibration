import pickle
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
from utils.data import get_ee_cross_section_idx, get_ee_idx, get_key_points, get_roi_mask
from utils.visualization import create_coordinate_frame, generate_colors
from utils.transformation import get_quaternion_rotation_matrix, select_closest_points_to_line
from utils.preprocess import center_at_origin


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

if __name__ == "__main__":

    np.random.seed(13)
    key_point_colors = generate_colors(10)

    (
        points,
        rgb,
        labels,
        instance_label,
        pose,
    ), semantic_pred = file_utils.load_alive_file(sys.argv[1])

    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

    arm_idx = labels == 1

    print('# of points:', len(rgb))
    print('# of arm points:', arm_idx.sum())

    points = points[arm_idx]
    rgb = rgb[arm_idx]
    labels = [arm_idx]

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    kinect_frame = copy.deepcopy(frame).translate([0, 0, 0])
    kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

    # kinect_frame = create_coordinate_frame([0] * 7, switch_w=False)
    ee_frame = create_coordinate_frame(pose, switch_w=True)



    # ee_points = points - pose[:3]
    # ee_rgb = rgb

    # new_ee_points = ee_points - pose[:3]
    #
    ee_idx = get_ee_idx(points, pose, switch_w=True) # switch_w=False in dataloader



    ee_rgb = rgb[ee_idx]
    ee_rgb = np.zeros_like(ee_rgb) + np.array([0.3, 0.3, 0.3])
    rgb[ee_idx] = np.array([1.0, 1.0, 0.13])

    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)  # switch_w=False in dataloader
    ee_points = points[ee_idx]

    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))

    # new_ee_points, origin_offset = base_at_origin(new_ee_points)
    # new_ee_points, y_offset = center_at_y(new_ee_points)
    new_ee_pose_pos = new_ee_points[-1:]
    new_ee_points = new_ee_points[:-1]
    new_ee_pose_points, ee_pose_offset = center_at_origin(new_ee_pose_pos)
    new_ee_points -= ee_pose_offset
    # ipdb.set_trace()

    key_points = get_key_points(ee_points, pose, switch_w=True)


    spheres = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    spheres.translate(key_points[0])
    spheres.paint_uniform_color(key_point_colors[0])

    for i in range(1, len(key_points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(key_points[i])
        sphere.paint_uniform_color(key_point_colors[i])

        spheres += sphere

    # # reverse the transformation above
    # new_ee_points_reverse = np.array(new_ee_points, copy=True)
    # new_ee_points_reverse += origin_offset
    # new_ee_points_reverse = (rot_mat @ np.concatenate((new_ee_points_reverse, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    # new_ee_points_reverse = new_ee_points_reverse[:-1]
    # ee_reverse_rgb = np.zeros_like(ee_rgb) + np.array([1.0, 0.13, 1.0])


    # _, min_y, min_z = new_ee_points.min(axis=0)
    # max_y = new_ee_points.max(axis=0)[1]

    # ee_pos_magic = np.array([-0.01, 0.0, min_z])
    # ee_pos_magic_reverse = ee_pos_magic + origin_offset
    # ee_pos_magic_reverse = rot_mat @ ee_pos_magic_reverse

    # ee_pos_magic_reverse_pose = ee_pos_magic_reverse.tolist() + ee_orientation
    # # ee_pos_magic_reverse_pose = new_ee_pose_pos.tolist() + [0] * 4
    # ee_magic_frame = create_coordinate_frame(ee_pos_magic_reverse_pose, switch_w=False)

    # ipdb.set_trace()
    points_show = np.concatenate((points, new_ee_points), axis=0)
    rgb = np.concatenate((rgb, ee_rgb), axis=0)
    # points_show = new_ee_points
    # rgb = ee_rgb


    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # pick_points(pcd)

    print('# of masked points:', len(rgb))
    o3d.visualization.draw_geometries(
        # [pcd, kinect_frame, ee_frame]
        [pcd, kinect_frame, spheres]
        # [pcd, ee_magic_frame, ee_frame]
        # [pcd]
    )


# import numpy as np
# import copy
# import open3d as o3d


# def demo_crop_geometry():
#     print("Demo for manual geometry cropping")
#     print(
#         "1) Press 'Y' twice to align geometry with negative direction of y-axis"
#     )
#     print("2) Press 'K' to lock screen and to switch to selection mode")
#     print("3) Drag for rectangle selection,")
#     print("   or use ctrl + left click for polygon selection")
#     print("4) Press 'C' to get a selected geometry and to save it")
#     print("5) Press 'F' to switch to freeview mode")
#     pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
#     o3d.visualization.draw_geometries_with_editing([pcd])


# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])



# def demo_manual_registration():
#     print("Demo for manual ICP")
#     source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
#     target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_2.pcd")
#     print("Visualization of two point clouds before manual alignment")
#     draw_registration_result(source, target, np.identity(4))

#     # pick points from two point clouds and builds correspondences
#     picked_id_source = pick_points(source)
#     picked_id_target = pick_points(target)
#     assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
#     assert (len(picked_id_source) == len(picked_id_target))
#     corr = np.zeros((len(picked_id_source), 2))
#     corr[:, 0] = picked_id_source
#     corr[:, 1] = picked_id_target

#     # estimate rough transformation using correspondences
#     print("Compute a rough transform using the correspondences given by user")
#     p2p = o3d.registration.TransformationEstimationPointToPoint()
#     trans_init = p2p.compute_transformation(source, target,
#                                             o3d.utility.Vector2iVector(corr))

#     # point-to-point ICP for refinement
#     print("Perform point-to-point ICP refinement")
#     threshold = 0.03  # 3cm distance threshold
#     reg_p2p = o3d.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.registration.TransformationEstimationPointToPoint())
#     draw_registration_result(source, target, reg_p2p.transformation)
#     print("")


# if __name__ == "__main__":
#     demo_crop_geometry()
#     demo_manual_registration()