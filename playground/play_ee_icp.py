import ipdb

import sys
import os
import copy

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils
from utils.data import get_ee_cross_section_idx, get_ee_idx, get_roi_mask
from utils.visualization import create_coordinate_frame
from utils.transformation import get_quaternion_rotation_matrix, get_transformation_matrix
from utils.preprocess import center_at_origin


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


if __name__ == "__main__":
    data, semantic_pred = file_utils.load_alive_file(sys.argv[1])

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
    else:
        points, rgb, labels, _, pose = data
        points = np.array(points, dtype=np.float32)
        rgb = np.array(rgb, dtype=np.float32)
        pose = np.array(pose, dtype=np.float32)

    ee_position = pose[:3]
    ee_orientation = pose[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]
    gt_trans_mat = get_transformation_matrix(pose, switch_w=True)  # switch_w=False in dataloader

    arm_idx = labels == 1

    pcd = o3d.geometry.PointCloud()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    kinect_frame = copy.deepcopy(frame).translate([0, 0, 0])
    kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

    # kinect_frame = create_coordinate_frame([0] * 7, switch_w=False)
    ee_frame = create_coordinate_frame(pose, switch_w=True)

    ee_idx = get_ee_idx(points, pose, switch_w=True) # switch_w=False in dataloader
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)  # switch_w=False in dataloader
    ee_points = points[ee_idx]
    ee_rgb = rgb[ee_idx] * 0

    pcd_ee = o3d.geometry.PointCloud()
    pcd_ee.points = o3d.utility.Vector3dVector(ee_points)
    pcd_ee.colors = o3d.utility.Vector3dVector(ee_rgb)
    # must do this for point to plane icp
    # pcd_ee.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5))

    new_ee_points = (rot_mat.T @ np.concatenate((ee_points, pose[:3].reshape(1, 3))).reshape((-1, 3, 1))).reshape((-1, 3))
    new_ee_points, origin_offset = center_at_origin(new_ee_points)
    new_ee_pose_pos = new_ee_points[-1]
    new_ee_points = new_ee_points[:-1]

    # points_show = np.concatenate((points, new_ee_points), axis=0)
    # rgb_show = np.concatenate((rgb, ee_rgb), axis=0)

    points_show = points
    rgb_show = rgb

    pcd.points = o3d.utility.Vector3dVector(points_show)
    pcd.colors = o3d.utility.Vector3dVector(rgb_show)

    textured_mesh = o3d.io.read_triangle_mesh("../others/hand_files/hand.obj")
    # textured_mesh = o3d.io.read_triangle_mesh("../others/hand_files/hand_notblender.obj")

    pcd_cad = textured_mesh.sample_points_uniformly(number_of_points=8192*2)  # has normal since converted from mesh
    pcd_cad = textured_mesh.sample_points_poisson_disk(number_of_points=4096*2, pcl=pcd_cad)

    pcd_cad_points = np.asarray(pcd_cad.points)
    pcd_cad_normals = np.asarray(pcd_cad.normals)
    pcd_cad_mask = (pcd_cad_points[:, 0] > 0.0) * (pcd_cad_points[:, 2] > -0.0)
    pcd_cad.points = o3d.utility.Vector3dVector(pcd_cad_points[pcd_cad_mask])
    pcd_cad.normals = o3d.utility.Vector3dVector(pcd_cad_normals[pcd_cad_mask])
    # ipdb.set_trace()


    pose_jiggled = pose + (np.random.rand(7) * 2 - 1) * 0.01
    trans_mat_jiggled = get_transformation_matrix(pose_jiggled, switch_w=True)  # switch_w=False in dataloader

    # pcd_cad.transform(trans_mat_jiggled)


    mu, sigma = 0, 0.1
    source_noisy = apply_noise(pcd_cad, mu, sigma)

    threshold = 0.2
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)

    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd_cad, pcd_ee, threshold, trans_mat_jiggled,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    pcd_cad.transform(reg_p2l.transformation)
    # pcd_cad.transform(trans_mat_jiggled)
    # pcd_cad.transform(gt_trans_mat)
    # ipdb.set_trace()

    o3d.visualization.draw_geometries(
        [pcd_ee, pcd_cad, kinect_frame, ee_frame]
        # [pcd, pcd_ee, pcd_cad, kinect_frame, ee_frame]
    )

