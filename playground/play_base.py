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
from utils.transformation import get_base2cam_matrix, get_base2cam_pose, get_pose_from_matrix, get_quaternion_rotation_matrix, get_transformation_matrix, get_transformation_matrix_inverse, switch_w
from utils.preprocess import center_at_origin


SCALE = 1

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


if __name__ == "__main__":
    data, semantic_pred = file_utils.load_alive_file(sys.argv[1])

    robot2ee_pose = None
    robot2ee_pose_w_first = None
    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        pose = data['pose']
        robot2ee_pose = data.get('robot2ee_pose')
        robot2ee_pose_w_first = switch_w(robot2ee_pose)
    else:
        points, rgb, labels, _, pose = data
        points = np.array(points, dtype=np.float32)
        rgb = np.array(rgb, dtype=np.float32)
        pose = np.array(pose, dtype=np.float32)

    pose_w_first = switch_w(pose)

    # 0.514, -0.887, 0.954), (-0.22849161094160622, 0.22759203767018255, 0.6575880614106856, 0.6808607710894938))

    # pose[:3] = pose[:3] * SCALE

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

    ee_idx = get_ee_idx(
        points,
        pose,
        switch_w=True,
        # ee_dim={
        #     'min_z': -0.15,
        #     'max_z': 0.12,
        #     'min_x': -0.1,
        #     'max_x': 0.1,
        #     'min_y': -0.11,
        #     'max_y': 0.11
        # }
    ) # switch_w=False in dataloader
    rot_mat = get_quaternion_rotation_matrix(pose[3:], switch_w=True)  # switch_w=False in dataloader
    ee_points = points[ee_idx] * SCALE
    ee_rgb = rgb[ee_idx] * 0

    pcd_ee = o3d.geometry.PointCloud()
    pcd_ee.points = o3d.utility.Vector3dVector(ee_points)
    pcd_ee.colors = o3d.utility.Vector3dVector(ee_rgb)
    # must do this for point to plane icp

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

    textured_mesh = o3d.io.read_triangle_mesh("../app/hand_files/hand.obj")
    textured_mesh.paint_uniform_color([1, 0.01, 0])
    # textured_mesh = o3d.io.read_triangle_mesh("../app/hand_files/hand_notblender.obj")

    pcd_cad = textured_mesh.sample_points_uniformly(number_of_points=8192*2)  # has normal since converted from mesh
    pcd_cad = textured_mesh.sample_points_poisson_disk(number_of_points=4096*2, pcl=pcd_cad)

    pcd_cad_points = np.asarray(pcd_cad.points) * SCALE
    pcd_cad_colors = np.asarray(pcd_cad.colors)
    pcd_cad_normals = np.asarray(pcd_cad.normals)
    pcd_cad_mask = (pcd_cad_points[:, 0] > 0.0) # * (pcd_cad_points[:, 2] > -0.0)
    pcd_cad.points = o3d.utility.Vector3dVector(pcd_cad_points[pcd_cad_mask])
    pcd_cad.colors = o3d.utility.Vector3dVector(pcd_cad_colors[pcd_cad_mask])
    pcd_cad.normals = o3d.utility.Vector3dVector(pcd_cad_normals[pcd_cad_mask])

    pcd_ee.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd_cad.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # ipdb.set_trace()

    jiggle = (np.random.rand(7) * 2 - 1) * 0.1
    pose_jiggled = pose + jiggle
    # pose_jiggled = pose
    trans_mat_jiggled = get_transformation_matrix(pose_jiggled, switch_w=True)  # switch_w=False in dataloader
    print(jiggle)
    # pcd_cad.transform(trans_mat_jiggled)

    threshold = 0.1
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd_cad, pcd_ee, threshold, trans_mat_jiggled,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation, "\n")

    # pcd_cad.transform(reg_p2l.transformation)
    # textured_mesh.transform(reg_p2l.transformation)
    # textured_mesh.transform(trans_mat_jiggled)
    # pcd_cad.transform(trans_mat_jiggled)
    # pcd_cad.transform(gt_trans_mat)

    pred_pose = get_pose_from_matrix(reg_p2l.transformation)

    pred_frame = create_coordinate_frame(pred_pose, switch_w=False)

    robot2ee_trans_mat = get_transformation_matrix(robot2ee_pose, switch_w=True)
    robot2ee_trans_mat_inv = get_transformation_matrix_inverse(robot2ee_trans_mat)

    # ee_trans_mat = gt_trans_mat
    ee_trans_mat = reg_p2l.transformation
    ee_trans_mat_inv = get_transformation_matrix_inverse(ee_trans_mat)

    robot2ee_pose = get_pose_from_matrix(robot2ee_trans_mat)
    base_frame = copy.deepcopy(ee_frame)
    base_frame_pred = copy.deepcopy(pred_frame)

    kinect2base_trans = ee_trans_mat @ robot2ee_trans_mat_inv
    kinect2base_pose = get_pose_from_matrix(kinect2base_trans)
    print(kinect2base_pose)
    print("GT base2cam:", "0.665 0.404 0.992 0.648 0.289 0.287 -0.644")
    # 0.665 0.404 0.992 0.2886047700164364 0.2870696382610298 -0.643987771393059 0.6477484541152086

    base2kinect_trans_gt = get_base2cam_matrix(pose_w_first, robot2ee_pose_w_first)
    base2kinect_pose_gt = get_pose_from_matrix(base2kinect_trans_gt)
    print("GT transformed base2cam:", base2kinect_pose_gt.tolist())

    base2kinect_trans = get_base2cam_matrix(pred_pose, robot2ee_pose_w_first)
    base2kinect_pose = get_pose_from_matrix(base2kinect_trans)
    print("PRED transformed base2cam:", base2kinect_pose.tolist())



    # base2kinect_trans = get_transformation_matrix(base2kinect_pose, switch_w=False)
    kinect_frame.transform(kinect2base_trans)
    # ipdb.set_trace()

    o3d.visualization.draw_geometries(
        # [pcd, pcd_cad, kinect_frame, ee_frame, textured_mesh]
        [pcd, pcd_cad, kinect_frame, pred_frame, textured_mesh]
        # [pcd, pcd_cad, kinect_frame, pred_frame, textured_mesh]
        # [pcd, pcd_cad, kinect_frame, ee_frame, textured_mesh]
        # [pcd, pcd_ee, pcd_cad, kinect_frame, pred_frame, textured_mesh]
        # [pcd_ee, pcd_cad, kinect_frame, pred_frame, textured_mesh]
        # [pcd_ee, pcd_cad, kinect_frame, ee_frame, pred_frame, textured_mesh]
        # [pcd, pcd_ee, pcd_cad, kinect_frame, ee_frame]
    )
