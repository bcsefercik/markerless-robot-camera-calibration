import ipdb
import os
import sys

import cv2
import open3d as o3d
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import transformation


_device = o3d.core.Device("CPU:0")
_dtype = o3d.core.float32

_camera_matrix_default = np.array(
    [
        [520.342706004118, 0, 323.0580496437712],
        [0, 513.826209565285, 263.4994539787398],
        [0, 0, 1],
    ]
)  # kinect1 rgb

_distortion_coeff_default = np.array(
    [
        0.1594658601746339,
        -0.283618200726369,
        -0.003065915455824548,
        0.003289899753081607,
        0,
    ]
)  # kinect1 rgb


def compute_ee_pose(
    points,
    rgb,
    camera_matrix=_camera_matrix_default,
    image_width=640,
    image_height=480,
    aruco_key=cv2.aruco.DICT_6X6_1000,
    aruco_tag_size=0.075,
    t_tag2ee=np.array([-0.012, -0.0, -0.05]),
):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(points, _dtype, _device)
    pcd.point["colors"] = o3d.core.Tensor(rgb, _dtype, _device)

    intrinsic_tensor = o3d.core.Tensor(camera_matrix)

    depth_scale = 1000.0
    # TODO: get it from matrix
    fx = 520.342706004118
    cx = 323.0580496437712
    fy = 513.826209565285
    cy = 263.4994539787398
    rgbd_reproj = pcd.project_to_rgbd_image(
        image_width,
        image_height,
        intrinsic_tensor,
        depth_scale=depth_scale,
        depth_max=4.0,
    )

    rgb_img = np.asarray(rgbd_reproj.color.to_legacy()) * 255
    depth_img = np.asarray(rgbd_reproj.depth.to_legacy())

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype("uint8")

    aruco_dict = cv2.aruco.Dictionary_get(aruco_key)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray_img, aruco_dict, parameters=parameters
    )

    if len(corners) != 1:
        return None

    corners_3d = list()
    for i in range(4):
        u, v = tuple(corners[0][0][i])
        u = int(u)
        v = int(v)

        d = depth_img[v, u]
        z = d / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        tvec = np.array([x, y, z])
        corners_3d.append(tvec)

    corners_3d = np.asarray(corners_3d, dtype=np.float32)

    corners_3d_ref = np.array(
        [
            [0, aruco_tag_size / 2, -aruco_tag_size / 2],
            [0, -aruco_tag_size / 2, -aruco_tag_size / 2],
            [0, -aruco_tag_size / 2, aruco_tag_size / 2],
            [0, aruco_tag_size / 2, aruco_tag_size / 2],
        ],
        dtype=np.float32,
    )

    R, tvec = transformation.get_rigid_transform_3D(corners_3d_ref, corners_3d)
    tvec = tvec + (R @ t_tag2ee)

    qvec = transformation.get_q_from_matrix(R)

    pose = np.concatenate((tvec, qvec))

    return pose
