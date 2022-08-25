import os
import sys

import ipdb
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils, icp, transformation
from utils.quaternion import euler_to_quaternion
from utils.visualization import create_coordinate_frame, create_sphere


def pose_esitmation(
    frame, aruco_dict_type, matrix_coefficients, distortion_coefficients
):

    """
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:-
    frame - The frame with the axis drawn on it
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("uint8")

    # gray = cv2.flip(gray, 0)
    # ipdb.set_trace()
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    # corners = rejected_img_points
    # ids = [12] * len(corners)
    # ipdb.set_trace()
    print(len(corners))

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.075, matrix_coefficients, distortion_coefficients
            )
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.drawFrameAxes(
                frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.05
            )

    # top_left, top_right, bottom_right, bottom_left
    tcoor = np.asarray(corners[0][0], dtype=np.int32)

    return frame, rvec, tcoor


if __name__ == "__main__":
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32

    if sys.argv[1].endswith("pickle"):
        data, semantic_pred = file_utils.load_alive_file(sys.argv[1])
        points = data["points"]
        rgb = data["rgb"]

        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(points, dtype, device)
        pcd.point["colors"] = o3d.core.Tensor(rgb, dtype, device)

    else:
        pcd = o3d.t.io.read_point_cloud(sys.argv[1])

    # intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
    #                              [0, 0, 1]])

    # intrinsic = np.array(
    #     [
    #         [588.1710164208122, 0, 322.3853181081311],
    #         [0, 581.0604076590711, 244.1382373698799],
    #         [0, 0, 1],
    #     ]
    # )  # kinect1 depth

    # distortion_coeff = np.array(
    #     [
    #         -0.1195669067222111,
    #         0.3193720631249336,
    #         -0.00531485394824838,
    #         0.004069078980375383,
    #         0,
    #     ]
    # )  # kinect1 depth

    intrinsic = np.array(
        [
            [520.342706004118, 0, 323.0580496437712],
            [0, 513.826209565285, 263.4994539787398],
            [0, 0, 1],
        ]
    )  # kinect1 rgb

    distortion_coeff = np.array(
        [
            0.1594658601746339,
            -0.283618200726369,
            -0.003065915455824548,
            0.003289899753081607,
            0,
        ]
    )  # kinect1 rgb

    # intrinsic = np.array(
    #     [
    #         [530.9794921875, 0, 325.0990461931215],
    #         [0, 526.4757080078125, 262.4656608709811],
    #         [0, 0, 1],
    #     ]
    # )
    # distortion_coeff = np.array([0.04360382222641044, -0.1264981673473666, -0.0026697608002011, -0.001095539179069799, 0])

    # distortion_coeff = np.array(
    #     [
    #         0.0,0,0,0,0,
    #     ]
    # )

    # intrinsic = np.array(
    #     [
    #         [1.0, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1],
    #     ]
    # )  # kinect1 rgb
    intrinsic_tensor = o3d.core.Tensor(intrinsic)

    # o3d.visualization.draw([pcd])
    depth_scale = 1000.0
    rgbd_reproj = pcd.project_to_rgbd_image(
        640, 480, intrinsic_tensor, depth_scale=depth_scale, depth_max=4.0
    )
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(np.asarray(rgbd_reproj.color.to_legacy()))
    # axs[1].imshow(np.asarray(rgbd_reproj.depth.to_legacy()))
    # plt.show()

    # bgr = cv2.imread('/Users/bugra.sefercik/Desktop/camera_image.jpeg')
    # bgr = cv2.imread('/Users/bugra.sefercik/Desktop/yo.png')
    rgb = np.asarray(rgbd_reproj.color.to_legacy())
    depth = np.asarray(rgbd_reproj.depth.to_legacy())
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * 255
    # cv2.imwrite("nane.jpg", bgr * 255)
    # cv2.imshow("NANE", bgr)
    # cv2.waitKey(0)

    aruco_type = cv2.aruco.DICT_6X6_1000

    frame, rvec, tcoor = pose_esitmation(bgr, aruco_type, intrinsic, distortion_coeff)
    qvec = euler_to_quaternion(rvec[0][0])
    # tvec = tvec[0][0]
    # 264., 198

    corners_3d = list()
    for i in range(4):
        u, v = tuple(tcoor[i])
        fx = 520.342706004118
        cx = 323.0580496437712
        fy = 513.826209565285
        cy = 263.4994539787398

        d = depth[v, u]
        z = d / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        tvec = np.array([x, y, z])
        corners_3d.append(tvec)

    corners_3d = np.asarray(corners_3d, dtype=np.float32)

    tag_size = 0.075  # meters

    # corners_3d_ref = np.array(
    #     [
    #         [-tag_size / 2, -tag_size / 2, 0],
    #         [-tag_size / 2, tag_size / 2, 0],
    #         [tag_size / 2, tag_size / 2, 0],
    #         [tag_size / 2, -tag_size / 2, 0],
    #     ],
    #     dtype=np.float32
    # )  # original ar tag

    corners_3d_ref = np.array(
        [
            [0, tag_size / 2, -tag_size / 2],
            [0, -tag_size / 2, -tag_size / 2],
            [0, -tag_size / 2, tag_size / 2],
            [0, tag_size / 2, tag_size / 2],
        ],
        dtype=np.float32
    )

    t_tag2ee = np.array([-0.012, -0.0, -0.05])

    R, tvec = transformation.get_rigid_transform_3D(corners_3d_ref, corners_3d)
    tvec = tvec + (R @ t_tag2ee)

    qvec = transformation.get_q_from_matrix(R)
    tag_frame = create_coordinate_frame(np.concatenate((tvec, qvec)), switch_w=False, radius=0.004)

    o3d.visualization.draw([pcd, tag_frame], show_skybox=False)

    # R =  R @ R_tag2ee
    # qvec = transformation.get_q_from_matrix(R)
    # tag_frame = create_coordinate_frame(np.concatenate((tvec, qvec)), switch_w=False, radius=0.004)

    # o3d.visualization.draw([pcd, tag_frame], show_skybox=False)

    # cv2.imshow("NANE", frame / 255)
    # cv2.waitKey(0)
