import os
import sys

import ipdb
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils, icp, transformation


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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')
    gray = cv2.flip(gray, 0)
    # ipdb.set_trace()
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    # ipdb.set_trace()
    print(len(corners))

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.2, matrix_coefficients, distortion_coefficients
            )
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(
                frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01
            )

    return frame


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

    # intrinsic = o3d.core.Tensor(
    #     [
    #         [588.1710164208122, 0, 322.3853181081311],
    #         [0, 581.0604076590711, 244.1382373698799],
    #         [0, 0, 1],
    #     ]
    # )  # kinect1 depth
    intrinsic = np.array(
        [
            [520.342706004118, 0, 323.0580496437712],
            [0, 513.826209565285, 263.4994539787398],
            [0, 0, 1],
        ]
    )  # kinect1 rgb
    intrinsic_tensor = o3d.core.Tensor(intrinsic)

    distortion_coeff = np.array(
        [
            0.1594658601746339,
            -0.283618200726369,
            -0.003065915455824548,
            0.003289899753081607,
            0,
        ]
    )  # kinect1 rgb

    # o3d.visualization.draw([pcd])

    rgbd_reproj = pcd.project_to_rgbd_image(
        640, 480, intrinsic_tensor, depth_scale=1000.0, depth_max=4.0
    )
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(np.asarray(rgbd_reproj.color.to_legacy()))
    # axs[1].imshow(np.asarray(rgbd_reproj.depth.to_legacy()))
    # plt.show()

    img = cv2.imread('/Users/bugra.sefercik/Desktop/camera_image.jpeg')
    # img = cv2.imread('/Users/bugra.sefercik/Desktop/wefasdf.png')
    rgb = np.asarray(rgbd_reproj.color.to_legacy())
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("nane.jpg", bgr * 255)
    # cv2.imshow("NANE", bgr)
    # cv2.waitKey(0)

    aruco_type = cv2.aruco.DICT_6X6_250

    frame = pose_esitmation(rgb, aruco_type, intrinsic, distortion_coeff)

    cv2.imshow("NANE", img)
    cv2.waitKey(0)
