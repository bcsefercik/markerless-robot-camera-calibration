import sys
import os

import ipdb
import copy
import numpy as np
import open3d as o3d


if __name__ == "__main__":

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    # frame = frame.rotate(frame.get_rotation_matrix_from_xyz((np.pi,0,np.pi/4)))
    if os.path.isfile(sys.argv[1].replace(".pcd", ".npy").replace(".ply", ".npy")):
        ee = np.load(sys.argv[1].replace(".pcd", ".npy"), allow_pickle=True)
    else:
        ee = np.zeros((7))
    ee_position = ee[:3]
    ee_orientation = ee[3:].tolist()
    ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]
    ee_frame = copy.deepcopy(frame).translate(ee_position)
    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    pcd = o3d.io.read_point_cloud(sys.argv[1])
    # bg = o3d.io.read_point_cloud(sys.argv[2])
    # ipdb.set_trace()

    points = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)

    roi_mask = points[:, 0] > -500

    # c1_p1
    roi_mask = np.logical_and(points[:, 0] < 0.43, roi_mask)  # x
    roi_mask = np.logical_and(points[:, 0] > -0.6, roi_mask)
    roi_mask = np.logical_and(points[:, 1] < 0.3, roi_mask)  # y
    # # # roi_mask = np.logical_and(points[:, 1] > -0.02, roi_mask)
    roi_mask = np.logical_and(points[:, 2] < 1.2, roi_mask)  # z
    roi_mask = np.logical_and(points[:, 2] > 0, roi_mask)

    # p2_1
    # roi_mask = np.logical_and(points[:, 0] < 0.39, roi_mask)  # x
    # roi_mask = np.logical_and(points[:, 0] > -0.52, roi_mask)
    # roi_mask = np.logical_and(points[:, 1] < 0.27, roi_mask)  # y
    # # # # roi_mask = np.logical_and(points[:, 1] > -0.02, roi_mask)
    # roi_mask = np.logical_and(points[:, 2] < 1.15, roi_mask)  # z
    # roi_mask = np.logical_and(points[:, 2] > 0.3, roi_mask)

    # p3_1
    # roi_mask = np.logical_and(points[:, 0] < 0.22, roi_mask)  # x
    # roi_mask = np.logical_and(points[:, 0] > -0.52, roi_mask)
    # roi_mask = np.logical_and(points[:, 1] < 0.26, roi_mask)  # y
    # # # roi_mask = np.logical_and(points[:, 1] > -0.02, roi_mask)
    # roi_mask = np.logical_and(points[:, 2] < 1.2, roi_mask)  # z
    # roi_mask = np.logical_and(points[:, 2] > 0, roi_mask)

    # p2h2l1
    # roi_mask = np.logical_and(points[:, 0] < 0.5, roi_mask)  # x
    # roi_mask = np.logical_and(points[:, 0] > -0.5, roi_mask)
    # roi_mask = np.logical_and(points[:, 1] < 0.27, roi_mask)  # y
    # # # roi_mask = np.logical_and(points[:, 1] > 0.2, roi_mask)
    # roi_mask = np.logical_and(points[:, 2] < 1.3, roi_mask)  # z
    # # # # roi_mask = np.logical_and(points[:, 2] > 1.5, roi_mask)

    # p2h3l1
    # roi_mask = np.logical_and(points[:, 0] < 0.5, roi_mask)  # x
    # roi_mask = np.logical_and(points[:, 0] > -0.5, roi_mask)
    # roi_mask = np.logical_and(points[:, 1] < 0.27, roi_mask)  # y
    # # roi_mask = np.logical_and(points[:, 1] > -0.02, roi_mask)
    # roi_mask = np.logical_and(points[:, 2] < 1.27, roi_mask)  # z
    # # roi_mask = np.logical_and(points[:, 2] > 1, roi_mask)

    points = points[roi_mask]
    rgb = rgb[roi_mask]

    print("# of points:", len(points))

    pcd_th = o3d.geometry.PointCloud()
    pcd_th.points = o3d.utility.Vector3dVector(points)
    pcd_th.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd_th, ee_frame])

    # ipdb.set_trace()

    # o3d.io.write_point_cloud(
    #     sys.argv[1].replace(".pcd", "_processed.pcd"),
    #     pcd_th,
    #     print_progress=True
    # )
