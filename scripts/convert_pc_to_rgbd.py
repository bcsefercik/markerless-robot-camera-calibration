import os
import sys

import ipdb
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils, icp, transformation

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

    intrinsic = o3d.core.Tensor(
        [
            [520.342706004118, 0, 323.0580496437712],
            [0, 513.826209565285, 263.4994539787398],
            [0, 0, 1],
        ]
    )  # kinect1 rgb

    # o3d.visualization.draw([pcd])
    rgbd_reproj = pcd.project_to_rgbd_image(
        640, 480, intrinsic, depth_scale=5000.0, depth_max=10.0
    )

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.asarray(rgbd_reproj.color.to_legacy()))
    axs[1].imshow(np.asarray(rgbd_reproj.depth.to_legacy()))
    plt.show()
