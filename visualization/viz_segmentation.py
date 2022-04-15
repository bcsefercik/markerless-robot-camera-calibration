import sys
import os

import ipdb
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils


def generate_colors(n):
    return np.random.rand(n, 3)


if __name__ == "__main__":
    np.random.seed(39)

    (
        points,
        rgb,
        labels,
        instance_label,
        pose,
    ), semantic_pred = file_utils.load_alive_file(sys.argv[1])

    pcd = o3d.geometry.PointCloud()

    def switch_to_segmentation(vis):
        pcd.colors = o3d.utility.Vector3dVector(colors[kmeans.labels_[p2v]])
        vis.update_geometry(pcd)
        return False

    def switch_to_rgb_segmentation(vis):
        pcd.colors = o3d.utility.Vector3dVector(colors[kmeans_rgb.labels_])
        vis.update_geometry(pcd)
        return False

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    key_to_callback = dict()
    key_to_callback[ord("J")] = switch_to_segmentation
    key_to_callback[ord("L")] = switch_to_rgb_segmentation
    key_to_callback[ord("K")] = switch_to_normal
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    # o3d.visualization.draw_geometries([pcd, pcd_th])
    # ipdb.set_trace()
