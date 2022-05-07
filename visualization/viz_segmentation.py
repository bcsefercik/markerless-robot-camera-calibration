import pickle
import sys
import os
import argparse
import json

import ipdb
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_utils
from utils.data import get_roi_mask
from utils.visualization import generate_colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viz segmentation.")
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--results", type=str, required=False)
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--roi", type=str, default=None)
    parser.add_argument("--roi_offset", type=float, default=0.13)
    parser.add_argument("--position", type=str, default='pos')
    args = parser.parse_args()

    np.random.seed(39)
    colors = generate_colors(args.classes)

    results = None
    if args.results:
        with open(args.results, 'rb') as fp:
            results = pickle.load(fp)

    roi = dict()
    if args.roi is not None:
        with open(args.roi, 'r') as fp:
            roi = json.load(fp)

    data, semantic_pred = file_utils.load_alive_file(args.pickle)

    if isinstance(data, dict):
        points = data['points']
        rgb = data['rgb']
        labels = data['labels']
        instance_label = data['instance_labels']
        pose = data['pose']
    else:
        points, rgb, labels, instance_label, pose = data

    roi_mask = get_roi_mask(points, offset=args.roi_offset, **roi.get(args.position, dict()))

    points = points[roi_mask]
    rgb = rgb[roi_mask]
    labels = labels[roi_mask]
    labels = labels.astype(int).tolist()

    predicted_labels = results[args.instance]['labels'] if results is not None  else labels

    pcd = o3d.geometry.PointCloud()

    def switch_to_pred_segmentation(vis):
        pcd.colors = o3d.utility.Vector3dVector(colors[predicted_labels])
        vis.update_geometry(pcd)
        return False

    def switch_to_gt_segmentation(vis):
        pcd.colors = o3d.utility.Vector3dVector(colors[labels])
        vis.update_geometry(pcd)
        return False

    def switch_to_normal(vis):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        return False

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    key_to_callback = dict()
    key_to_callback[ord("J")] = switch_to_pred_segmentation
    key_to_callback[ord("L")] = switch_to_gt_segmentation
    key_to_callback[ord("K")] = switch_to_normal
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    # o3d.visualization.draw_geometries([pcd, pcd_th])
    # ipdb.set_trace()
