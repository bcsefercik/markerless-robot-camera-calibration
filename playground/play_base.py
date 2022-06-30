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
from utils.transformation import switch_w, transform_pose2pose
from utils.preprocess import center_at_origin


BASE_POSE = np.array(
   [-0.032, 0.0763, 1.5966, -0.3511, -0.6685, -0.589, 0.2878]
)  # w first

# BASE_POSE = np.array(
#     [
#         -0.6603,
#         0.117,
#         1.5072,
#         -0.4742,
#         -0.8801,
#         -0.0127,
#         -0.0201,
#     ]
# )
# # w first p3_training

if __name__ == "__main__":
    ee2base_pose = None
    ee2base_pose_w_first = None
    if sys.argv[1].endswith("pickle"):
        data, semantic_pred = file_utils.load_alive_file(sys.argv[1])
        points = data["points"]
        rgb = data["rgb"]
        pose = data["pose"]
        ee2base_pose = data.get("robot2ee_pose")
        ee2base_pose_w_first = switch_w(ee2base_pose)
    else:
        _pcd = o3d.io.read_point_cloud(sys.argv[1])
        points = np.asarray(_pcd.points, dtype=np.float32)
        rgb = np.asarray(_pcd.colors, dtype=np.float32)
        pose = np.load(sys.argv[1].replace(".pcd", ".npy"), allow_pickle=True)
        ee2base_pose = np.load(
            sys.argv[1].replace(".pcd", "_robot2ee_pose.npy"), allow_pickle=True
        )
        ee2base_pose_w_first = switch_w(ee2base_pose)
    # ipdb.set_trace()
    pose_w_first = switch_w(pose)

    ee_pose_w_first_transformed = transform_pose2pose(BASE_POSE, ee2base_pose_w_first)

    roi_mask = get_roi_mask(points)
    points = points[roi_mask]
    rgb = rgb[roi_mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    kinect_frame = copy.deepcopy(frame).translate([0, 0, 0])
    kinect_frame.rotate(frame.get_rotation_matrix_from_quaternion([0] * 4))

    # kinect_frame = create_coordinate_frame([0] * 7, switch_w=False)
    ee_frame = create_coordinate_frame(ee_pose_w_first_transformed, switch_w=False)
    base_frame = create_coordinate_frame(BASE_POSE, switch_w=False)

    o3d.visualization.draw_geometries([pcd, kinect_frame, ee_frame, base_frame])
