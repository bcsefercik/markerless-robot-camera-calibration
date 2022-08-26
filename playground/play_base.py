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


# BASE_POSE = np.array(
#     [-0.0113, 0.0775, 1.592, -0.3419, -0.6686, -0.5882, 0.3004]
# )  # w first p2 training


# BASE_POSE = np.array(
#     [-0.6657, 0.1012, 1.5035, -0.4688, -0.8828, -0.014, -0.0276]
# )  # w first p3 training

# BASE_POSE = np.array(
#     [-0.66641757, 0.1187079, 1.48400726, 0.46608805, 0.88422785, -0.00244041, 0.02995122]
# )  # w first p3 training


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

# BASE_POSE = np.array([0.055, 0.0998, 1.4681, 0.3208, 0.6593, 0.6209, -0.2773])  # p2
# BASE_POSE = np.array([0.0456, 0.1017, 1.4779, -0.3267, -0.6638, -0.6142, 0.2746])  # p2
# BASE_POSE = np.array([0.047, 0.1, 1.472, 0.3208, 0.6593, 0.6209, -0.2773])  # p2
# BASE_POSE = np.array([0.0464, 0.1046, 1.4775, -0.3252, -0.6599, -0.6187, 0.2757])  # p2

# BASE_POSE = np.array([0.0519, 0.0988, 1.4776, -0.3212, -0.6615, -0.6189, 0.276])  # p2
# BASE_POSE = np.array([0.0621, 0.0989, 1.471, -0.3172, -0.6559, -0.6237, 0.283])  # p2
# BASE_POSE = np.array([0.0618, 0.0996, 1.4652, -0.3177, -0.6542, -0.6263, 0.2807])  # p2


# BASE_POSE = np.array([-0.6568, 0.3475, 1.1568, -0.45, -0.8917, 0.0186, -0.0444])  # p3
# BASE_POSE = np.array([-0.6674, 0.3508, 1.1362, -0.4583, -0.8867, 0.0239, -0.0561])  # p3
# BASE_POSE = np.array([-0.6711, 0.3456, 1.1408, -0.4557, -0.8881, 0.0137, -0.059])  # p3 tested calib
# BASE_POSE = np.array([-0.68, 0.3334, 1.1257, -0.452, -0.8889, 0.0099, -0.0742])
# BASE_POSE = np.array([0.5953, 0.3371, 0.9775, -0.0235, -0.038, 0.9222, -0.3841])
# BASE_POSE = np.array([0.5850, 0.3446, 0.9857, -0.0222, -0.0486, 0.9256, -0.3748]) # rocks p1
# BASE_POSE = np.array([0.5963, 0.3352, 0.9774, 0.0263, 0.0348, -0.9226, 0.3833])
# BASE_POSE = np.array([0.6259, 0.3257, 1.0161, -0.0206, -0.0182, 0.9011, -0.4327])  # p1_tag
BASE_POSE = np.array( [0.6178, 0.3371, 1.0274, -0.0141, -0.0227, 0.9011, -0.4328])  # p1_tag_calib
BASE_POSE = np.array([-0.0035, -0.1101, 1.6094, -0.2996, -0.6655, -0.6307, 0.2638])  # p2
BASE_POSE = np.array([-0.756, 0.0512, 1.2172, 0.3614, 0.9317, -0.0307, 0.0206])  # p3

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
    ee_frame = create_coordinate_frame(ee_pose_w_first_transformed, switch_w=False, radius=0.004)
    base_frame = create_coordinate_frame(BASE_POSE, switch_w=False)

    print(ee_pose_w_first_transformed)

    o3d.visualization.draw_geometries([pcd, kinect_frame, ee_frame, base_frame])
