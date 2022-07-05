import os
import sys

import numpy as np
import open3d as o3d


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_PATH))
from utils.transformation import get_pose_from_matrix, get_transformation_matrix


def get_point2point_matcher(cad_name=os.path.join(BASE_PATH, "..", "app", "hand_files", "hand_notblender.obj")):
    # _cad_mesh = o3d.io.read_triangle_mesh(
    #     os.path.join(BASE_PATH, "..", "app", "hand_files", "hand_notblender.obj")
    # )
    if cad_name.endswith(".pcd"):
        _cad_pcd = o3d.io.read_point_cloud(cad_name)
    else:
        _cad_mesh = o3d.io.read_triangle_mesh(
            os.path.join(
                BASE_PATH, "..", "app", "hand_files", "hand_notblender.obj"
            )  # seems to work better
        )

        _cad_pcd = _cad_mesh.sample_points_uniformly(
            number_of_points=16384
        )  # has normal since converted from mesh
        _cad_pcd = _cad_mesh.sample_points_poisson_disk(
            number_of_points=8192, pcl=_cad_pcd
        )
        _pcd_cad_points = np.asarray(_cad_pcd.points)
        _pcd_cad_normals = np.asarray(_cad_pcd.normals)
        _pcd_cad_mask = _pcd_cad_points[:, 0] > 0.0 * (_pcd_cad_points[:, 2] > -0.02)
        _cad_pcd.points = o3d.utility.Vector3dVector(
            _pcd_cad_points[_pcd_cad_mask]
        )
        _cad_pcd.normals = o3d.utility.Vector3dVector(
            _pcd_cad_normals[_pcd_cad_mask]
        )
    _ee_pcd = o3d.geometry.PointCloud()
    _icp_threshold = 0.1
    _icp_method = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    _icp_normal_search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.02, max_nn=30
    )

    def match(ee_points, pose_initial):
        if ee_points is None or pose_initial is None:
            return pose_initial

        trans_mat_initial = get_transformation_matrix(pose_initial, switch_w=False)

        _ee_pcd.points = o3d.utility.Vector3dVector(ee_points)
        _ee_pcd.estimate_normals(search_param=_icp_normal_search_param)

        reg_p2l = o3d.pipelines.registration.registration_icp(
            _cad_pcd,
            _ee_pcd,
            _icp_threshold,
            trans_mat_initial,
            _icp_method,
        )

        return get_pose_from_matrix(reg_p2l.transformation)

    return match
