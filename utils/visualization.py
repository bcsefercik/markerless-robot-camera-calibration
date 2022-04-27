import copy

import open3d as o3d


def get_frame_from_pose(base_frame, pose, switch_w=True):
    frame = copy.deepcopy(base_frame)

    if not isinstance(pose, list):
        pose = pose.tolist()

    ee_position = pose[:3]
    ee_orientation = pose[3:]
    if switch_w:
        ee_orientation = ee_orientation[-1:] + ee_orientation[:-1]

    ee_frame = frame.translate(ee_position)
    ee_frame.rotate(frame.get_rotation_matrix_from_quaternion(ee_orientation))

    return ee_frame


def get_ee_center_from_pose(pose, switch_w=True):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    ee_frame = get_frame_from_pose(frame, pose, switch_w=switch_w)

    return ee_frame.get_center()
