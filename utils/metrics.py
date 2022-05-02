import torch
import torch.nn.functional as F
import ipdb

EPS = 1e-7


def compute_pose_dist(gt, pred, position_voxelization=1):
    with torch.no_grad():
        position = gt[:, :3]
        position *= position_voxelization
        orientation = gt[:, 3:7]
        position_pred = pred[:, :3]
        position_pred *= position_voxelization
        orientation_pred = pred[:, 3:7]

        gt_orientation_normalized = F.normalize(orientation, p=2, dim=1)
        orientation_pred_normalized = F.normalize(orientation_pred, p=2, dim=1)

        dist = torch.norm(gt - pred[:, :7], dim=1)
        dist_position = torch.norm(position - position_pred, dim=1)
        dist_orientation = torch.min(
            torch.norm(orientation - orientation_pred, dim=1),
            torch.norm(orientation + orientation_pred, dim=1),
        )  # we need have both here since negative of a quaternion represents the same rotation

        angle_diff = torch.acos(
            2
            * (
                torch.sum(
                    gt_orientation_normalized * orientation_pred_normalized, dim=1
                )
                ** 2
            )
            - 1,
        )

        return dist, dist_position, dist_orientation, angle_diff
