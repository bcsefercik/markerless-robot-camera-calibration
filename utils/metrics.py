import torch


EPS = 1e-7


def compute_pose_dist(gt, pred):
    position = gt[:, :3]
    orientation = gt[:, 3:]
    position_pred = pred[:, :3]
    orientation_pred = pred[:, 3:]

    dist = torch.norm(gt - pred, dim=1)
    dist_position = torch.norm(position - position_pred, dim=1)
    dist_orientation = torch.norm(orientation - orientation_pred, dim=1)

    angle_diff = torch.acos(
        torch.clamp(
            2 * (torch.sum(orientation * orientation_pred, dim=1) ** 2) - 1,
            min=-1 + EPS,
            max=1 - EPS
        )
    )

    return dist, dist_position, dist_orientation, angle_diff
