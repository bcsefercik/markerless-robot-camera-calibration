from operator import gt
import statistics
from unittest import result
import torch
import torch.nn.functional as F
import numpy as np

from utils.quaternion import qmul_np, qconj_np
from utils.transformation import (
    get_quaternion_rotation_matrix
)
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


def compute_segmentation_metrics(gt: np.array, pred: np.array, classes=['background', 'arm', 'ee']):
    results = {
        "class_results": dict(),
    }

    precisions = list()
    recalls = list()

    for ci, cn in enumerate(classes):
        gt_idx = set(np.where(gt == ci)[0])
        pred_idx = set(np.where(pred == ci)[0])

        tp_idx = gt_idx & pred_idx
        tp = len(tp_idx)
        fn = len(gt_idx - tp_idx)
        fp = len(pred_idx - tp_idx)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        results["class_results"][cn] = {
            "precision": precision,
            "recall": recall
        }

        precisions.append(precision)
        recalls.append(recall)

    results["accuracy"] = sum(gt == pred) / len(gt)
    results["precision"] = statistics.mean(precisions)
    results["recall"] = statistics.mean(recalls)

    return results


def compute_pose_metrics(gt: np.array, pred: np.array):
    '''
    input: x, y, z, qw, qx, qy, qz
    '''
    results = dict()

    results['dist_position'] = np.linalg.norm(gt[:3] - pred[:3])

    gt_rot = gt[3:] / np.linalg.norm(gt[3:])
    pred_rot = pred[3:] / np.linalg.norm(pred[3:])
    q_mul = qmul_np(gt_rot, qconj_np(pred_rot))
    angle_diff = np.abs(2 * np.arctan2(np.linalg.norm(q_mul[1:]), q_mul[0]))
    results['angle_diff'] = min(angle_diff, 2 * np.pi - angle_diff)  # exact same result with calculation in compute_pose_dist()

    # angle_diff_old = torch.acos(2 * (torch.sum(torch.from_numpy(gt[3:] * pred[3:]).view(1, -1), dim=1) ** 2) - 1)[0].item()
    # print(results['dist_position'], results['angle_diff'], angle_diff_old)

    return results


def compute_kp_error(gt_coords, kp_coords, kp_classes):
    gt_coords_selected = gt_coords[kp_classes]

    return np.linalg.norm(gt_coords_selected - kp_coords, axis=1).mean()


def compute_ADD_np(points, gt_pose, pred_pose):
    gt_rot_mat = get_quaternion_rotation_matrix(gt_pose[3:], switch_w=False)
    gt_translation = gt_pose[:3]
    pred_rot_mat = get_quaternion_rotation_matrix(pred_pose[3:], switch_w=False)
    pred_translation = pred_pose[:3]

    gt_part = (gt_rot_mat @ points.T) + gt_translation.reshape(3, 1)
    pred_part = (pred_rot_mat @ points.T) + pred_translation.reshape(3, 1)

    ADD = np.linalg.norm(gt_part - pred_part, axis=0).mean()

    return ADD
