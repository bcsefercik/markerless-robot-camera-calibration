from enum import Enum
from ipaddress import ip_address

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.quaternion import qeuler
from utils.metrics import compute_pose_dist
from utils import config
from utils.transformation import get_quaternion_rotation_matrix_torch

import ipdb


_config = config.Config()


class LossType(Enum):
    MSE = "mse"
    COS = "cos"
    ANGLE = "angle"
    COS2 = "cos2"
    WGEODESIC = "wgeodesic"
    SMOOTHL1 = "smoothl1"
    PLOSS = "ploss"


def get_criterion(device="cuda", loss_type=LossType.ANGLE, reduction="mean"):
    regression_criterion = nn.MSELoss(reduction=reduction).to(device)
    cos_regression_criterion = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    confidence_criterion = nn.BCELoss(reduction=reduction)
    smooth_l1_criterion = nn.SmoothL1Loss(reduction=reduction).to(device)

    confidence_enabled = _config()['STRUCTURE'].get('compute_confidence', False)

    reduction_func = torch.sum if reduction == "sum" else torch.mean

    gamma = 50
    gamma2 = 1

    def compute_angle_loss(q_expected, q_pred, reduction=reduction, x=None):
        expected_euler = qeuler(q_expected, order="zyx", epsilon=1e-6)
        predicted_euler = qeuler(q_pred, order="zyx", epsilon=1e-6)
        angle_distance = (
            torch.remainder(predicted_euler - expected_euler + np.pi, 2 * np.pi) - np.pi
        )

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(torch.abs(angle_distance))

    def compute_cos_loss(y, y_pred, reduction=reduction, x=None):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_rot = 1.0 - cos_regression_criterion(y[:, :3], y_pred[:, :3])

        reduction_func = torch.sum if reduction == "sum" else torch.mean

        return reduction_func(loss_rot) + loss_coor

    def compute_loss(y, y_pred, reduction=reduction, x=None):
        loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])
        loss_quaternion = compute_angle_loss(y[:, 3:7], y_pred[:, 3:7])

        loss = gamma * loss_coor + gamma2 * loss_quaternion

        return loss

    def compute_cos2_loss(y, y_pred, reduction=reduction, x=None):
        reduction_func = torch.sum if reduction == "sum" else torch.mean

        gamma_cos = 2

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            if not _config()['STRUCTURE'].get('disable_position', False):
                loss_rot = 1.0 - cos_regression_criterion(y[:, :7], y_pred[:, :7])
                loss_rot = reduction_func(loss_rot)
            else:
                loss_rot = regression_criterion(y[:, 3:7], y_pred[:, 3:7])
            loss_rot *= gamma_cos

        loss_confidence = 0
        if confidence_enabled:
            _, dist_position, _, angle_diff = compute_pose_dist(y, y_pred[:, :7])
            position_confidence_idx = (dist_position < _config.STRUCTURE.position_threshold) + (dist_position > _config.STRUCTURE.position_ignore_threshold)
            position_confidence = (dist_position < _config.STRUCTURE.position_threshold).float()
            loss_confidence += confidence_criterion(
                y_pred[:, 7][position_confidence_idx],
                position_confidence[position_confidence_idx]
            )

            orientation_confidence_idx = (angle_diff < _config.STRUCTURE.angle_diff_threshold) + (angle_diff > _config.STRUCTURE.angle_diff_ignore_threshold)
            orientation_confidence = (angle_diff < _config.STRUCTURE.angle_diff_threshold).float()
            loss_confidence += confidence_criterion(
                y_pred[:, 8][orientation_confidence_idx],
                orientation_confidence[orientation_confidence_idx]
            )

            overall_confidence_idx = position_confidence_idx * orientation_confidence_idx
            overall_confidence = position_confidence * orientation_confidence
            loss_confidence += confidence_criterion(
                y_pred[:, 9][overall_confidence_idx],
                overall_confidence[overall_confidence_idx]
            )

        return loss_rot + loss_coor + loss_confidence

    def compute_with_geodesic_loss(y, y_pred, reduction=reduction, x=None):
        reduction_func = torch.sum if reduction == "sum" else torch.mean

        gamma_cos = 1

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = regression_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            y_normalized = F.normalize(y[:, 3:7], p=2, dim=1)
            y_pred_normalized = F.normalize(y_pred[:, 3:7], p=2, dim=1)

            loss_rot = torch.acos(
                            (torch.sum(y_normalized * y_pred_normalized, dim=1) - 1) * 0.5,
                        )
            loss_rot = reduction_func(loss_rot)
            loss_rot *= gamma_cos

        loss_confidence = 0

        return loss_rot + loss_coor + loss_confidence

    def compute_smooth_l1_loss(y, y_pred, reduction=reduction, x=None):
        reduction_func = torch.sum if reduction == "sum" else torch.mean

        gamma_cos = 1

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = smooth_l1_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            y_normalized = F.normalize(y[:, 3:7], p=2, dim=1)
            y_pred_normalized = F.normalize(y_pred[:, 3:7], p=2, dim=1)

            loss_rot = torch.acos(
                            (torch.sum(y_normalized * y_pred_normalized, dim=1) - 1) * 0.5,
                        )
            loss_rot = reduction_func(loss_rot)
            loss_rot *= gamma_cos

        loss_confidence = 0

        return loss_rot + loss_coor + loss_confidence

    def compute_ploss(y, y_pred, reduction=reduction, x=None):
        loss_rot_total = 0.0
        rot_mat = get_quaternion_rotation_matrix_torch(y[:, 3:])
        rot_mat_pred = get_quaternion_rotation_matrix_torch(y_pred[:, 3:])

        for i, coords in enumerate(x.decomposed_coordinates):
            y_translated = torch.matmul(rot_mat[i], torch.transpose(coords.float(), 0, 1))
            y_pred_translated = torch.matmul(rot_mat_pred[i], torch.transpose(coords.float(), 0, 1))

            ipdb.set_trace()

        gamma_cos = 1

        loss_coor = 0
        if not _config()['STRUCTURE'].get('disable_position', False):
            loss_coor = smooth_l1_criterion(y[:, :3], y_pred[:, :3])

        loss_rot = 0
        if not _config()['STRUCTURE'].get('disable_orientation', False):
            y_normalized = F.normalize(y[:, 3:7], p=2, dim=1)
            y_pred_normalized = F.normalize(y_pred[:, 3:7], p=2, dim=1)

            loss_rot = torch.acos(
                            (torch.sum(y_normalized * y_pred_normalized, dim=1) - 1) * 0.5,
                        )
            loss_rot = reduction_func(loss_rot)
            loss_rot *= gamma_cos

        loss_confidence = 0

        return loss_rot + loss_coor + loss_confidence

    loss_func = compute_loss

    if loss_type == LossType.COS:
        loss_func = compute_cos_loss
    elif loss_type == LossType.MSE:
        loss_func = regression_criterion
    elif loss_type == LossType.COS2:
        loss_func = compute_cos2_loss
    elif loss_type == LossType.WGEODESIC:
        loss_func = compute_with_geodesic_loss
    elif loss_type == LossType.SMOOTHL1:
        loss_func = compute_smooth_l1_loss
    elif loss_type == LossType.PLOSS:
        loss_func = compute_ploss
    else:
        loss_func = compute_loss

    return loss_func
