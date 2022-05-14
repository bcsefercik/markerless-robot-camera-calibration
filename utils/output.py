
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from utils.transformation import get_quaternion_rotation_matrix_torch

import MinkowskiEngine as ME

import ipdb


class ClusterUtil():
    def __init__(self, dist=0.06, linkage='single'):
        self.cluster = AgglomerativeClustering(
            distance_threshold=dist,
            n_clusters=None,
            affinity='euclidean',
            linkage=linkage
        )

    def get_largest_cluster(self, points):
        labels = self.cluster.fit(points).labels_
        unique, counts = np.unique(labels, return_counts=True)
        cluster_id = unique[counts.argmax()]
        cluster_idx = np.where(labels == cluster_id)[0]

        return cluster_idx

    def get_most_confident_cluster(self, points, confidences):
        # TODO: implement this method if necessary
        mean_conf = 1.0

        labels = self.cluster.fit(points).labels_
        unique, counts = np.unique(labels, return_counts=True)
        # if len(unique) > 1:
        #     ipdb.set_trace()
        cluster_id = unique[counts.argmax()]
        cluster_idx = np.where(labels == cluster_id)[0]

        return cluster_idx, mean_conf


def get_pred_center(out, coords, ee_r=0.03, q=None):
    selected_indices = out[:, 1].sort(descending=True)[1][:8]
    # selected_indices = out[:, 1].argmax()
    pred_center = mean_without_outliers(coords[selected_indices.cpu().numpy()])

    if q is not None:
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)

        q = q.view(1, -1)
        rot_mat = get_quaternion_rotation_matrix_torch(q)[0]
        offset = torch.tensor([-ee_r, 0, 0])
        offset_rotated = torch.matmul(rot_mat, offset)

        if isinstance(pred_center, np.ndarray):
            offset_rotated = offset_rotated.cpu().numpy()

        pred_center += offset_rotated

    return pred_center


def get_segmentations_from_tensor_field(field: ME.TensorField):
    logits = field.features
    conf, preds = logits.max(1)
    preds = preds.cpu().numpy()
    conf = torch.sigmoid(conf).cpu().numpy()

    return preds, conf


def mean_without_outliers(arr: np.array, axis_based: bool = False):
    # TODO: implement
    return arr.mean(axis=0)


def get_key_points(logits: torch.tensor, conf_th=0.999):
    softmax = logits.softmax(1).max(0)
    classes = np.where(softmax[0].cpu() > conf_th)[0]
    idx = softmax[1][classes].cpu().numpy()

    return idx, classes

