
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from utils.transformation import get_quaternion_rotation_matrix_torch

import ipdb


class ClusterUtil():
    def __init__(self, dist=0.06, linkage='single'):
        self.cluster = AgglomerativeClustering(
            distance_threshold=dist,
            n_clusters=None,
            affinity='euclidean',
            linkage=linkage
        )

    def get_biggest_cluster(self, points):
        labels = self.cluster.fit(points).labels_
        unique, counts = np.unique(labels, return_counts=True)
        cluster_id = unique[counts.argmax()]
        cluster_idx = np.where(labels == cluster_id)[0]

        return cluster_idx


def get_pred_center(out, coords, ee_r=0.075, q=None):
    pred_center = coords[out[:, 1].argmax()]

    if q is not None:
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32).view(1, -1)
        q = q.view(1, -1)
        rot_mat = get_quaternion_rotation_matrix_torch(q)[0]
        offset = torch.tensor([-ee_r, 0, 0])
        pred_center += torch.matmul(rot_mat, offset)

    return pred_center
