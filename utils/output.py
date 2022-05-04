import numpy as np
from sklearn.cluster import AgglomerativeClustering

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
