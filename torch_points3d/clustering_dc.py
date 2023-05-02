# Adapted from clustering file of deepCluster depository
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# from torch_points3d.datasets.change_detection.Urb3DCD_deepCluster import Urb3DCDPairCylinder_reassignedDC
__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')


    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def run_kmeans(x, nmb_clusters):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    x = np.ascontiguousarray(x)
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)
    # clus.seed = 1234

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.min_points_per_centroid = 1
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    D, I = index.search(x, 1)
    # stats = clus.iteration_stats
    loss = D.mean()
    return I, loss, faiss.vector_float_to_array(clus.centroids).reshape(nmb_clusters, d), index




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def arrange_clustering(pts_lists):
    pseudolabels = []
    pt_indexes = []
    for cluster, pts in enumerate(pts_lists):
        pt_indexes.extend(images)
        pseudolabels.extend([cluster] * len(pts))
    indexes = np.argsort(pt_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)
        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.pts_class = np.zeros((data.shape[0], 1))
        # self.pts_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            # self.pts_lists[I[i]].append(i)
            self.pts_class[i] = I[i]
        self.pts_class = self.pts_class.astype('int')
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


################################################################################
#                                   Faiss related                              #
################################################################################
#
def get_faiss_module(in_dim):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0  # NOTE: Single GPU only.
    idx = faiss.GpuIndexFlatL2(res, in_dim, cfg)
    return idx



def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)
    return index


def index_from_centroid(centroids):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    return index

