import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
import csv
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
import laspy

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair
from torch_points3d.metrics.change_detection_tracker import CDTracker
from torch_points3d.metrics.urb3DCD_tracker import Urb3DCDTracker
from torch_points3d.datasets.change_detection.AHNPairCylinder import AHNCylinder, to_ply


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# IGNORE_LABEL: int = 0

AHN_NUM_CLASSES = 4
viridis = cm.get_cmap('viridis', AHN_NUM_CLASSES)
INV_OBJECT_LABEL = {
    0: "Unchanged",
    1: "New building",
    2: "Demolition",
    3: "New tree,car, ...",

    # 4: "New water",
    # 5: "New bridge",
}
# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(AHN_NUM_CLASSES)}
#V1
# OBJECT_COLOR = np.asarray(
#     [
#         [67, 1, 84],  # 'unchanged'
#         [0, 150, 128],  # 'newlyBuilt'
#         [255, 208, 0],  # 'deconstructed'
#
#     ]
# )
OBJECT_COLOR = np.asarray(
    [
        [67, 1, 84],  # 'unchanged'
        [0, 183, 255],  # 'newlyBuilt'
        [0, 12, 235],  # 'deconstructed'
        [255, 230, 0],  # 'vegetationGrowUp'
    ]
)
# OBJECT_COLOR = viridis.colors[:,:3]
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}


# nb_elt_per_class =torch.tensor([366211.,  33109.,  36035.]) + torch.tensor([371420.,   9112.,  74458.]) #Single
# nb_elt_per_class =torch.tensor([3387760.,  385290.,  277884.])+ torch.tensor([1649863.,  175879.,  167870.]) #Little
# WEIGHTS = 1-nb_elt_per_class/nb_elt_per_class.sum()

class ForwardAHNDataset(BaseSiameseDataset):
    """ Wrapper around Semantic Kitti that creates train and test datasets.
        Parameters
        ----------
        dataset_opt: omegaconf.DictConfig
            Config dictionary that should contain
                - root,
                - split,
                - transform,
                - pre_transform
                - process_workers
        """
    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        print(self.dataset_opt.dataroot)
        self.test_dataset = AHNCylinder(
            filePaths=self.dataset_opt.dataroot,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Transfer"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
        )



    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme
            Parameters
            ----------
            pos : torch.Tensor
                tensor that contains the positions of the points
            label : torch.Tensor
                predicted label
            file : string
                Save location
            """
        to_ply(pos, label, file)

    def get_tracker(self, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCDTracker(self, wandb_log=False, use_tensorboard=False, stage='test',
                                 full_pc=full_pc, full_res=full_res)
    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset

    @property
    def num_classes(self):
        return self.test_data.num_classes

################################### UTILS #######################################

def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("pred", "i1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    ply_array["pred"] = np.asarray(label)
    el = PlyElement.describe(ply_array, "Urb3DSimul")
    PlyData([el], byte_order=">").write(file)

def read_from_ply(filename, nameInPly = "params"):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        # num_verts = plydata["Urb3DSimul"].count
        num_verts = plydata[nameInPly].count
        vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
        vertices[:, 0] = plydata[nameInPly].data["x"]
        vertices[:, 1] = plydata[nameInPly].data["y"]
        vertices[:, 2] = plydata[nameInPly].data["z"]
        vertices[:, 3] = plydata[nameInPly].data["label_ch"]
    return vertices
