import torch
import glob
import os
import os.path as osp
from torch_geometric.io import read_txt_array
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import knn_interpolate
import numpy as np
import logging

from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.utils import is_list
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.change_detection.Urb3DSimulPairCylinder import Urb3DSimulCylinder, to_ply
from torch_points3d.metrics.urb3DCD_tracker import Urb3DCDTracker
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

IGNORE_LABEL: int = 0

URB3DSIMUL_NUM_CLASSES = 7
viridis = cm.get_cmap('viridis', URB3DSIMUL_NUM_CLASSES)

INV_OBJECT_LABEL = {
    0: "unchanged",
    1: "newlyBuilt",
    2: "deconstructed",
    3: "newVegetation",
    4: "vegetationGrowUp",
    5: "vegetationRemoved",
    6: "mobileObjects"
}

# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(URB3DSIMUL_NUM_CLASSES)}
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
        [0, 217, 33],  # 'newVegetation'
        [255, 230, 0],  # 'vegetationGrowUp'
        [255, 140, 0],  # 'vegetationRemoved'
        [255, 0, 0],  # 'mobileObjects'
    ]
)
# OBJECT_COLOR = viridis.colors[:,:3]
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}


class ForwardUrb3DSimulDataset(BaseSiameseDataset):
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

        self.test_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataroot,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Transfer"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
        )
        # self.num_classes = self.num_classes()
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