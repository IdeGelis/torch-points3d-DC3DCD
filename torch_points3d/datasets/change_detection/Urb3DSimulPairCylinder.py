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

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair
from torch_points3d.metrics.change_detection_tracker import CDTracker
from torch_points3d.metrics.urb3DCD_tracker import Urb3DCDTracker


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



IGNORE_LABEL: int = -1

URB3DCD_NUM_CLASSES = 7
viridis = cm.get_cmap('viridis', URB3DCD_NUM_CLASSES)

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

#V2
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
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}



class Urb3DSimul(Dataset):
    """
    Definition of Urb3DCD Dataset
    """

    def __init__(self, filePaths="", split="train", DA=False, pre_transform=None, transform=None, preprocessed_dir="",
                 reload_preproc=False, reload_trees=False, nameInPly="params", comp_norm = False ):
        super(Urb3DSimul, self).__init__(None, None, pre_transform)
        self.class_labels = OBJECT_LABEL
        self._ignore_label = IGNORE_LABEL
        self.preprocessed_dir = preprocessed_dir
        if not osp.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        self.filePaths = filePaths
        self.nameInPly = nameInPly
        self._get_paths()
        self.split = split
        self.DA = DA
        self.pre_transform = pre_transform
        self.transform = None
        self.manual_transform = transform
        self.reload_preproc = reload_preproc
        self.reload_trees = reload_trees
        self.num_classes = URB3DCD_NUM_CLASSES
        self.nb_elt_class = torch.zeros(self.num_classes)
        self.filesPC0_prepoc = [None] * len(self.filesPC0)
        self.filesPC1_prepoc = [None] * len(self.filesPC1)
        self.process(comp_normal=comp_norm)
        if self.nb_elt_class.sum() == 0:
            self.get_nb_elt_class()
        self.weight_classes = 1 - self.nb_elt_class / self.nb_elt_class.sum()


    def _get_paths(self):
        self.filesPC0 = []
        self.filesPC1 = []
        globPath = os.scandir(self.filePaths)
        for dir in globPath:
            if dir.is_dir():
                curDir = os.scandir(dir)
                for f in curDir:
                    if f.name == "pointCloud0.ply":
                        self.filesPC0.append(f.path)
                    elif f.name == "pointCloud1.ply":
                        self.filesPC1.append(f.path)
                curDir.close()
        globPath.close()


    def size(self):
        return len(self.filesPC0)

    def get_nb_elt_class(self):
        self.nb_elt_class = torch.zeros(self.num_classes)
        for idx in range(len(self.filesPC0)):
            pc1 = torch.load(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))
            cpt = torch.bincount(pc1.y)
            for c in range(cpt.shape[0]):
                self.nb_elt_class[c] += cpt[c]

    def hand_craft_process(self,comp_normal=False):
        existfile = True
        for idx in range(len(self.filesPC0)):
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(idx)))
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))
        if not self.reload_preproc or not exist_file:
            for idx in range(len(self.filesPC0)):
                pc0, pc1, label = self.clouds_loader(idx, nameInPly=self.nameInPly)
                pc0 = Data(pos=pc0)
                pc1 = Data(pos=pc1, y=label)
                if comp_normal:
                    normal0 = getFeaturesfromPDAL(pc0.pos.numpy())
                    pc0.norm = torch.from_numpy(normal0)
                    normal1 = getFeaturesfromPDAL(pc1.pos.numpy())
                    pc1.norm = torch.from_numpy(normal1)
                if self.pre_transform is not None:
                    pc0 = self.pre_transform(pc0)
                    pc1 = self.pre_transform(pc1)
                cpt = torch.bincount(pc1.y)
                for c in range(cpt.shape[0]):
                    self.nb_elt_class[c] += cpt[c]
                torch.save(pc0, osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(idx)))
                torch.save(pc1, osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))

    def process(self, comp_normal = False):
        self.hand_craft_process(comp_normal)

    def get(self, idx):
        if self.pre_transform is not None:
            pc0, pc1, label = self._preproc_clouds_loader(idx)
        else:
            pc0, pc1, label = self.clouds_loader(idx, nameInPly=self.nameInPly)
        if (hasattr(pc0, "multiscale")):
            batch = MultiScalePair(pos=pc0, pos_target=pc1, y=label)
        else:
            batch = Pair(pos=pc0, pos_target=pc1, y=label)
            batch.normalise()
        return batch.contiguous()

    def clouds_loader(self, area, nameInPly = "params"):
        print("Loading " + self.filesPC1[area])
        pc = self.cloud_loader(self.filesPC1[area], nameInPly=nameInPly)
        pc1 = pc[:, :3]
        gt = pc[:, 3].long() #/!\ Labels should be at the 4th column 0:X 1:Y 2:Z 3:LAbel
        pc0 = self.cloud_loader(self.filesPC0[area], nameInPly=nameInPly)[:, :3]
        return pc0.type(torch.float), pc1.type(torch.float), gt

    def _preproc_clouds_loader(self, area):
        data_pc0 = torch.load(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(area)))
        data_pc1 = torch.load(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(area)))
        return data_pc0.pos, data_pc1.pos, data_pc1.y

    def read_from_ply(self,filename, nameInPly="params", name_feat="label_ch"):
        """read XYZ for each vertex."""
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            vertices[:, 3] = plydata[nameInPly].data[name_feat]
        return vertices


    def cloud_loader(self, pathPC, cuda=False, nameInPly=None, name_feat = "label_ch"):
        """
      load a tile and returns points features (normalized xyz + intensity) and
      ground truth
      INPUT:
      pathPC = string, path to the tile of PC
      OUTPUT
      pc_data, [n x 3] float array containing points coordinates and intensity
      lbs, [n] long int array, containing the points semantic labels
      """
        if nameInPly is None:
            pc_data = self.read_from_ply(pathPC, nameInPly="params", name_feat=name_feat)
        else:
            pc_data = self.read_from_ply(pathPC, nameInPly=nameInPly, name_feat=name_feat)
        # load the point cloud data
        pc_data = torch.from_numpy(pc_data)

        if cuda:  # put the cloud data on the GPU memory
            pc_data = pc_data.cuda()
        return pc_data


class Urb3DSimulSphere(Urb3DSimul):
    """ Small variation of Urb3DCD that allows random sampling of spheres
    within an Area during training and validation. Spheres have a radius of 2m. If sample_per_epoch is not specified, spheres
    are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, sample_per_epoch=100, radius=2, fix_cyl=False, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = GridSampling3D(size=radius / 10.0)
        self.fix_cyl = fix_cyl
        super().__init__(*args, **kwargs)
        self._prepare_centers()
        # Trees are built in case it needs, now don't need to compute anymore trees
        self.reload_trees = True

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return self.grid_regular_centers.shape[0]

    def get(self, idx, dc = False):
        if self._sample_per_epoch > 0:
            if self.fix_cyl:
                centre = self._centres_for_sampling_fixed[idx, :3]
                area_sel = self._centres_for_sampling_fixed[idx, 3].int()
                pair = self._load_save(area_sel)
                sphere_sampler = SphereSampling(self._radius, centre, align_origin=False)
                dataPC0 = Data(pos=pair.pos)
                setattr(dataPC0, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
                dataPC1 = Data(pos=pair.pos_target, y=pair.y)
                setattr(dataPC1, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
                dataPC0_sphere = sphere_sampler(dataPC0)
                dataPC1_sphere = sphere_sampler(dataPC1)
                pair_spheres = Pair(pos=dataPC0_sphere.pos, pos_target=dataPC1_sphere.pos, y=dataPC1_sphere.y)
                pair_spheres.normalise()
                return pair_spheres
            else:
                return self._get_random()
        else:
            centre = self.grid_regular_centers[idx, :3]
            area_sel = self.grid_regular_centers[idx, 3].int()
            pair = self._load_save(area_sel)
            sphere_sampler = SphereSampling(self._radius, centre, align_origin=False)
            dataPC0 = Data(pos=pair.pos)
            setattr(dataPC0, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
            dataPC1 = Data(pos=pair.pos_target, y=pair.y)
            setattr(dataPC1, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
            dataPC0_sphere = sphere_sampler(dataPC0)
            dataPC1_sphere = sphere_sampler(dataPC1)
            if self.manual_transform is not None:
                dataPC0_sphere = self.manual_transform(dataPC0_sphere)
                dataPC1_sphere = self.manual_transform(dataPC1_sphere)
            pair_spheres = Pair(pos=dataPC0_sphere.pos, pos_target=dataPC1_sphere.pos, y=dataPC1_sphere.y)
            pair_spheres.normalise()
            return pair_spheres.contiguous()


    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        #  choice of the corresponding PC if several PCs are loaded
        area_sel = centre[3].int()
        pair = self._load_save(area_sel)
        sphere_sampler = SphereSampling(self._radius, centre[:3], align_origin=False)
        dataPC0 = Data(pos=pair.pos)
        setattr(dataPC0, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
        dataPC1 = Data(pos=pair.pos_target, y=pair.y)
        setattr(dataPC1, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
        dataPC0_sphere = sphere_sampler(dataPC0)
        dataPC1_sphere = sphere_sampler(dataPC1)
        if self.manual_transform is not None:
            dataPC0_sphere = self.manual_transform(dataPC0_sphere)
            dataPC1_sphere = self.manual_transform(dataPC1_sphere)
        pair_sphere = Pair(pos=dataPC0_sphere.pos, pos_target=dataPC1_sphere.pos, y=dataPC1_sphere.y)
        pair_sphere.normalise()
        return pair_sphere

    def _prepare_centers(self):
        self._centres_for_sampling = []
        grid_sampling = GridSampling3D(size=self._radius / 2)
        self.grid_regular_centers = []
        for i in range(len(self.filesPC0)):
            pair = self._load_save(i)
            if self._sample_per_epoch > 0:
                dataPC1 = Data(pos=pair.pos_target, y=pair.y)
                low_res = self._grid_sphere_sampling(dataPC1)
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
            else:
                # Get regular center on PC1, PC0 will be sampled using the same center
                dataPC1 = Data(pos=pair.pos_target, y=pair.y)
                grid_sample_centers = grid_sampling(dataPC1.clone())
                centres = torch.empty((grid_sample_centers.pos.shape[0], 4), dtype=torch.float)
                centres[:, :3] = grid_sample_centers.pos
                centres[:, 3] = i
                self.grid_regular_centers.append(centres)

        if self._sample_per_epoch > 0:
            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            print(uni_counts)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            print(self._label_counts)
            self._labels = uni
            self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)
            if self.fix_cyl:
                self._centres_for_sampling_fixed = []
                # choice of cylinders for all the training
                np.random.seed(1)
                chosen_labels = np.random.choice(self._labels, p=self._label_counts, size=(self._sample_per_epoch, 1))
                uni, uni_counts = np.unique(chosen_labels, return_counts=True)
                print("fixed cylinder", uni, uni_counts)
                for c in range(uni.shape[0]):
                    valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, -1] == uni[c]]
                    centres_idx = np.random.randint(low = 0, high=valid_centres.shape[0], size=(uni_counts[c],1))
                    self._centres_for_sampling_fixed.append(np.squeeze(valid_centres[centres_idx,:], axis=1))
                self._centres_for_sampling_fixed = torch.cat(self._centres_for_sampling_fixed, 0)
        else:
            self.grid_regular_centers = torch.cat(self.grid_regular_centers, 0)

    def _load_save(self, i):
        if self.pre_transform is not None:
            pc0, pc1, label = self._preproc_clouds_loader(i)
        else:
            pc0, pc1, label = self.clouds_loader(i, nameInPly=self.nameInPly)
        pair = Pair(pos=pc0, pos_target=pc1, y=label)
        path = self.filesPC0[i]
        name_tree = os.path.basename(path).split(".")[0] + "_radius" + str(int(self._radius)) + "_" + str(i) + ".p"
        path_treesPC0 = os.path.join(self.preprocessed_dir, "tp3DTree", name_tree)  # osp.dirname(path)
        if self.reload_trees and osp.isfile(path_treesPC0):
            file = open(path_treesPC0, "rb")
            tree = pickle.load(file)
            file.close()
            pair.KDTREE_KEY_PC0 = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory exists
            if not osp.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(osp.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pc0), leaf_size=10)
            file = open(path_treesPC0, "wb")
            pickle.dump(tree, file)
            file.close()
            pair.KDTREE_KEY_PC0 = tree

        path = self.filesPC1[i]
        name_tree = os.path.basename(path).split(".")[0] + "_radius" + str(int(self._radius)) + "_" + str(i) + ".p"
        path_treesPC1 = os.path.join(self.preprocessed_dir, "tp3DTree", name_tree)
        if self.reload_trees and osp.isfile(path_treesPC1):
            file = open(path_treesPC1, "rb")
            tree = pickle.load(file)
            file.close()
            pair.KDTREE_KEY_PC1 = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory exists
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pc1), leaf_size=10)
            file = open(path_treesPC1, "wb")
            pickle.dump(tree, file)
            file.close()
            pair.KDTREE_KEY_PC1 = tree
        return pair


class Urb3DSimulCylinder(Urb3DSimulSphere):
    def get(self, idx):
        if self._sample_per_epoch > 0:
            if self.fix_cyl:
                pair_correct = False
                while not pair_correct and idx < self._centres_for_sampling_fixed.shape[0]:
                    centre = self._centres_for_sampling_fixed[idx, :3]
                    area_sel = self._centres_for_sampling_fixed[idx, 3].int()  # ---> ici choix du pc correspondant si pls pc chargés
                    pair = self._load_save(area_sel)
                    cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
                    dataPC0 = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]).reshape(-1))
                    setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
                    dataPC1 = Data(pos=pair.pos_target, y=pair.y, idx=torch.arange(pair.pos_target.shape[0]).reshape(-1))
                    setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
                    dataPC0_cyl = cylinder_sampler(dataPC0)
                    dataPC1_cyl = cylinder_sampler(dataPC1)
                    pair_cylinders = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                                          idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel)
                    try:
                        pair_cylinders.normalise()
                        pair_correct = True
                    except:
                        print(pair_cylinders.pos.shape)
                        print(pair_cylinders.pos_target.shape)
                        idx += 1
                return pair_cylinders
            else:
                return self._get_random()
        else:
            pair_correct = False
            while not pair_correct and idx<self.grid_regular_centers.shape[0]:
                centre = self.grid_regular_centers[idx, :3]
                area_sel = self.grid_regular_centers[idx, 3].int()
                pair = self._load_save(area_sel)
                cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
                dataPC0 = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]).reshape(-1))
                setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
                dataPC1 = Data(pos=pair.pos_target, y=pair.y, idx=torch.arange(pair.pos_target.shape[0]).reshape(-1))
                setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
                dataPC0_cyl = cylinder_sampler(dataPC0)
                dataPC1_cyl = cylinder_sampler(dataPC1)
                try:
                    if self.manual_transform is not None:
                        dataPC0_cyl = self.manual_transform(dataPC0_cyl)
                        dataPC1_cyl = self.manual_transform(dataPC1_cyl)
                    pair_cylinders = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                                          idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel)
                    if self.DA:
                        pair_cylinders.data_augment()
                    pair_cylinders.normalise()
                    pair_correct = True
                except:
                    print('pair not correct')
                    idx += 1
            return pair_cylinders

    def _get_random(self):
        # Random cylinder biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        #  choice of the corresponding PC if several PCs are loaded
        area_sel = centre[3].int()
        pair = self._load_save(area_sel)
        cylinder_sampler = CylinderSampling(self._radius, centre[:3], align_origin=False)
        dataPC0 = Data(pos=pair.pos)
        setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
        dataPC1 = Data(pos=pair.pos_target, y=pair.y)
        setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
        dataPC0_cyl = cylinder_sampler(dataPC0)
        dataPC1_cyl = cylinder_sampler(dataPC1)
        if self.manual_transform is not None:
            dataPC0_cyl = self.manual_transform(dataPC0_cyl)
            dataPC1_cyl = self.manual_transform(dataPC1_cyl)
        pair_cyl = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y)
        if self.DA:
            pair_cyl.data_augment()
        pair_cyl.normalise()
        return pair_cyl

    def _load_save(self, i):
        if self.pre_transform is not None:
            pc0, pc1, label = self._preproc_clouds_loader(i)
        else:
            pc0, pc1, label = self.clouds_loader(i, nameInPly=self.nameInPly)
        pair = Pair(pos=pc0, pos_target=pc1, y=label)
        pair = self._get_tree(pair, i)
        return pair

    def _get_tree(self, pair, i):
        path = self.filesPC0[i]
        name_tree = os.path.basename(path).split(".")[0] + "_2D_radius" + str(int(self._radius)) + "_" + str(i) + ".p"
        path_treesPC0 = os.path.join(self.preprocessed_dir, "tp3DTree", name_tree)
        if self.reload_trees and osp.isfile(path_treesPC0):
            try:
                file = open(path_treesPC0, "rb")
                tree = pickle.load(file)
                file.close()
                pair.KDTREE_KEY_PC0 = tree
            except:
                print('not able to load tree')
                print(file)
                print(pair)
                tree = KDTree(np.asarray(pair.pos[:, :-1]), leaf_size=10)
                pair.KDTREE_KEY_PC0 = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory is existing
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pair.pos[:, :-1]), leaf_size=10)
            file = open(path_treesPC0, "wb")
            pickle.dump(tree, file)
            file.close()
            pair.KDTREE_KEY_PC0 = tree

        path = self.filesPC1[i]
        name_tree = os.path.basename(path).split(".")[0] + "_2D_radius" + str(int(self._radius)) + "_" + str(i) + ".p"
        path_treesPC1 = os.path.join(self.preprocessed_dir, "tp3DTree", name_tree)
        if self.reload_trees and osp.isfile(path_treesPC1):
            try:
                file = open(path_treesPC1, "rb")
                tree = pickle.load(file)
                file.close()
                pair.KDTREE_KEY_PC1 = tree
            except:
                print('not able to load tree')
                print(file)
                print(pair)
                tree = KDTree(np.asarray(pair.pos_target[:, :-1]), leaf_size=10)
                pair.KDTREE_KEY_PC1 = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory is existing
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pair.pos_target[:, :-1]), leaf_size=10)
            file = open(path_treesPC1, "wb")
            pickle.dump(tree, file)
            file.close()
            pair.KDTREE_KEY_PC1 = tree
        return pair


class Urb3DCDDataset(BaseSiameseDataset): #Urb3DCDDataset Urb3DSimulDataset
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
    FORWARD_CLASS = "forward.urb3DSimulPairCyl.ForwardUrb3DSimulDataset"

    def __init__(self, dataset_opt):
        # self.pre_transform = dataset_opt.get("pre_transforms", None)
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.TTA = False
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.train_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            fix_cyl=self.dataset_opt.fix_cyl,
        )
        self.val_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch= int(self.sample_per_epoch / 2),
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Val"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            fix_cyl=self.dataset_opt.fix_cyl,
        )
        self.test_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
        )

    @property
    def train_data(self):
        if type(self.train_dataset) == list:
            return self.train_dataset[0]
        else:
            return self.train_dataset

    @property
    def val_data(self):
        if type(self.val_dataset) == list:
            return self.val_dataset[0]
        else:
            return self.val_dataset

    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset

    @staticmethod
    def to_ply(pos, label, file, color=OBJECT_COLOR):
        """ Allows to save Urb3DCD predictions to disk using Urb3DCD color scheme
            Parameters
            ----------
            pos : torch.Tensor
                tensor that contains the positions of the points
            label : torch.Tensor
                predicted label
            file : string
                Save location
            """
        to_ply(pos, label, file, color=color)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCDTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                 full_pc=full_pc, full_res=full_res, ignore_label=IGNORE_LABEL)


################################### UTILS #######################################


def to_ply(pos, label, file, color = OBJECT_COLOR, sf = None):
    """ Allows to save Urb3DCD predictions to disk using Urb3DCD color scheme
       Parameters
       ----------
       pos : torch.Tensor
           tensor that contains the positions of the points
       label : torch.Tensor
           predicted label
       file : string
           Save location
    """
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    if max(label)<= color.shape[0]:
        colors = color[np.asarray(label)]
    else:
        colors = color[np.zeros(pos.shape[0], dtype=np.int)]
    if sf is None:
        ply_array = np.ones(
            pos.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                   ("green", "u1"), ("blue", "u1"), ("pred", "u2")]
        )
    else:
        ply_array = np.ones(
            pos.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                   ("green", "u1"), ("blue", "u1"), ("pred", "u2"), ("sf","f4")]
        )
        ply_array["sf"] = np.asarray(sf)
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    ply_array["pred"] = np.asarray(label)
    el = PlyElement.describe(ply_array, "params")
    PlyData([el], byte_order=">").write(file)

