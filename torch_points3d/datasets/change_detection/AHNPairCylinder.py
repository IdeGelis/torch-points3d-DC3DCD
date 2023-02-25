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


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

IGNORE_LABEL: int = 0

AHN_NUM_CLASSES = 4
viridis = cm.get_cmap('viridis', AHN_NUM_CLASSES)
INV_OBJECT_LABEL = {
    0: "Unchanged",
    1: "New building",
    2: "Demolition",
    3: "New clutter",
    # 4: "New water",
    # 5: "New bridge",
}
# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(AHN_NUM_CLASSES)}

OBJECT_COLOR = np.asarray(
    [
        [67, 1, 84],  # 'unchanged'
        [0, 183, 255],  # 'newlyBuilt'
        [0, 12, 235],  # 'deconstructed'
        [0, 217, 33],  # 'newVegetation'
        [255, 230, 0],  # 'vegetationGrowUp'
        [255, 140, 0],  # 'vegetationRemoved'
    ]
)
# OBJECT_COLOR = viridis.colors[:,:3]
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}


class AHN(Dataset):
    """
    Definition of AHN Dataset
    """

    def __init__(self, filePaths="", split="train", DA=False, pre_transform=None, transform=None, preprocessed_dir="",
                 reload_preproc=False, reload_trees=False, ):
        super(AHN, self).__init__(None, None, pre_transform)
        self.class_labels = OBJECT_LABEL
        self.ignore_label = IGNORE_LABEL
        self.preprocessed_dir = preprocessed_dir
        if not osp.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        self.filePaths = filePaths
        self._get_paths()
        self.split = split
        self.DA = DA
        self.pre_transform = pre_transform
        self.transform = None
        self.manual_transform = transform
        self.reload_preproc = reload_preproc
        self.reload_trees = reload_trees
        self.num_classes = AHN_NUM_CLASSES
        self.nb_elt_class = torch.zeros(self.num_classes)
        self.filesPC0_prepoc = [None] * len(self.filesPC0)
        self.filesPC1_prepoc = [None] * len(self.filesPC1)
        self.process()
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
                    if "AHN3" in f.name and "ply" in f.name: #and not "feature" in f.name:
                        self.filesPC0.append(f.path)
                    elif "AHN4" in f.name and "ply" in f.name: #and not "feature" in f.name:
                        self.filesPC1.append(f.path)
                curDir.close()
        globPath.close()

    def size(self):
        return len(self.filesPC0)

    def get_nb_elt_class(self):
        for idx in range(len(self.filesPC0)):
            pc1 = torch.load(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))
            cpt = torch.bincount(pc1.y)
            for c in range(cpt.shape[0]):
                self.nb_elt_class[c] += cpt[c]

    def hand_craft_process(self):
        existfile = True
        for idx in range(len(self.filesPC0)):
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(idx)))
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))
        if not self.reload_preproc or not exist_file:
            for idx in range(len(self.filesPC0)):
                pc0, pc1, label = self.clouds_loader(idx, normalise=False)
                pc0 = Data(pos=pc0)
                pc1 = Data(pos=pc1, y=label)
                if self.pre_transform is not None:
                    pc0 = self.pre_transform(pc0)
                    pc1 = self.pre_transform(pc1)
                cpt = torch.bincount(pc1.y)
                for c in range(cpt.shape[0]):
                    self.nb_elt_class[c] += cpt[c]
                torch.save(pc0, osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(idx)))
                torch.save(pc1, osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))

    def process(self):
        self.hand_craft_process()

    def get(self, idx):
        if self.pre_transform is not None:
            pc0, pc1, label = self._preproc_clouds_loader(idx)
            pc0, pc1 = self.normalise(pc0, pc1)
        else:
            pc0, pc1, label = self.clouds_loader(idx, normalise=True)
        if (hasattr(pc0, "multiscale")):
            batch = MultiScalePair(pos=pc0, pos_target=pc1, y=label)
        else:
            batch = Pair(pos=pc0, pos_target=pc1, y=label)
        return batch.contiguous()

    def clouds_loader(self, area, normalise=False):
        pc1, gt = self.cloud_loader(self.filesPC1[area])
        gt = gt.long()
        pc0, _ = self.cloud_loader(self.filesPC0[area])
        if normalise:
            pc0, pc1 = self.normalise(pc0, pc1)
        return pc0.type(torch.float), pc1.type(torch.float), gt

    def cloud_loader(self, pathPC, cuda=False):
        """
      load a tile and returns points features (normalized xyz + intensity) and
      ground truth
      INPUT:
      pathPC = string, path to the tile of PC
      OUTPUT
      pc_data, [n x 3] float array containing points coordinates and intensity
      lbs, [n] long int array, containing the points semantic labels
      """
        print(pathPC)
        # pc_data, gt = self.lazReader(pathPC, verbose=False)
        pc_data = read_from_ply(pathPC, nameInPly="params")
        # load the point cloud data
        pc = torch.from_numpy(pc_data[:,:3])  # .type(torch.float)
        gt = torch.from_numpy(pc_data[:, 3])
        if cuda:  # put the cloud data on the GPU memory
            pc = pc.cuda()
        return pc, gt

    def lazReader(self, lazFile, verbose=True):
        if verbose:
            print('Reading ' + lazFile)
        file = laspy.read(lazFile)
        coords = np.vstack((file.x, file.y, file.z)).transpose()
        try:
            gt = file.change_classification
        except:
            gt = None
        return coords, gt

    def _preproc_clouds_loader(self, area):
        data_pc0 = torch.load(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(area)))
        data_pc1 = torch.load(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(area)))
        return data_pc0.pos, data_pc1.pos, data_pc1.y

    def normalise(self, pc0, pc1):
        min0 = torch.unsqueeze(pc0.min(0)[0], 0)
        min1 = torch.unsqueeze(pc1.min(0)[0], 0)
        minG = torch.cat((min0, min1), axis=0).min(0)[0]

        # normalizing data with the same xmean ymean zmin
        pc0[:, 0] = (pc0[:, 0] - minG[0])  # x
        pc0[:, 1] = (pc0[:, 1] - minG[1])  # y
        pc0[:, 2] = (pc0[:, 2] - minG[2])  # z

        pc1[:, 0] = (pc1[:, 0] - minG[0])  # x
        pc1[:, 1] = (pc1[:, 1] - minG[1])  # y
        pc1[:, 2] = (pc1[:, 2] - minG[2])  # z

        max0 = torch.unsqueeze(pc0.max(0)[0], 0)
        max1 = torch.unsqueeze(pc1.max(0)[0], 0)
        maxG = torch.cat((max0, max1), axis=0).max(0)[0]
        pc0[:, 0] = (pc0[:, 0] / maxG[0])  # x
        pc0[:, 1] = (pc0[:, 1] / maxG[1])  # y
        pc0[:, 2] = (pc0[:, 2] / maxG[2])  # z

        pc1[:, 0] = (pc1[:, 0] / maxG[0])  # x
        pc1[:, 1] = (pc1[:, 1] / maxG[1])  # y
        pc1[:, 2] = (pc1[:, 2] / maxG[2])  # z
        return pc0, pc1


class AHNSphere(AHN):
    """ Small variation of Urb3DSimul that allows random sampling of spheres
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

    def __init__(self, sample_per_epoch=100, radius=2, fix_cyl = False, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = GridSampling3D(size=radius / 10.0)
        super().__init__(*args, **kwargs)
        self.fix_cyl = fix_cyl
        self._prepare_centers()
        # Trees are built in case it needs, now don't need to compute anymore trees
        self.reload_trees = True
        self.TTA = False

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return self.grid_regular_centers.shape[0]

    def get(self, idx):
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
            pair_spheres = Pair(pos=dataPC0_sphere.pos, pos_target=dataPC1_sphere.pos, y=dataPC1_sphere.y)
            pair_spheres.normalise()
            return pair_spheres.contiguous()

    # def process(self):  # We have to include this method, otherwise the parent class skips processing
    #     super().process()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_sel = centre[3].int()  # ---> ici choix du pc correspondant si pls pc chargés
        pair = self._load_save(area_sel)
        sphere_sampler = SphereSampling(self._radius, centre[:3], align_origin=False)
        dataPC0 = Data(pos=pair.pos)
        setattr(dataPC0, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
        dataPC1 = Data(pos=pair.pos_target, y=pair.y)
        setattr(dataPC1, SphereSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
        dataPC0_sphere = sphere_sampler(dataPC0)
        dataPC1_sphere = sphere_sampler(dataPC1)
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
                centres[:, 3] = i #area
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
            # self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)

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
            pc0, pc1, label = self.clouds_loader(i, normalise=False)
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
            # test if tp3D directory is existing
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
            # test if tp3D directory is existing
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pc1), leaf_size=10)
            file = open(path_treesPC1, "wb")
            pickle.dump(tree, file)
            file.close()
            pair.KDTREE_KEY_PC1 = tree
        return pair


class AHNCylinder(AHNSphere):
    def get(self, idx):
        if self._sample_per_epoch > 0:
            if self.fix_cyl:
                centre = self._centres_for_sampling_fixed[idx, :3]
                # Choice on the corresponding PC if several available
                area_sel = self._centres_for_sampling_fixed[idx, 3].int()
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
                pair_cylinders.normalise()
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
                pair_cylinders = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                                      idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel)
                try:
                    pair_cylinders.normalise()
                    pair_correct = True
                except:
                    print("pair not correct")
                    print(pair_cylinders.pos.shape)
                    print(pair_cylinders.pos_target.shape)
                    idx += 1
            return pair_cylinders

    def _get_random(self):
        # Random cylinder biased towards getting more low frequency classes
        if self.split =="val":
            proba = [0.05, 0.20, 0.5, 0.25]
            # proba = self._label_counts
            chosen_label = np.random.choice(self._labels, p=proba)
        else:
            chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        # Choice on the corresponding PC if several available
        area_sel = centre[3].int()
        pair = self._load_save(area_sel)
        cylinder_sampler = CylinderSampling(self._radius, centre[:3], align_origin=False)
        dataPC0 = Data(pos=pair.pos)
        setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
        dataPC1 = Data(pos=pair.pos_target, y=pair.y)
        setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
        dataPC0_cyl = cylinder_sampler(dataPC0)
        dataPC1_cyl = cylinder_sampler(dataPC1)
        pair_cyl = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y)
        if self.DA:
            pair_cyl.data_augment()
        pair_cyl.normalise()
        return pair_cyl

    def _load_save(self, i):
        if self.pre_transform is not None:
            pc0, pc1, label = self._preproc_clouds_loader(i)
        else:
            pc0, pc1, label = self.clouds_loader(i, normalise=False)
        pair = Pair(pos=pc0, pos_target=pc1, y=label)
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
                tree = KDTree(np.asarray(pc0[:, :-1]), leaf_size=10)
                pair.KDTREE_KEY_PC0 = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory is existing
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pc0[:, :-1]), leaf_size=10)
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
                tree = KDTree(np.asarray(pc1[:, :-1]), leaf_size=10)
                pair.KDTREE_KEY_PC1 = tree
        else:
            # tree not existing yet should be saved
            # test if tp3D directory is existing
            if not os.path.isdir(os.path.join(self.preprocessed_dir, "tp3DTree")):
                os.makedirs(os.path.join(self.preprocessed_dir, "tp3DTree"))
            tree = KDTree(np.asarray(pc1[:, :-1]), leaf_size=10)
            file = open(path_treesPC1, "wb")
            pickle.dump(tree, file)
            file.close()
            pair.KDTREE_KEY_PC1 = tree
        return pair


class AHNDataset(BaseSiameseDataset):
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
    FORWARD_CLASS = "forward.AHNPairCyl.ForwardAHNDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.TTA = False
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.train_dataset = AHNCylinder(
            filePaths=self.dataset_opt.dataTrainFile,
            fix_cyl=self.dataset_opt.fix_cyl,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
        )
        self.val_dataset = AHNCylinder(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch=500,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Val"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
        )
        self.test_dataset = AHNCylinder(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
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

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCDTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                 full_pc=full_pc, full_res=full_res)


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
    el = PlyElement.describe(ply_array, "params")
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
