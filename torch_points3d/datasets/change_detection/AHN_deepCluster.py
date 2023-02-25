import os.path as osp
import numpy as np
import torch
import random

from torch_geometric.data import Data, extract_zip, Dataset

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.AHNPairCylinder import to_ply, AHNCylinder
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.metrics.Urb3DCD_deepCluster_tracker import Urb3DCD_deepCluster_tracker
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair

IGNORE_LABEL = -1


# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(URB3DSIMUL_NUM_CLASSES)}

class AHNPairCylinder_DC(AHNCylinder):
    def __init__(self, nb_cluster_kmeans=2, *args, **kwargs):
        self.pseudo_labels = None
        self.pseudo_labels_past = None
        self.pseudo_label_vote = None
        self.stage_dc = "train"
        super().__init__(*args, **kwargs)
        self.nb_cluster_kmeans = nb_cluster_kmeans
        if self._sample_per_epoch > 0:
            self._plabel_counts = self._label_counts
            self._plabels = self._labels

    def reset_pseudo_label(self):
        if self.pseudo_label_vote is None:
            self.pseudo_label_vote = []
            for i in range(len(self.filesPC0)):
                pair = self._load_save(i)
                self.pseudo_label_vote.append(torch.zeros((pair.pos_target.shape[0], self.nb_cluster_kmeans)))
        else:
            self.pseudo_label_vote = [torch.zeros(i.shape, dtype=torch.int8) for i in self.pseudo_label_vote]
        if self.pseudo_labels is not None:
            self.pseudo_labels_past = self.pseudo_labels

    def set_sample_per_epoch(self,n):
        self._sample_per_epoch = n

    def add_batch_pseudo_label(self, data, labels_onehot):
        areas = data.area[data.batch_target]
        for a in range(len(self.pseudo_label_vote)):
            # per area
            self.pseudo_label_vote[a][data.idx_target[areas == a], :] += labels_onehot[areas == a, :]

    def finalise_pseudo_label(self):
        self.pseudo_labels = [self.pseudo_label_vote[a].argmax(axis=1) for a in range(len(self.pseudo_label_vote))]
        self.pseudo_label_vote = None
        if self._sample_per_epoch > 0:
            self._prepare_centers_pl()

    def _load_save(self, i):
        pair = super(AHNPairCylinder_DC, self)._load_save(i)
        if self.pseudo_labels is not None:
            assert pair.y.shape[0] == self.pseudo_labels[i].shape[0]
            pair.y = self.pseudo_labels[i]
        return pair


    def hand_craft_process(self):
        existfile = True
        for idx in range(len(self.filesPC0)):
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(idx)))
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))
        if not self.reload_preproc or not exist_file:
            for idx in range(len(self.filesPC0)):
                pc0, pc1, label = self.clouds_loader(idx)
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

    def load_kmean_as_pseudo_lab(self):
        self.pseudo_labels = []
        for i in range(len(self.filesPC0)):
            data_pc1 = torch.load(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(i)))
            plab_kmean = data_pc1.x
            self.pseudo_labels.append(plab_kmean)

    def clouds_loader_kmean(self, area, nameInPly = "params"):
        print("Loading " + self.filesPC1[area])
        pc = self.cloud_loader(self.filesPC1[area][:-4] + "Kmean.ply", nameInPly=nameInPly, name_feat = "pred")
        gt = pc[:, 3].long() #/!\ Wanted labels should be at the 4th column 0:X 1:Y 2:Z 3:LAbel
        return gt

    def load(self, i):
        pair = super(AHNPairCylinder_DC, self)._load_save(i)
        return pair

    def get_weights_pseudolabel(self):
        nb_elt_class = np.bincount(self.pseudo_labels[0], minlength=self.nb_cluster_kmeans)
        for i in range(1, len(self.pseudo_labels)):
            nb_elt_class += np.bincount(self.pseudo_labels[i], minlength=self.nb_cluster_kmeans)
        weight_classes = np.sqrt(nb_elt_class.mean() / nb_elt_class)
        weight_classes = weight_classes / np.sum(weight_classes)
        self.weight_classes = torch.from_numpy(weight_classes).float()

    def _prepare_centers_pl(self):
        self._centres_for_sampling = []
        for i in range(len(self.filesPC0)):
            pair = self._load_save(i)
            dataPC1 = Data(pos=pair.pos_target, y=self.pseudo_labels[i])
            low_res = self._grid_sphere_sampling(dataPC1)
            centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
            centres[:, :3] = low_res.pos
            centres[:, 3] = i
            centres[:, 4] = low_res.y
            self._centres_for_sampling.append(centres)

        self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
        uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
        # sqrt weight strategy cf base dataset
        uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
        self._plabel_counts = uni_counts / np.sum(uni_counts)
        self._plabels = uni
        self.weight_classes = torch.from_numpy(self._plabel_counts).type(torch.float)

    def _get_random(self):
        # Random cylinder biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._plabels, p=self._plabel_counts)
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
        pair_cyl = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y)
        if self.DA:
            pair_cyl.data_augment()
        pair_cyl.normalise()
        return pair_cyl


class AHNDataset_deepCluster(BaseSiameseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.nb_cluster_kmeans = self.dataset_opt.nb_cluster_kmeans_cd

        self.train_dataset = AHNPairCylinder_DC(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nb_cluster_kmeans=self.nb_cluster_kmeans,
        )
        self.train_dataset_kmeans = AHNPairCylinder_DC(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=-1,
            DA=False,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nb_cluster_kmeans=self.nb_cluster_kmeans,
        )
        self.val_dataset = AHNPairCylinder_DC(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "val"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nb_cluster_kmeans=self.nb_cluster_kmeans,
        )

        self.test_dataset = AHNPairCylinder_DC(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nb_cluster_kmeans=self.nb_cluster_kmeans,
        )
        self.num_classes_orig = self.train_dataset.num_classes

    @property
    def train_data(self):
        if type(self.train_dataset) == list:
            return self.train_dataset[0]
        else:
            return self.train_dataset

    @property
    def train_data_kmeans(self):
        if type(self.train_dataset_kmeans) == list:
            return self.train_dataset_kmeans[0]
        else:
            return self.train_dataset_kmeans

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
        to_ply(pos, label, file)

    def set_nbclass(self, nb_class_cd, nb_class_seg = None):
        self.train_data.num_classes = nb_class_cd
        self.test_data.num_classes = nb_class_cd
        self.train_data.num_classes_cd = nb_class_cd
        self.test_data.num_classes_cd = nb_class_cd

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCD_deepCluster_tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                           full_pc=full_pc, full_res=full_res, ignore_label=IGNORE_LABEL)
