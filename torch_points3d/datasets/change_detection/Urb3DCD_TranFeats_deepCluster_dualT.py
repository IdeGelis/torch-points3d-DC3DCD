import os.path as osp
import os
import numpy as np
import torch
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors, KDTree
import pickle
from torch_geometric.data import Data, extract_zip, Dataset

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.Urb3DSimulPairCylinder import to_ply, Urb3DSimulCylinder
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.metrics.Urb3DCD_deepCluster_dualTask_tracker import Urb3DCD_deepCluster_dualT_tracker
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair

IGNORE_LABEL = -1


# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(URB3DSIMUL_NUM_CLASSES)}

class Urb3DCDPairCylinder_DC_dualT(Urb3DSimulCylinder):
    def __init__(self, nb_cluster_kmeans_cd=2, nb_cluster_kmeans_seg=2, load_GT=True, m3c2_gt=False,  c2c_gt=False,*args, **kwargs):
        self.pseudo_labels_cd = None
        self.pseudo_labels_past = None
        self.pseudo_labels_seg0 = None
        self.pseudo_labels_seg1 = None
        self.pseudo_label_vote_cd = None
        self.pseudo_label_vote_seg0 = None
        self.pseudo_label_vote_seg1 = None
        self.pseudo_labels_changeSeg = None
        self.pseudo_label_vote_changeSeg = None
        self.m3c2_gt = m3c2_gt
        self.c2c_gt = c2c_gt
        if self.m3c2_gt and self.c2c_gt:
            raise Exception("Both M3C2 and C2C is set to True, impossible choose only one or zero")

        self.stage_dc = "train"
        self.n_feats = 10
        super().__init__(*args, **kwargs)
        self.nb_cluster_kmeans_cd = nb_cluster_kmeans_cd
        self.nb_cluster_kmeans_seg = nb_cluster_kmeans_seg
        self.weight_classes_cd = None
        self.weight_classes_seg = None
        self.load_GT = load_GT
        if self._sample_per_epoch > 0:
            self._plabel_counts_cd = self._label_counts
            self._plabels_cd = self._labels
        self.binaryCD_segsem = False

    def set_binaryCD_segsem(self):
        self.binaryCD_segsem = True

    def unset_binaryCD_segsem(self):
        self.binaryCD_segsem = False

    def _get_paths(self):
        self.filesPC0 = []
        self.filesPC1 = []
        globPath = os.scandir(self.filePaths)
        for dir in globPath:
            if dir.is_dir():
                curDir = os.scandir(dir)
                for f in curDir:
                    if f.name == "pointCloud0featuresRad5KNN10.ply":
                        self.filesPC0.append(f.path)
                    elif f.name == "pointCloud1featuresRad5KNN10.ply":
                        self.filesPC1.append(f.path)
                curDir.close()
        globPath.close()

    def reset_pseudo_label(self):
        if self.pseudo_label_vote_cd is None:
            self.pseudo_label_vote_cd = []
            for i in range(len(self.filesPC0)):
                pair = self._load_save(i)
                self.pseudo_label_vote_cd.append(torch.zeros((pair.pos_target.shape[0], self.nb_cluster_kmeans_cd)))
        else:
            self.pseudo_label_vote_cd = [torch.zeros(i.shape, dtype=torch.int8) for i in self.pseudo_label_vote_cd]

        if self.pseudo_label_vote_seg0 is None:
            self.pseudo_label_vote_seg0 = []
            for i in range(len(self.filesPC0)):
                pair = self._load_save(i)
                self.pseudo_label_vote_seg0.append(torch.zeros((pair.pos.shape[0], self.nb_cluster_kmeans_seg)))
        else:
            self.pseudo_label_vote_seg0 = [torch.zeros(i.shape, dtype=torch.int8) for i in self.pseudo_label_vote_seg0]

        if self.pseudo_label_vote_seg1 is None:
            self.pseudo_label_vote_seg1 = []
            for i in range(len(self.filesPC0)):
                pair = self._load_save(i)
                self.pseudo_label_vote_seg1.append(torch.zeros((pair.pos_target.shape[0], self.nb_cluster_kmeans_seg)))
        else:
            self.pseudo_label_vote_seg1 = [torch.zeros(i.shape, dtype=torch.int8) for i in self.pseudo_label_vote_seg1]

        if self.pseudo_labels_cd is not None:
            self.pseudo_labels_past = self.pseudo_labels_cd

        if self.binaryCD_segsem:
            if self.pseudo_label_vote_changeSeg is None:
                self.pseudo_label_vote_changeSeg = []
                for i in range(len(self.filesPC0)):
                    pair = self._load_save(i)
                    self.pseudo_label_vote_changeSeg.append(torch.zeros((pair.pos_target.shape[0], 2)))
            else:
                self.pseudo_label_vote_changeSeg = [torch.zeros(i.shape, dtype=torch.int8) for i in
                                                    self.pseudo_label_vote_changeSeg]

    def add_batch_pseudo_label(self, data, labels_onehot_cd = None, labels_onehot_seg0 = None, labels_onehot_seg1 = None):
        areas_t = data.area[data.batch_target]
        areas = data.area[data.batch]
        for a in range(len(self.pseudo_label_vote_cd)):
            # per area
            if labels_onehot_cd is not None:
                self.pseudo_label_vote_cd[a][data.idx_target[areas_t == a], :] += labels_onehot_cd[areas_t == a, :]
            if labels_onehot_seg0 is not None:
                self.pseudo_label_vote_seg0[a][data.idx[areas == a], :] += labels_onehot_seg0[areas == a, :]
            if labels_onehot_seg1 is not None:
                self.pseudo_label_vote_seg1[a][data.idx_target[areas_t == a], :] += labels_onehot_seg1[areas_t == a, :]

    def add_batch_pseudo_label_change(self, data, labels_onehot_changeSeg):
        areas_t = data.area[data.batch_target]
        for a in range(len(self.pseudo_label_vote_cd)):
            # per area
            self.pseudo_label_vote_changeSeg[a][data.idx_target[areas_t == a], :] += labels_onehot_changeSeg[
                                                                                     areas_t == a, :]

    def finalise_pseudo_label(self):
        self.pseudo_labels_cd = [self.pseudo_label_vote_cd[a].argmax(axis=1) for a in
                                 range(len(self.pseudo_label_vote_cd))]
        self.pseudo_label_vote_cd = None

        self.pseudo_labels_seg0 = [self.pseudo_label_vote_seg0[a].argmax(axis=1) for a in
                                   range(len(self.pseudo_label_vote_seg0))]
        self.pseudo_label_vote_seg0 = None

        self.pseudo_labels_seg1 = [self.pseudo_label_vote_seg1[a].argmax(axis=1) for a in
                                   range(len(self.pseudo_label_vote_seg1))]
        self.pseudo_label_vote_seg1 = None

        if self.binaryCD_segsem:
            self.pseudo_labels_changeSeg = [self.pseudo_label_vote_changeSeg[a].argmax(axis=1) for a in
                                            range(len(self.pseudo_label_vote_changeSeg))]
            self.pseudo_label_vote_changeSeg = None
        if self._sample_per_epoch > 0:
            self._prepare_centers_pl()

    def _preproc_clouds_loader(self, area):
        data_pc0 = torch.load(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(area)))
        data_pc1 = torch.load(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(area)))
        return data_pc0.pos, data_pc1.pos, data_pc1.y, data_pc0.x, data_pc1.x, data_pc1.m3c2

    def hand_craft_process(self, comp_normal=False):
        existfile = True
        for idx in range(len(self.filesPC0)):
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc0_{}.pt'.format(idx)))
            exist_file = existfile and osp.isfile(osp.join(self.preprocessed_dir, 'pc1_{}.pt'.format(idx)))
        if not self.reload_preproc or not exist_file:
            for idx in range(len(self.filesPC0)):
                pc0, pc1, label, feat0, feat1 = self.clouds_loader(idx, nameInPly=self.nameInPly, feats=True)
                pc0 = Data(pos=pc0, x=feat0)
                if self.m3c2_gt  or self.c2c_gt:
                    gt_m3c2 = self.get_M3C2_C2C_gt(self.filesPC1[idx], pc1)
                    pc1 = Data(pos=pc1, y=label, x=feat1, m3c2=gt_m3c2)
                else:
                    pc1 = Data(pos=pc1, y=label, x=feat1, m3c2=None)
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

    def _get_tree(self, pair, idx):
        path = self.filesPC0[idx]
        name_tree = os.path.basename(path).split(".")[0] + "_2D_radius" + str(int(self._radius)) + "_" + str(idx) + ".p"
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

        path = self.filesPC1[idx]
        name_tree = os.path.basename(path).split(".")[0] + "_2D_radius" + str(int(self._radius)) + "_" + str(idx) + ".p"
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

    def _load_save(self, i):
        if self.pre_transform is not None:
            pc0, pc1, label, feats0, feats1, m3c2 = self._preproc_clouds_loader(i)
        else:
            pc0, pc1, label, feats0, feats1 = self.clouds_loader(i, nameInPly=self.nameInPly, feats=True)
            if self.m3c2_gt or self.c2c_gt:
                m3c2 = self.get_M3C2_C2C_gt(self.filesPC1[i], pc1)
        pair = Pair(pos=pc0, pos_target=pc1, y=label, x=feats0, x_target=feats1)
        pair = self._get_tree(pair, i)
        if self.m3c2_gt or self.c2c_gt:
            pair.gt_change = m3c2
        else:
            pair.gt_change = pair.y
        if self.pseudo_labels_changeSeg is not None:
            pair.gt_change = self.pseudo_labels_changeSeg[i]
        if self.pseudo_labels_cd is not None:
            assert pair.y.shape[0] == self.pseudo_labels_cd[i].shape[0]
            pair.y = self.pseudo_labels_cd[i]
        if self.pseudo_labels_seg1 is not None:
            assert pair.y.shape[0] == self.pseudo_labels_seg1[i].shape[0]
            pair.y1 = self.pseudo_labels_seg1[i]
        if self.pseudo_labels_seg0 is not None:
            assert pair.pos.shape[0] == self.pseudo_labels_seg0[i].shape[0]
            pair.y0 = self.pseudo_labels_seg0[i]
        return pair

    def load(self, i):
        if self.pre_transform is not None:
            pc0, pc1, label, feats0, feats1, _ = self._preproc_clouds_loader(i)
        else:
            pc0, pc1, label, feats0, feats1 = self.clouds_loader(i, nameInPly=self.nameInPly, feats=True)
        pair = Pair(pos=pc0, pos_target=pc1, y=label, x=feats0, x_target=feats1)
        pair = self._get_tree(pair, i)
        return pair

    def clouds_loader(self, area, nameInPly="params", feats=False):
        print("Loading " + self.filesPC1[area])
        pc = self.cloud_loader(self.filesPC1[area], nameInPly=nameInPly, nb_feat=self.n_feats)
        pc1 = pc[:, :3]
        gt = pc[:, 3].long()  # /!\ Wanted labels should be at the 4th column 0:X 1:Y 2:Z 3:LAbel
        feats1 = pc[:, 4:]
        pc = self.cloud_loader(self.filesPC0[area], nameInPly=nameInPly, nb_feat=self.n_feats)
        pc0 = pc[:, :3]
        feats0 = pc[:, 4:]
        if feats:
            return pc0.type(torch.float), pc1.type(torch.float), gt, feats0, feats1
        else:
            return pc0.type(torch.float), pc1.type(torch.float), gt

    def cloud_loader(self, pathPC, cuda=False, nameInPly=None, name_feat="label_ch", nb_feat=0):
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
            pc_data = self.read_from_ply(pathPC, nameInPly="params", name_feat=name_feat, n_feat=nb_feat)
        else:
            pc_data = self.read_from_ply(pathPC, nameInPly=nameInPly, name_feat=name_feat, n_feat=nb_feat)
        # load the point cloud data
        pc_data = torch.from_numpy(pc_data)  # .type(torch.float)

        if cuda:  # put the cloud data on the GPU memory
            pc_data = pc_data.cuda()
        return pc_data

    def get_weights_pseudolabel(self):
        nb_elt_class = np.bincount(self.pseudo_labels_cd[0], minlength=self.nb_cluster_kmeans_cd)
        for i in range(1, len(self.pseudo_labels_cd)):
            nb_elt_class += np.bincount(self.pseudo_labels_cd[i], minlength=self.nb_cluster_kmeans_cd)
        # weight_classes = 1 - (nb_elt_class / nb_elt_class.sum())
        weight_classes = np.sqrt(nb_elt_class.mean() / nb_elt_class)
        weight_classes = weight_classes / np.sum(weight_classes)
        self.weight_classes_cd = torch.from_numpy(weight_classes).float()

        # for segmentation rgoup point from PC0 and PC1
        nb_elt_class = np.bincount(self.pseudo_labels_seg0[0], minlength=self.nb_cluster_kmeans_seg)
        for i in range(1, len(self.pseudo_labels_seg0)):
            nb_elt_class += np.bincount(self.pseudo_labels_seg0[i], minlength=self.nb_cluster_kmeans_seg)
        for i in range(0, len(self.pseudo_labels_seg1)):
            nb_elt_class += np.bincount(self.pseudo_labels_seg1[i], minlength=self.nb_cluster_kmeans_seg)
        # weight_classes = 1 - (nb_elt_class / nb_elt_class.sum())
        weight_classes = np.sqrt(nb_elt_class.mean() / nb_elt_class)
        weight_classes = weight_classes / np.sum(weight_classes)
        self.weight_classes_seg = torch.from_numpy(weight_classes).float()

    def _prepare_centers_pl(self):
        """
        Idem _prepare_centers_pl without dual task, the choice is still made according CD task
        """
        self._centres_for_sampling = []
        for i in range(len(self.filesPC0)):
            pair = self._load_save(i)
            dataPC1 = Data(pos=pair.pos_target, y=self.pseudo_labels_cd[i])
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
        self._plabel_counts_cd = uni_counts / np.sum(uni_counts)
        self._plabels_cd = uni
        self.weight_classes_cd = torch.from_numpy(self._plabel_counts_cd).type(torch.float)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            pair_correct = False
            while not pair_correct and idx < self.grid_regular_centers.shape[0]:
                centre = self.grid_regular_centers[idx, :3]
                area_sel = self.grid_regular_centers[idx, 3].int()
                pair = self._load_save(area_sel)
                cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
                dataPC0 = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]).reshape(-1), ys=pair.y0, x=pair.x)
                setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
                dataPC1 = Data(pos=pair.pos_target, y=pair.y, idx=torch.arange(pair.pos_target.shape[0]).reshape(-1),
                               ys=pair.y1, gt_change=pair.gt_change, x=pair.x_target)
                setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
                dataPC0_cyl = cylinder_sampler(dataPC0)
                dataPC1_cyl = cylinder_sampler(dataPC1)
                try:
                    if self.manual_transform is not None:
                        dataPC0_cyl = self.manual_transform(dataPC0_cyl)
                        dataPC1_cyl = self.manual_transform(dataPC1_cyl)
                    if pair.y0 is not None:
                        pair_cylinders = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                                              idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel,
                                              y0=dataPC0_cyl.ys, y1=dataPC1_cyl.ys, gt_change=dataPC1_cyl.gt_change,
                                              x=dataPC0_cyl.x, x_target=dataPC1_cyl.x)
                    else:
                        pair_cylinders = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                                              idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel,
                                              y0=None, y1=None, gt_change=dataPC1_cyl.gt_change,
                                              x=dataPC0_cyl.x, x_target=dataPC1_cyl.x)
                    if self.DA:
                        pair_cylinders.data_augment()
                    pair_cylinders.normalise()
                    pair_correct = True
                except:
                    print('pair not correct')
                    idx += 1
            return pair_cylinders

    def _get_random(self):
        """
        Add y0 and y1 pseudo-label for segmentation for each PC separately
        """
        # Random cylinder biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._plabels_cd, p=self._plabel_counts_cd)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        #  choice of the corresponding PC if several PCs are loaded
        area_sel = centre[3].int()
        pair = self._load_save(area_sel)
        cylinder_sampler = CylinderSampling(self._radius, centre[:3], align_origin=False)
        dataPC0 = Data(pos=pair.pos, ys=pair.y0, x=pair.x)
        setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)
        dataPC1 = Data(pos=pair.pos_target, y=pair.y, ys=pair.y1, gt_change=pair.gt_change, x=pair.x_target)
        setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)
        dataPC0_cyl = cylinder_sampler(dataPC0)
        dataPC1_cyl = cylinder_sampler(dataPC1)
        if pair.y0 is not None:
            pair_cyl = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y, y0=dataPC0_cyl.ys,
                            y1=dataPC1_cyl.ys, gt_change=dataPC1_cyl.gt_change, x=dataPC0_cyl.x, x_target=dataPC1_cyl.x)
        else:
            pair_cyl = Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y, y0=None, Y1=None,
                            gt_change=dataPC1_cyl.gt_change, x=dataPC0_cyl.x, x_target=dataPC1_cyl.x)

        if self.DA:
            pair_cyl.data_augment()
        pair_cyl.normalise()
        return pair_cyl

    def read_from_ply(self, filename, nameInPly="params", name_feat="label_ch", n_feat=0):
        """read XYZ for each vertex."""
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 4 + n_feat], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            vertices[:, 3] = plydata[nameInPly].data[name_feat]
            for i in range(n_feat):
                vertices[:, 4 + i] = plydata[nameInPly].data["f{}".format(str(i))]
        return vertices

    def get_M3C2_C2C_gt(self, filename, pc, nameInPly="vertex"):
        if self.m3c2_gt:
            filename = os.path.join(os.path.dirname(filename), "pointCloud1_M3C2.ply")
        elif self.c2c_gt:
            filename = os.path.join(os.path.dirname(filename), "pointCloud1_C2C_Otsu.ply")
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            if self.m3c2_gt:
                vertices[:, 3] = plydata[nameInPly].data["scalar_significant_change"]
            elif self.c2c_gt:
                vertices[:, 3] = plydata[nameInPly].data["C2C_label"]
        vertices = torch.from_numpy(vertices)
        assert torch.all(vertices[:, :3] == pc)
        return vertices[:, 3].long()


class Urb3DCDDataset_deepCluster_dualT(BaseSiameseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.nb_cluster_kmeans_cd = self.dataset_opt.nb_cluster_kmeans_cd
        self.nb_cluster_kmeans_seg = self.dataset_opt.nb_cluster_kmeans_seg

        self.train_dataset = Urb3DCDPairCylinder_DC_dualT(
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
            nb_cluster_kmeans_cd=self.dataset_opt.nb_cluster_kmeans_cd,
            nb_cluster_kmeans_seg=self.dataset_opt.nb_cluster_kmeans_seg,
            m3c2_gt=self.dataset_opt.m3c2_binary_GT,
            c2c_gt=self.dataset_opt.c2c_binary_GT
        )
        self.train_dataset_kmeans = Urb3DCDPairCylinder_DC_dualT(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=-1,
            DA=False,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            m3c2_gt=self.dataset_opt.m3c2_binary_GT,
            c2c_gt=self.dataset_opt.c2c_binary_GT
        )
        self.val_dataset = Urb3DCDPairCylinder_DC_dualT(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "val"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            nb_cluster_kmeans_cd=self.dataset_opt.nb_cluster_kmeans_cd,
            nb_cluster_kmeans_seg=self.dataset_opt.nb_cluster_kmeans_seg,
        )
        self.test_dataset = Urb3DCDPairCylinder_DC_dualT(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            reload_trees=self.dataset_opt.load_trees,
            nameInPly=self.dataset_opt.nameInPly,
            nb_cluster_kmeans_cd=self.dataset_opt.nb_cluster_kmeans_cd,
            nb_cluster_kmeans_seg=self.dataset_opt.nb_cluster_kmeans_seg,
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

    def set_nbclass(self, nb_class_cd, nb_class_seg):
        self.train_data.num_classes_cd = nb_class_cd
        self.val_data.num_classes_cd = nb_class_cd
        self.test_data.num_classes_cd = nb_class_cd

        self.train_data.num_classes = nb_class_cd
        self.val_data.num_classes = nb_class_cd
        self.test_data.num_classes = nb_class_cd

        self.train_data.num_classes_seg = nb_class_seg
        self.val_data.num_classes_seg = nb_class_seg
        self.test_data.num_classes_seg = nb_class_seg

        self.train_data.nb_cluster_kmeans_cd = nb_class_cd
        self.val_data.nb_cluster_kmeans_cd = nb_class_cd
        self.test_data.nb_cluster_kmeans_cd = nb_class_cd

        self.train_data.nb_cluster_kmeans_seg = nb_class_seg
        self.val_data.nb_cluster_kmeans_seg = nb_class_seg
        self.test_data.nb_cluster_kmeans_seg = nb_class_seg

        self.nb_cluster_kmeans_cd = nb_class_cd
        self.nb_cluster_kmeans_seg = nb_class_seg

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCD_deepCluster_dualT_tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                                 full_pc=full_pc, full_res=full_res, ignore_label=IGNORE_LABEL)
