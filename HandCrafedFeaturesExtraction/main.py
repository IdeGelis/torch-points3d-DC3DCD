#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:26:10 2020

@author: degelis
"""
import os
import os.path as osp
import time
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import joblib
import  cv2
from sklearn.ensemble import RandomForestClassifier
import time
from plyfile import PlyData, PlyElement
import laspy

from FeatureExtract import getFeaturesfromPDAL, neighborhoodFeatures, pdalDTM, normalizeHeight, stability, pointWiseFeats, mppointWiseFeats
from evaluation import Eval

class Dataset:
    def __init__(self, paths, namePC, colLbs):
        """
        IN: paths list : list : of path where PCs are available
            namePC : str : ex: pointCloud name of the file (not including date)
            colLbs : int :  index of the column containing labels for each points
        """
        self.paths = paths
        self.namePC = namePC
        self.colLbs = colLbs
        self.ptsFeats = None
        self.labels = None
        self.pcs = []
        self.pcs_otherdate = []
        self.paths_pc = []

    def loadDS(self, numPC, numPCOtherDate, knn, radius, try2LoadFeats = True, saveTXT = False, extension = "ply", nameInPly = "", binary = False):
        self.knn = knn
        self.radius = radius
        for path in self.paths:
            for f in glob(path + "*/" +self.namePC + str(numPC) + "." + extension):
            # for f in glob(path + "*/AHN4_*." + extension):
                if "features" in f or "Kmean" in f:
                    continue
                print(f)
                self.paths_pc.append(f)
                pc = PointCloud()
                pc.load(f, nameInPly=nameInPly)
                self.pcs.append(pc)
                lb = pc.getLabels(self.colLbs)
                pc_otherdate = PointCloud()
                pc_otherdate.load(f[:-5] + str(numPCOtherDate) + "." + extension, nameInPly=nameInPly)
                self.pcs_otherdate.append(pc_otherdate)
                time_cpt = time.time()
                ptsFeats = pc.getFeaturesFromDivision(pc_otherdate, knn, radius, try2Load = try2LoadFeats, saveNPY = True, saveTXT = saveTXT, savePLY=True)
                print("--- %s sec ---\n" % str((time.time() - time_cpt)))
                if self.ptsFeats is None:
                    self.ptsFeats = ptsFeats
                else:
                    self.ptsFeats = np.concatenate((self.ptsFeats, ptsFeats), axis=0)

                if self.labels is None:
                    self.labels = lb
                else:
                    self.labels = np.concatenate((self.labels, lb), axis=0)
        if binary:
            self.labels[self.labels>1] = 1
        print('Total Number of points : ' + str(self.labels.shape[0]))
        print('Loading done')

    def getWeights(self, nb_class):
        if self.labels is None:
            self.loadDS()
        weights = np.ones(nb_class)
        for i in range(nb_class):
            weights[i] -= np.sum(self.labels==i)/self.labels.shape[0]
        return weights

    def save_ds(self, new_labels):
        new_labels = self.get_per_pc_attr(new_labels)
        for p in range(len(self.pcs)):
            self.pcs[p].save(new_labels[p])


    def get_per_pc_attr(self, feat):
        feat_sep = []
        start = 0
        end = 0
        for p in range(len(self.pcs)):
            end += self.pcs[p].pc.shape[0]
            feat_sep.append(feat[start:end])
            start += self.pcs[p].pc.shape[0]
        return feat_sep

    def get_per_pc_feat(self, save_ply=True):
        for p in range(len(self.pcs)):
            self.pcs_otherdate[p].getFeaturesFromDivision(self.pcs[p], self.knn, self.radius, try2Load = True,
                                                        saveNPY = True, saveTXT = False, savePLY=save_ply)






class PointCloud:
    def __init__(self):
        self.path = ""
        self.pc = None
        self.divided = False

    def load(self,path, nameInPly = ""):
        self.path = path
        if "npy" in path:
            self.pc = np.load(path)
        else:
            if "ply" in path:
                self.pc = read_from_ply(path,nameInPly=nameInPly)
            else:
                if "laz" or "LAZ" in path:
                    self.pc, gt = lazReader(path, verbose=False)
                    if gt is not None:
                        self.pc = np.hstack((self.pc, np.expand_dims(gt,axis=1)))
                else:
                    print("Unknown format, available: ply, laz or npy")
                    sys.exit(1)
        self.features = np.zeros((self.shape()[0], 10))

    def fromNumpy(self,pc):
        self.pc = pc
        self.features = np.zeros((self.shape()[0], 10))

    def save(self, lab):
        to_ply(self.pc, lab, self.path[:-4] + "Kmean.ply")


    def getFeatures(self,pcOtherDate, knn, radius):
        self.features = np.zeros((self.shape()[0], 10))

        # PDAL part
        # pdalFeats 0:X 1:Y 2:Z 3:Nx 4:Ny 5:Nz 6:Curvature
        pdalFeats = getFeaturesfromPDAL(self.pc[:, :3])
        DTM = pdalDTM(self.pc[:, :3])

        # Normal
        self.features[:, 0:3] = pdalFeats[:, 3:6]

        neighborFeats, normHeight, stab = mppointWiseFeats(self.pc[:, :3], DTM, pcOtherDate.pc[:, :3], knn=knn,
                                                           radius=radius, fill_in=0)

        # neighborFeats = neighborhoodFeatures(self.pc[:,:3], knn = 10)

        # PCA distribution features
        # Lt = 1 (E2/E1)
        self.features[:, 3] = neighborFeats[:, 0]
        # Pt = (E2 -E3)/E1
        self.features[:, 4] = neighborFeats[:, 1]
        # Ot = racine 3 (E1E2E3)
        self.features[:, 5] = neighborFeats[:, 2]

        print("PCA distribution features done")

        # Z neighborhood
        # zRk
        self.features[:, 6] = neighborFeats[:, 3]
        # zRg
        self.features[:, 7] = neighborFeats[:, 4]

        print("Neighborhood Z stats done")

        # Normalized height
        # normHeight = normalizeHeight(self.pc[:,:3], DTM)
        self.features[:, 8] = normHeight
        print("Normalized height done")

        # Stability
        # stab = stability(self.pc[:,:3], pcOtherDate.pc[:,:3], radius=2)
        self.features[:, 9] = stab
        print("Stability done")
        return self.features


    def getFeaturesFromDivision(self, pcOtherDate, knn, radius, try2Load = True, saveNPY = True, saveTXT = False, savePLY=True, nb_divi = 0):
        if os.path.isfile(self.path[:-4] + "featuresRad" + str(radius) + "KNN" + str(knn) + ".npy") and try2Load:
            self.features = np.load(self.path[:-4] + "featuresRad" + str(radius) + "KNN" + str(knn) + ".npy")[:,3:] # get rid of points coordinates
        else:
            coef = 2
            print(self.shape()[0])
            if self.shape()[0] > 1e9:
                self.divided = True
                axis, min, max = self.largerAxis(pcOtherDate)
                indexDiv = np.arange(min, max + (max - min) / coef, (max - min) / coef)
                # cpt = np.zeros((self.shape()[0],1))
                for i in range(len(indexDiv) - 1):
                    ind = np.where(np.logical_and(np.greater_equal(self.pc[:, axis], indexDiv[i]-2*radius),
                                                  np.greater(indexDiv[i + 1]+2*radius, self.pc[:, axis])))[0]
                    miniPc = PointCloud()
                    miniPc.fromNumpy(self.pc[ind, :])
                    ind_otherDate = np.where(np.logical_and(np.greater_equal(pcOtherDate.pc[:, axis], indexDiv[i]-2*radius),
                                                            np.greater(indexDiv[i + 1]+2*radius, pcOtherDate.pc[:, axis])))[0]
                    miniPcOtherDate = PointCloud()
                    miniPcOtherDate.fromNumpy(pcOtherDate.pc[ind_otherDate, :])
                    feats = miniPc.getFeaturesFromDivision(miniPcOtherDate, knn, radius, try2Load = False, saveNPY = False, saveTXT = False, nb_divi=nb_divi+1)
                    ind_feat = np.where(np.logical_and(np.greater_equal(self.pc[:, axis], indexDiv[i]),
                                                  np.greater(indexDiv[i + 1], self.pc[:, axis])))[0]
                    ind_feat_for_feat = np.where(np.logical_and(np.greater_equal(miniPc.pc[:, axis], indexDiv[i]),
                                                       np.greater(indexDiv[i + 1], miniPc.pc[:, axis])))[0]
                    import pdb;
                    pdb.set_trace()
                    self.features[ind_feat, :] = feats[ind_feat_for_feat,:]
                    # cpt[ind,:] +=1
                # self.features /=cpt
            else:
                print("number of division of the PC : " + str(nb_divi))
                self.getFeatures(pcOtherDate, knn, radius)
            if saveNPY:
                print("Saving "+self.path[:-4] + "featuresRad" + str(radius) + "KNN" + str(knn) + ".npy")
                np.save(self.path[:-4] + "featuresRad" + str(radius) + "KNN" + str(knn) + ".npy", np.concatenate((self.pc[:,:3],self.features), axis=1))
        if saveTXT:
            np.savetxt(self.path[:-4] + "featuresRad" + str(radius) + "KNN" + str(knn) + ".txt", np.concatenate((self.pc[:,:3],self.features), axis=1))
        if savePLY:
            to_ply_feat(self.path[:-4] + "featuresRad" + str(radius) + "KNN" + str(knn) + ".ply",self.pc[:,:3],self.pc[:,3],self.features)
        return self.features


    def getLabels(self, colLbs):
        lbs = self.pc[:,colLbs]
        lbs = np.squeeze(lbs.T)

        return lbs

    def shape(self):
        return self.pc.shape

    def largerAxis(self, pc2 = None):
        minX = np.min(self.pc[:, 0])
        maxX = np.max(self.pc[:, 0])
        min = np.min(self.pc[:, 1])
        max = np.max(self.pc[:, 1])
        if maxX-minX > max - min:
            axis = 0
            min = minX
            max = maxX
        else:
            axis = 1
        if pc2 is not None:
            ax2, min2, max2 = pc2.largerAxis()
            if max2-min2>max-min:
                axis = ax2
                min =  min2
                max = max2
        return axis, min, max

def toBinaryClass(gt):
    """
    Convert a multi class array to a binary class array.
    All classes that are not 0 are set to 1
    """
    gt[gt>1] = 1
    return gt

def read_from_ply(filename, nameInPly):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata[nameInPly].count
        vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
        vertices[:, 0] = plydata[nameInPly].data["x"]
        vertices[:, 1] = plydata[nameInPly].data["y"]
        vertices[:, 2] = plydata[nameInPly].data["z"]
        vertices[:, 3] = plydata[nameInPly].data["label_ch"]
    return vertices

def lazReader(lazFile, verbose=True):
    if verbose:
        print('Reading ' + lazFile)
    file = laspy.read(lazFile)
    coords = np.vstack((file.x, file.y, file.z)).transpose()
    try:
        gt = file.change_classification
    except:
        gt = None
    return coords, gt

def to_ply(pos, label, file, obj_color = None):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                             ("pred", "i4")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    if obj_color is not None:
        colors = obj_color[np.asarray(label).astype(int)]
        ply_array["red"] = colors[:, 0]
        ply_array["green"] = colors[:, 1]
        ply_array["blue"] = colors[:, 2]
    ply_array["pred"] = np.asarray(label)
    el = PlyElement.describe(ply_array, "params")
    PlyData([el], byte_order=">").write(file)

def to_ply_feat(file, pos, label, feat, obj_color = None):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                             ("label_ch", "i4") ]
    for f in range(feat.shape[1]):
        dtype.append(("f{}".format(str(f)), "f4"))
    ply_array = np.ones(
        pos.shape[0], dtype=dtype)
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    if obj_color is not None:
        colors = obj_color[np.asarray(label).astype(int)]
        ply_array["red"] = colors[:, 0]
        ply_array["green"] = colors[:, 1]
        ply_array["blue"] = colors[:, 2]
    ply_array["label_ch"] = np.asarray(label)
    for f in range(feat.shape[1]):
        ply_array["f{}".format(str(f))] = feat[:,f]
    el = PlyElement.describe(ply_array, "params")
    PlyData([el], byte_order=">").write(file)

if __name__=="__main__":
    
    knn = 10
    radius = 5
    nb_class = 7
    
    t = time.time()
    # Train DS
    extension = "ply"
    nameInPly = "params"
    path_gal = "/share/projects/deep3dt/datasets/Simul/CDBiDate/IEEE_Dataset_V2/1-Lidar05/"
    paths = [path_gal + "Train/"]#, "/share/projects/deep3dt/datasets/Simul/CDBiDate/LyonLid0.5Single/" + "Val/"]
    ds = Dataset(paths, "pointCloud", colLbs = 3)
    ds.loadDS(numPC = 0, numPCOtherDate = 1, knn = knn, radius = radius, try2LoadFeats = True, saveTXT = True,
              extension=extension, nameInPly=nameInPly)

    # Test DS
    path_gal = "/share/projects/deep3dt/datasets/Simul/CDBiDate/IEEE_Dataset_V2/1-Lidar05/"
    paths = [path_gal + "Test/"]
    dsTest = Dataset(paths, "pointCloud", colLbs = 3)
    dsTest.loadDS(numPC = 0, numPCOtherDate = 1, knn = knn, radius = radius, try2LoadFeats = True, saveTXT = False,
                  extension=extension, nameInPly=nameInPly)

    # Val DS
    path_gal = "/share/projects/deep3dt/datasets/Simul/CDBiDate/IEEE_Dataset_V2/1-Lidar05/"
    paths = [path_gal + "Val/"]
    dsTest = Dataset(paths, "pointCloud", colLbs=3)
    dsTest.loadDS(numPC=0, numPCOtherDate=1, knn=knn, radius=radius, try2LoadFeats=True, saveTXT=False,
                  extension=extension, nameInPly=nameInPly)
