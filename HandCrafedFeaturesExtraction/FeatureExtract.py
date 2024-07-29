#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:23:00 2020

@author: degelis
"""
import pdal
import numpy as np
import math as m
import scipy
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from progress.bar import IncrementalBar
import warnings
import time
import multiprocessing as mp

def npArray2PDALNdarray(pc):
    x_vals = np.squeeze(pc[:,0].T)
    y_vals = np.squeeze(pc[:,1].T)
    z_vals = np.squeeze(pc[:,2].T)
    pcPdal = np.array(
        [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)],
        dtype=[('X', np.float), ('Y', np.float), ('Z', np.float)]
    )
    return pcPdal

def PDALNdarray2NpArray(pcPdal):
    nb_pt = pcPdal.shape[0]
    pc = np.array([list(pcPdal[i]) for i in range(nb_pt)])
    return pc

def getFeaturesfromPDAL(pc, saveTXT = False, saving_path = ""):
    json =  """
                {
                  "pipeline": [
                    {
                       "type":"filters.normal",
                       "knn":10,
                       "refine":true
                   }
                  ]
            }"""
    pcPdal = npArray2PDALNdarray(pc)
    p = pdal.Pipeline(json = json, arrays = [pcPdal])
    p.validate()
    result = p.execute()
    array = p.arrays[0]
    print('PDAL pipeline (normals) done')
    pc_feat = PDALNdarray2NpArray(array)
    if saveTXT:
        np.savetxt(saving_path, pc_feat)
    return pc_feat


def neighborhoodFeatures(pc, knn):
    feats = np.zeros((pc.shape[0],5))
    kdt = cKDTree(pc)
    distance, point_id = kdt.query(pc, knn)
    for pt in range(pc.shape[0]):
        neighborhood = getCoordNeighborhood(pc, point_id[pt,:])
        eigenvalues = pcaNeighbor(neighborhood[1:,:])
        Lt, Pt, Ot = compGeomNeighbor(eigenvalues)
        zRk, zRg = ZRanking(neighborhood)
        feats[pt,0] = Lt
        feats[pt,1] = Pt
        feats[pt,2] = Ot
        feats[pt,3] = zRk
        feats[pt,4] = zRg
    return feats

# def getCoordNeighborhood(pc, point_id):
#     # neighborhood = np.array([pc[i,:] for i in point_id])
#     neighborhood = pc[point_id,:]
#     return neighborhood



def pcaNeighbor(neighborhood):
    pca.fit(neighborhood)
    return pca.explained_variance_


def compGeomNeighbor(eigenvalues):
    # If E1>=E2>=E3>=0
    #Lt = 1 (E2/E1)
    Lt = 1 - eigenvalues[1]/eigenvalues[0]
    #Pt = (E2 -E3)/E1
    Pt = (eigenvalues[1] - eigenvalues[2])/eigenvalues[0]
    # Ot = racine 3 (E1E2E3)
    Ot = np.cbrt(eigenvalues[0]*eigenvalues[1]*eigenvalues[2])
    return Lt, Pt, Ot


def ZRanking(neighborhood):
    ind = np.argsort(neighborhood[:,2])
    zRk = np.where(np.equal(ind,0))[0][0] + 1 # not to begin at 0
    zRg = neighborhood[ind[-1],2] - neighborhood[ind[0],2]
    return zRk, zRg



def pdalDTM(pc):
    json = """[
            {
                "type":"filters.smrf",
                "window":33,
                "slope":0.15,
                "threshold":0.5,
                "cell":1.0
            },
            {
                "type":"filters.range",
                "limits":"Classification[2:2]"
            }
            ]"""

    pcPdal = npArray2PDALNdarray(pc)
    p = pdal.Pipeline(json = json , arrays = [pcPdal])
    p.validate()
    result = p.execute()
    grdPts = p.arrays[0]
    grdPts = PDALNdarray2NpArray(grdPts)
    print('PDAL pipeline (ground filter) done')
    BB = getBB(pc)
    DTM= grdPts2DTM(grdPts, BB, grid_step=1)
    return DTM



def grdPts2DTM(grdPts, BB, grid_step):
    (Xmin, Ymin, Xmax,Ymax) = BB
    Xsize = int(m.ceil((Xmax - Xmin)/grid_step))
    Ysize = int(m.ceil((Ymax - Ymin)/grid_step))
    heights = np.zeros((Xsize,Ysize)) + 10000
    for pts in range(grdPts.shape[0]):
        pt = grdPts[pts,:]
        cellX = int(abs((pt[0]-Xmin)/grid_step))
        cellY = int(abs((pt[1]-Ymin)/grid_step))
        if cellX<Xsize and cellY<Ysize:
            if heights[cellX, cellY]> pt[2]:
                heights[cellX, cellY] = pt[2]
    heights = np.flip(heights, axis = 1)
    xx, yy = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]))
    heights[heights == 10000] = np.nan
    values = np.ravel(heights)
    values = values[~ np.isnan(values)]
    points = np.argwhere(~ np.isnan(heights))
    DTM = scipy.interpolate.griddata(points, values, (xx.T, yy.T), method='linear',fill_value = np.min(values) ).T
    return DTM

def getBB(PC):
    """
    Return the bounding box of the points cloud.
    IN:
        PC: nd array: Points cloud X Y Z GT...
    OUT:
        bb: tuple<float>: boudning box (Xmin, Ymin, Xmax,Ymax)
    """
    Xmin = np.min(PC[:,0])
    Ymin = np.min(PC[:,1])
    Xmax = np.max(PC[:,0])
    Ymax = np.max(PC[:,1])
    return (Xmin, Ymin, Xmax,Ymax)

def normalizeHeight(pc, DTM):
    BB = getBB(pc)
    xcoord = np.arange(start = BB[0], stop = BB[2], step = 1)
    ycoord = np.arange(start = BB[3], stop = BB[1], step = -1)

    normHeight = np.zeros((pc.shape[0],))
    for pt in range(pc.shape[0]):
        xpos = findPos(pc[pt,0], xcoord)
        ypos = findPos(pc[pt,1], ycoord)
        normHeight[pt] = pc[pt,2] - DTM[ypos,xpos]
    return normHeight


def findPos(elt,li):
    pos = min(range(len(li)), key = lambda i: abs(li[i]-elt))
    return pos

def stability(pc, pcOtherDate, radius=1, fill_in = 0):
    stab = np.zeros((pc.shape[0],))
    kdtOD = cKDTree(pcOtherDate)
    res3D = kdtOD.query_ball_point(pc,radius)
    for pt in range(pc.shape[0]):
        n3D = len(res3D[pt])
        n2D = getn2D(pc[pt,:], pcOtherDate, radius)
        stab[pt] = (n3D/n2D)*100
    stab[np.isnan(stab)] = fill_in
    return stab


def getn2D(pt, pc, radius):
    """
    Inspired by
    https://stackoverflow.com/questions/47932955/how-to-check-if-a-3d-point-is-inside-a-cylinder
    """
    # A long z axis
    res = np.linalg.norm(np.cross(pc - pt, np.array([0, 0, 1])), axis=1)
    # cpt = np.sum(res<=radius)
    cpt = np.sum(np.greater(radius,res))
    return cpt

def pointWiseFeats(pc, DTM, pcOtherDate, knn = 10, radius=1, fill_in = 0):
    time_cpt = time.time()
    global pca
    pca = PCA(n_components=3)
    grid_step = 1
    # Neighborhood features
    neighborFeats = np.zeros((pc.shape[0],5))
    kdt = cKDTree(pc)
    distance, point_id = kdt.query(pc, knn)
    print("Neighborhood features initialisation done")

    # normalizeHeight
    BB = getBB(pc)
    (Xmin, Ymin, Xmax,Ymax) = BB
    # xcoord = np.arange(start = BB[0], stop = BB[2], step = 1)
    # ycoord = np.arange(start = BB[3], stop = BB[1], step = -1)
    normHeight = np.zeros((pc.shape[0],))
    print("Height feature initialisation done")

    #stability
    stab = np.zeros((pc.shape[0],))
    kdtOD = cKDTree(pcOtherDate)
    kdtOD2D = cKDTree(pcOtherDate[:,:2])
    res3D = kdtOD.query_ball_point(pc,radius)
    res2D = kdtOD2D.query_ball_point(pc[:,:2],radius)
    print("Stability feature initialisation done")
    print("--- %s sec ---\n" % str((time.time() - time_cpt)))
    time_cpt = time.time()
    # For loop on points
    bar = IncrementalBar('Point wise features', max = pc.shape[0])
    for pt in range(pc.shape[0]):
        # Neighborhood features
        # neighborhood = getCoordNeighborhood(pc, point_id[pt,:])
        neighborhood = np.array([pc[i,:] for i in point_id])
        eigenvalues = pcaNeighbor(neighborhood[1:,:])
        Lt, Pt, Ot = compGeomNeighbor(eigenvalues)
        zRk, zRg = ZRanking(neighborhood)
        neighborFeats[pt,0] = Lt
        neighborFeats[pt,1] = Pt
        neighborFeats[pt,2] = Ot
        neighborFeats[pt,3] = zRk
        neighborFeats[pt,4] = zRg

        # Norm height
        xpos = min(int(round(abs((pc[pt,0]-Xmin)/grid_step))),DTM.shape[1]-1)
        ypos = min(int(round(abs((pc[pt,1]-Ymax)/grid_step))),DTM.shape[0]-1)

        normHeight[pt] = pc[pt,2] - DTM[ypos,xpos]
        # Stability
        n3D = len(res3D[pt])
        n2D = max(len(res2D[pt]),0.1) # if n2D = 0 then n3D = 0 then stability = 0
        stab[pt] = (n3D/n2D)*100
    bar.finish()
    print("--- %s sec ---\n" % str((time.time() - time_cpt)))
    return neighborFeats, normHeight, stab


def mppointWiseFeats(pc, DTM, pcOtherDate, knn = 10, radius=1, fill_in = 0, grid_step=1):
    time_cpt = time.time()
    # Neighborhood features
    neighborFeats = np.zeros((pc.shape[0],5))
    kdt = cKDTree(pc)
    distance, point_id = kdt.query(pc, knn)
    print("Neighborhood features initialisation done")

    # normalizeHeight
    (Xmin, Ymin, Xmax,Ymax) = getBB(pc)
    normHeight = np.zeros((pc.shape[0],))
    print("Height feature initialisation done")

    #stability
    stab = np.zeros((pc.shape[0],))
    kdtOD = cKDTree(pcOtherDate)
    kdtOD2D = cKDTree(pcOtherDate[:,:2])
    res3D = kdtOD.query_ball_point(pc,radius)
    res2D = kdtOD2D.query_ball_point(pc[:,:2],radius)
    print("Stability feature initialisation done")
    print("--- %s sec ---\n" % str((time.time() - time_cpt)))
    time_cpt = time.time()
    # For loop on points
    print(mp.cpu_count())
    pool = mp.Pool(mp.cpu_count(), initializer=setInit, initargs=(pc, point_id, DTM, Xmin, Ymax, grid_step, res3D, res2D, radius, pcOtherDate, ))
    results = pool.map(mpPointFeat,[pt for pt in range(pc.shape[0])])
    pool.close()
    print('Calculation done')
    print("--- %s sec ---\n" % str((time.time() - time_cpt)))
    time_cpt = time.time()
    for pt in range(len(results)):
        pt, Lt, Pt, Ot, zRk, zRg, xpos, ypos, nh, sta = results[pt]
        # Neighborhood features
        neighborFeats[pt,0] = Lt
        neighborFeats[pt,1] = Pt
        neighborFeats[pt,2] = Ot
        neighborFeats[pt,3] = zRk
        neighborFeats[pt,4] = zRg

        # Norm height
        normHeight[pt] = nh
        # Stability
        stab[pt] = sta
    print("--- %s sec ---\n" % str((time.time() - time_cpt)))
    return neighborFeats, normHeight, stab

def setInit(pc_in, point_id_in, DTM_in, Xmin_in, Ymax_in, grid_step_in, res3D_in, res2D_in,  radius_in, pcOtherDate_in):#, bar_in):
    global pc
    pc = pc_in
    global point_id
    point_id = point_id_in
    global DTM
    DTM = DTM_in
    global Xmin
    Xmin = Xmin_in
    global Ymax
    Ymax = Ymax_in
    global grid_step
    grid_step = grid_step_in
    global res3D
    res3D = res3D_in
    global res2D
    res2D = res2D_in
    global radius
    radius = radius_in
    global pcOtherDate
    pcOtherDate = pcOtherDate_in
    global pca
    pca = PCA(n_components=3)


def mpPointFeat(pt):
    # Neighborhood features
    neighborhood = np.array([pc[i,:] for i in point_id[pt,:]])
    eigenvalues = pcaNeighbor(neighborhood[1:,:])
    Lt, Pt, Ot = compGeomNeighbor(eigenvalues)
    zRk, zRg = ZRanking(neighborhood)

    # Norm height
    xpos = min(int(round(abs((pc[pt,0]-Xmin)/grid_step))),DTM.shape[1]-1)
    ypos = min(int(round(abs((pc[pt,1]-Ymax)/grid_step))),DTM.shape[0]-1)
    # print(xposo, xpos)
    nh = pc[pt,2] - DTM[ypos,xpos]
    # Stability
    n3D = len(res3D[pt])
    n2D = max(len(res2D[pt]),0.1)
    sta = (n3D/n2D)*100
    feats = [pt, Lt, Pt, Ot, zRk, zRg, xpos, ypos, nh, sta]
    return feats
