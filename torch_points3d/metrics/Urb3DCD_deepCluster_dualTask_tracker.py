import numpy as np
import sklearn.metrics as skmetric
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.stats import entropy as scipy_entropy
import os
import os.path as osp
import csv
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from plyfile import PlyData, PlyElement
import torch
from torch_geometric.nn.unpool import knn_interpolate
from torch_points3d.metrics.Urb3DCD_tracker import Urb3DCDTracker
from torch_points3d.datasets.change_detection import IGNORE_LABEL
from torch_points3d.metrics.Urb3DCD_deepCluster_tracker import MplColorHelper, Metric, adjusted_rand_score_manual, \
    to_ply
from torch_points3d.models import model_interface


class Urb3DCD_deepCluster_dualT_tracker(Urb3DCDTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 ignore_label: int = IGNORE_LABEL, full_pc: bool = False, full_res: bool = False):
        super(Urb3DCD_deepCluster_dualT_tracker, self).__init__(dataset, stage, wandb_log, use_tensorboard,
                                                                ignore_label,
                                                                full_pc, full_res)
        self._metric_func = {
            "miou": max,
            "miou_ch": max,  # miou over classes of change
            "macc": max,
            "acc": max,
            "loss": min,
            "loss_cont": min,
            "loss_nll_cd": min,
            "loss_nll_seg0": min,
            "loss_nll_seg1": min,
            "map": max,
            "RI": max,
            "ARI": max,
            "Homogeneity": max,
            "Completness": max,
            "V-meas": max,
            "MI": max,
            "NMI": max,
            "NMI t/t-1": max,
            "clust-acc": max,
            "mean-entropy": min,
            "Davies-bouldin": min,
            "NMI-binarySeg": max,
        }  # Those map subsentences to their optimization functions

        self._clusteringmetric_func = {
            "RI": max,
            "ARI": max,
            "homo": max,
            "compl": max,
            "v-measure": max,
            "MI": max,
            "NMI": max,
            "NMI-binarySeg": max,
            "NMI t/t-1": max,
            "mean-entropy": min,
            "clust-acc": max,
        }  # Those map subsentences to their optimization functions
        self._num_classes_real = dataset.num_classes_orig
        self._num_classes = dataset.nb_cluster_kmeans_cd
        self.class_color_plab = MplColorHelper('gist_rainbow', 0, self._num_classes - 1)
        self.class_color_lab = MplColorHelper('rainbow', 0, self._num_classes_real - 1)
        self.reset_dc()
        self.reset_unsup_met_epoch()

    def set_stage(self, stage):
        self._stage = stage

    def reset_unsup_met_epoch(self):
        # self.silhouette = Metric()
        # self.calinski_harabasz_score = Metric()
        self.davies_bouldin_score = Metric()
        self.pseudolabel_label_map = None

    def track(self, model: model_interface.TrackerInterface, data=None, full_pc=False, conv_classes=None, **kwargs):
        super().track(model, data=data, full_pc=full_pc, conv_classes=conv_classes)
        try:
            self._loss_cont = model.loss_cont.item()
        except:
            self._loss_cont = 0
        try:
            self._loss_nll_cd = model.loss_cd.item()
        except:
            self._loss_nll_cd = 0
        try:
            self._loss_nll_seg0 = model.loss_seg0.item()
        except:
            self._loss_nll_seg0 = 0
        try:
            self._loss_nll_seg1 = model.loss_seg1.item()
        except:
            self._loss_nll_seg1 = 0

    def track_epoch_unsup_met(self, feats, labels):
        labels = np.ravel(labels)
        # silhou = skmetric.silhouette_score(feats, np.ravel(labels))
        davies_bouldin_score = skmetric.davies_bouldin_score(feats, labels)
        # calinski_harabasz_score = skmetric.calinski_harabasz_score(feats,labels)
        # self.silhouette.update(silhou, n=feats.shape[0])
        # self.calinski_harabasz_score.update(calinski_harabasz_score, n=feats.shape[0])
        self.davies_bouldin_score.update(davies_bouldin_score, n=feats.shape[0])

    def reset_dc(self):
        self.metric_dc = {"{}_RI".format("train"): [],
                          "{}_ARI".format("train"): [],
                          "{}_Homogeneity".format("train"): [],
                          "{}_Completness".format("train"): [],
                          "{}_V-measure".format("train"): [],
                          "{}_MI".format("train"): [],
                          "{}_NMI".format("train"): [],
                          "{}_NMI-binarySeg".format("train"): [],
                          "{}_NMI t/t-1".format("train"): [],
                          "{}_clust-acc".format("train"): [],
                          "{}_mean-entropy".format("train"): [],
                          # "{}_Acc".format("train"):[],
                          # "{}_Silhouette".format("train"): [],
                          # "{}_Calinski-harabasz".format("train"): [],
                          "{}_Davies-bouldin".format("train"): [],
                          "{}_RI".format("val"): [],
                          "{}_ARI".format("val"): [],
                          "{}_Homogeneity".format("val"): [],
                          "{}_Completness".format("val"): [],
                          "{}_V-measure".format("val"): [],
                          "{}_MI".format("val"): [],
                          "{}_NMI".format("val"): [],
                          "{}_NMI-binarySeg".format("val"): [],
                          "{}_NMI t/t-1".format("val"): [],
                          "{}_clust-acc".format("val"): [],
                          "{}_mean-entropy".format("val"): [],
                          # "{}_Acc".format("val"):[],
                          # "{}_Silhouette".format("val"): [],
                          # "{}_Calinski-harabasz".format("val"): [],
                          "{}_Davies-bouldin".format("val"): [],
                          "{}_RI".format("test"): [],
                          "{}_ARI".format("test"): [],
                          "{}_Homogeneity".format("test"): [],
                          "{}_Completness".format("test"): [],
                          "{}_V-measure".format("test"): [],
                          "{}_MI".format("test"): [],
                          "{}_NMI".format("test"): [],
                          "{}_NMI-binarySeg".format("test"): [],
                          "{}_NMI t/t-1".format("test"): [],
                          "{}_clust-acc".format("test"): [],
                          "{}_mean-entropy".format("test"): [],
                          # "{}_Silhouette".format("test"): [],
                          # "{}_Calinski-harabasz".format("test"): [],
                          "{}_Davies-bouldin".format("test"): []
                          }

    def track_dc(self, dataset, verbose=False, rd_eq_sampling=False):
        ri, ari, homo, compl, vmeas, mi, nmi, acc, nmi_past, mean_entrop, nmi_binarySeg = self.compute_metric_deepcluster(dataset,
                                                                                                           verbose,
                                                                                                           rd_eq_sampling)
        self.metric_dc["{}_RI".format(self._stage)].append(ri)
        self.metric_dc["{}_ARI".format(self._stage)].append(ari)
        self.metric_dc["{}_Homogeneity".format(self._stage)].append(homo)
        self.metric_dc["{}_Completness".format(self._stage)].append(compl)
        self.metric_dc["{}_V-measure".format(self._stage)].append(vmeas)
        self.metric_dc["{}_MI".format(self._stage)].append(mi)
        self.metric_dc["{}_NMI-binarySeg".format(self._stage)].append(nmi_binarySeg)
        self.metric_dc["{}_NMI".format(self._stage)].append(nmi)
        self.metric_dc["{}_NMI t/t-1".format(self._stage)].append(nmi_past)
        self.metric_dc["{}_clust-acc".format(self._stage)].append(acc)
        self.metric_dc["{}_mean-entropy".format(self._stage)].append(mean_entrop)
        # self.metric_dc["{}_Acc".format(self._stage)].append(acc)
        # self.metric_dc["{}_Calinski-harabasz".format(self._stage)].append(self.calinski_harabasz_score.avg())
        # self.metric_dc["{}_Silhouette".format(self._stage)].append(self.silhouette.avg)
        self.metric_dc["{}_Davies-bouldin".format(self._stage)].append(self.davies_bouldin_score.avg())
        if verbose:
            # print("{}_Calinski-harabasz : ".format(self._stage) + str(self.calinski_harabasz_score.avg()))
            # print("{}_Silhouette : ".format(self._stage) + str(self.silhouette.avg))
            print("{}_Davies-bouldin : ".format(self._stage) + str(self.davies_bouldin_score.avg()))

    def compute_metric_deepcluster(self, dataset, verbose=False, rd_eq_sampling=False):
        ri = ari = homo = compl = vmeas = mi = nmi = acc = nmi_past = mean_entrop = 0
        label = []
        pseudo_label = []
        pseudo_label_change = []
        for a in range(len(dataset.pseudo_labels_cd)):
            pair = dataset.load(a)
            label.append(pair.y.cpu().numpy())
            pseudo_label.append(dataset.pseudo_labels_cd[a].cpu().numpy())
            if dataset.binaryCD_segsem:
                pseudo_label_change.append(dataset.pseudo_labels_changeSeg[a].cpu().numpy())

        label = np.concatenate(label)
        pseudo_label = np.concatenate(pseudo_label)
        if dataset.binaryCD_segsem:
            pseudo_label_change = np.concatenate(pseudo_label_change)
        if rd_eq_sampling:
            index = []
            for c in range(self._num_classes_real):
                index.append(np.random.choice(np.argwhere(label == c).squeeze(), size=int(min(dataset.nb_elt_class)),
                                              replace=False))
            index = np.concatenate(index)
            label_orig = label
            label = label[index]
            pseudo_label_orig = pseudo_label
            pseudo_label = pseudo_label[index]
            if dataset.binaryCD_segsem:
                pseudo_label_change = pseudo_label_change[index]

        ri = skmetric.rand_score(label, pseudo_label)
        ari = adjusted_rand_score_manual(label, pseudo_label)
        homo, compl, vmeas = skmetric.homogeneity_completeness_v_measure(label, pseudo_label)
        mi = skmetric.mutual_info_score(label, pseudo_label)
        nmi = skmetric.normalized_mutual_info_score(label, pseudo_label)
        acc = cluster_acc(label, pseudo_label)
        # acc = 0
        if dataset.pseudo_labels_past is not None:
            pseudo_label_past = [a.cpu().numpy() for a in dataset.pseudo_labels_past]
            pseudo_label_past = np.concatenate(pseudo_label_past)
            if rd_eq_sampling:
                nmi_past = skmetric.normalized_mutual_info_score(pseudo_label_orig, pseudo_label_past)
            else:
                nmi_past = skmetric.normalized_mutual_info_score(pseudo_label, pseudo_label_past)
        else:
            nmi_past = np.NaN
        if rd_eq_sampling:
            self._get_repartition(label_orig, pseudo_label_orig)
        else:
            self._get_repartition(label, pseudo_label)
        mean_entrop = np.nanmean(self.entropy)

        if dataset.binaryCD_segsem:
            label[label>0] = 1
            nmi_binarySeg = skmetric.normalized_mutual_info_score(label, pseudo_label_change)
        else:
            nmi_binarySeg = np.NaN
        if verbose:
            print("{}_RI : ".format(self._stage) + str(ri))
            print("{}_ARI : ".format(self._stage) + str(ari))
            print("{}_Homogeneity : ".format(self._stage) + str(homo))
            print("{}_Completness : ".format(self._stage) + str(compl))
            print("{}_V-measure : ".format(self._stage) + str(vmeas))
            print("{}_MI : ".format(self._stage) + str(mi))
            print("{}_NMI : ".format(self._stage) + str(nmi))
            print("{}_NMI binary change from semantic : ".format(self._stage) + str(nmi_binarySeg))
            print("{}_NMI t/t-1 : ".format(self._stage) + str(nmi_past))
            print("{}_cluster acc : ".format(self._stage) + str(acc))
            print("{}_mean entropy : ".format(self._stage) + str(mean_entrop))
        return ri, ari, homo, compl, vmeas, mi, nmi, acc, nmi_past, mean_entrop, nmi_binarySeg

    def _get_repartition(self, label, pseudo_label):
        self.repartition = np.zeros((self._num_classes, self._num_classes_real))
        for lab in range(self._num_classes_real):
            for plab in range(self._num_classes):
                self.repartition[plab, lab] = np.sum((label == lab) & (pseudo_label == plab))
        pourc = (self.repartition / self.repartition.sum(axis=1).reshape((self.repartition.shape[0], 1)))
        pourc[np.isnan(pourc)] = 0
        self.entropy = scipy_entropy(pourc, axis=1)  # , base=7)

    def plot_save_metric_dc(self, plot=False, saving_path=None, epoch=0):
        # Saving DC metrics into a CSV
        if saving_path is None:
            saving_path = os.getcwd()
        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)

        met_stage = get_split_dico(self.metric_dc, self._stage)

        with open(osp.join(saving_path, "metric_dc.csv"), "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(met_stage.keys())
            writer.writerows(zip(*met_stage.values()))

        if plot:
            plt.figure()
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.ylim((0, 1))
            for key, value in met_stage.items():
                if key not in ["ARI", "Calinski-harabasz", "MI", "RI",
                               "Silhouette", "Davies-bouldin", "NMI t/t-1", "clust-acc"]:
                    plt.plot(range(1, len(value) + 1), value, label=key)
            plt.legend()
            plt.savefig(osp.join(saving_path, "{}_metric_dc.png".format(self._stage)))
            plt.close()

            plt.figure()
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            # plt.ylim((0, 1))
            plt.plot(range(1, len(met_stage["ARI"]) + 1),
                     met_stage["ARI"], label="ARI")
            plt.legend()
            plt.savefig(osp.join(saving_path, "{}_metric_ARI_dc.png".format(self._stage)))
            plt.close()

            plt.figure()
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.ylim((0, 1))
            plt.plot(range(1, len(met_stage["NMI t/t-1"]) + 1),
                     met_stage["NMI t/t-1"], label="NMI t/t-1")
            plt.legend()
            plt.savefig(osp.join(saving_path, "{}_metric_NMI_tt-1_dc.png".format(self._stage)))
            plt.close()

            # plt.figure()
            # plt.grid()
            # plt.xlabel('Epoch')
            # plt.ylabel('Metric')
            # plt.plot(range(1, len(self.metric_dc["Silhouette"]) + 1),
            #          self.metric_dc["Silhouette"], label="Silhouette")
            # plt.legend()
            # plt.savefig(osp.join(saving_path, "metric_Silhouette_dc.png"))
            # plt.close()

            # plt.figure()
            # plt.grid()
            # plt.xlabel('Epoch')
            # plt.ylabel('Metric')
            # plt.plot(range(1, len(self.metric_dc["Calinski-harabasz"]) + 1),
            #          self.metric_dc["Calinski-harabasz"], label="Calinski-harabasz Score")
            # plt.legend()
            # plt.savefig(osp.join(saving_path, "metric_Calinski_harabasz_dc.png"))
            # plt.close()

            plt.figure()
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.ylim((0, 1))
            plt.plot(range(1, len(met_stage["Davies-bouldin"]) + 1),
                     met_stage["Davies-bouldin"], label="Davies-bouldin Score")
            plt.legend()
            plt.savefig(osp.join(saving_path, "{}_metric_Davies-bouldin_score_dc.png".format(self._stage)))
            plt.close()

            # pourc = (self.repartition/self.repartition.sum(axis=0))*100
            # pourc[np.isnan(pourc)] = 0
            # fig = plt.figure(1)
            # ax = fig.add_subplot(111)
            # ax.set_xlabel('Ground truth labels')
            # index = np.arange(self._num_classes_real)
            # y_offset = np.zeros(self._num_classes_real)
            # for plab in range(self._num_classes):
            #     ax.bar(index, pourc[plab, :], bottom=y_offset, label=plab,
            #             color=self.class_color_plab.get_rgb(plab))
            #     y_offset += pourc[plab, :]
            # lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=6)
            # ax.set_xticks(range(self._num_classes_real))
            # ax.set_ylabel("{}_Pseudo-labels repartition [%]".format(self._stage))
            # ax.set_ylim([0,100])
            # fig.savefig(osp.join(saving_path, "{}_repartition_dc_".format(self._stage) + str(epoch) + ".png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
            # plt.close(fig)
            if epoch % 10 == 0:
                argsort = np.argsort(self.entropy)
                pourc = (self.repartition / self.repartition.sum(axis=1).reshape((self.repartition.shape[0], 1))) * 100
                pourc[np.isnan(pourc)] = 0
                fig = plt.figure(1, figsize=(20, 8))
                ax = fig.add_subplot(111)
                ax.set_xlabel('Pseudo clusters')
                index = np.arange(self._num_classes)
                y_offset = np.zeros(self._num_classes)
                for lab in range(self._num_classes_real):
                    ax.bar(index, pourc[argsort, lab], bottom=y_offset, label=lab,
                           color=self.class_color_lab.get_rgb(lab), width=0.9)
                    y_offset += pourc[argsort, lab]
                lgd = ax.legend(title='Ground truth classes', loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=7)

                # ax.set_xticks(range(self._num_classes))
                # ax.set_xticklabels(argsort)
                ax.set_xlim([-1, self._num_classes + 1])
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_ylabel("Ground truth labels repartition [%]")
                ax.set_ylim([0, 100])
                ax2 = ax.twinx()
                ax2.scatter(range(self._num_classes), self.entropy[argsort], color='black', marker='o')
                ax2.set_ylim([0, 1.5])
                ax2.set_ylabel("Entropy")
                plt.subplots_adjust(left=0.04, right=0.97, top=0.99, bottom=0.1)
                fig.savefig(osp.join(saving_path, "{}_repartition2_dc_".format(self._stage) + str(epoch) + ".png"),
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close(fig)
                np.savetxt("{}_repartition2_dc_".format(self._stage) + str(epoch) + ".txt", self.repartition)

    def get_metrics(self, verbose=False):
        """ Returns a dictionnary of all metrics and losses being tracked
                """
        metrics = super().get_metrics(verbose)
        metrics["{}_loss_cont".format(self._stage)] = self._loss_cont
        metrics["{}_loss_nll_cd".format(self._stage)] = self._loss_nll_cd
        metrics["{}_loss_nll_seg0".format(self._stage)] = self._loss_nll_seg0
        metrics["{}_loss_nll_seg1".format(self._stage)] = self._loss_nll_seg1

        if len(self.metric_dc["{}_NMI".format(self._stage)]) > 0:
            metrics["{}_ARI".format(self._stage)] = self.metric_dc["{}_ARI".format(self._stage)][-1]
            metrics["{}_Homogeneity".format(self._stage)] = self.metric_dc["{}_Homogeneity".format(self._stage)][-1]
            metrics["{}_Completness".format(self._stage)] = self.metric_dc["{}_Completness".format(self._stage)][-1]
            metrics["{}_V-measure".format(self._stage)] = self.metric_dc["{}_V-measure".format(self._stage)][-1]
            metrics["{}_MI".format(self._stage)] = self.metric_dc["{}_MI".format(self._stage)][-1]
            metrics["{}_NMI".format(self._stage)] = self.metric_dc["{}_NMI".format(self._stage)][-1]
            metrics["{}_clust-acc".format(self._stage)] = self.metric_dc["{}_clust-acc".format(self._stage)][-1]
            metrics["{}_mean-entropy".format(self._stage)] = self.metric_dc["{}_mean-entropy".format(self._stage)][-1]
            metrics["{}_Davies-bouldin".format(self._stage)] = self.metric_dc["{}_Davies-bouldin".format(self._stage)][
                -1]
        return metrics

    def get_metrics_supervised(self, verbose=False):
        metrics = super().get_metrics(verbose)
        return metrics

    def _calc_pseudo_label_map(self):
        self.pseudolabel_label_map = (
                    self.repartition / self.repartition.sum(axis=1).reshape((self.repartition.shape[0], 1)))
        self.pseudolabel_label_map = self.pseudolabel_label_map.argmax(axis=1)

    def save_pseudo_label_map(self, epoch=-1):
        if self.pseudolabel_label_map is None:
            self._calc_pseudo_label_map()
        np.savetxt(osp.join(os.getcwd(), "plabel_map.txt"), self.pseudolabel_label_map)

    def save_data(self, dataset, epoch, saving_path=None):
        if self.pseudolabel_label_map is None:
            self._calc_pseudo_label_map()

        if saving_path is None:
            saving_path = os.getcwd()
        for a in range(len(dataset.pseudo_labels_cd)):
            pair = dataset._load_save(a)
            to_ply(pair.pos_target,
                   pseudolabel_label_map[dataset.pseudo_labels_cd[a]],
                   osp.join(saving_path, "pointcloud1_" + str(a) + "_ep" + str(epoch) + ".ply"),
                   color=self.class_color_plab, label2=dataset.pseudo_labels_cd[a])

    def save_data_binSegSem(self, dataset, epoch, saving_path=None):
        if saving_path is None:
            saving_path = os.getcwd()
        for a in range(len(dataset.pseudo_labels_changeSeg)):
            pair = dataset._load_save(a)
            pred = dataset.pseudo_labels_changeSeg[a]
            _, area_orig_pos, gt = self._ds.clouds_loader(a)
            # still on GPU no need for num_workers
            pred = knn_interpolate(torch.unsqueeze(pred, 1), pair.pos_target,
                                   area_orig_pos, k=1).numpy()
            pred = np.squeeze(pred)
            pred = pred.astype(int)
            pos = area_orig_pos
            to_ply(pos,
                   pred,
                   osp.join(saving_path, "pointcloud1_" + str(a) + "_ep" + str(epoch) + ".ply"),
                   color=self.class_color_plab)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def get_split_dico(dico, key1):
    dico_k1 = {}
    for (key, val) in dico.items():
        if key1 in key:
            key = key.split("_")[1]
            dico_k1[key] = val
    return dico_k1
