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

from torch_points3d.metrics.Urb3DCD_tracker import Urb3DCDTracker
from torch_points3d.datasets.change_detection import IGNORE_LABEL


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count != 0:
            return self.sum / self.count
        else:
            return np.NaN


class Urb3DCD_deepCluster_tracker(Urb3DCDTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 ignore_label: int = IGNORE_LABEL, full_pc: bool = False, full_res: bool = False):
        super(Urb3DCD_deepCluster_tracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, ignore_label,
                                                          full_pc, full_res)
        self._metric_func = {
            "miou": max,
            "miou_ch": max,  # miou over classes of change
            "macc": max,
            "acc": max,
            "loss": min,
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
        }  # Those map subsentences to their optimization functions

        self._clusteringmetric_func = {
            "RI": max,
            "ARI": max,
            "homo": max,
            "compl": max,
            "v-measure": max,
            "MI": max,
            "NMI": max,
            "NMI t/t-1": max,
            "mean-entropy": min,
            "clust-acc": max,
        }  # Those map subsentences to their optimization functions
        self._num_classes_real = dataset.num_classes_orig
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
                          "{}_NMI t/t-1".format("train"): [],
                          "{}_clust-acc".format("train"): [],
                          "{}_mean-entropy".format("train"): [],
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
                          "{}_NMI t/t-1".format("val"): [],
                          "{}_clust-acc".format("val"): [],
                          "{}_mean-entropy".format("val"): [],
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
                          "{}_NMI t/t-1".format("test"): [],
                          "{}_clust-acc".format("test"): [],
                          "{}_mean-entropy".format("test"): [],
                          # "{}_Silhouette".format("test"): [],
                          # "{}_Calinski-harabasz".format("test"): [],
                          "{}_Davies-bouldin".format("test"): []
                          }

    def track_dc(self, dataset, verbose=False, rd_eq_sampling=False):
        ri, ari, homo, compl, vmeas, mi, nmi, acc, nmi_past, mean_entrop = self.compute_metric_deepcluster(dataset,
                                                                                                           verbose,
                                                                                                           rd_eq_sampling)
        self.metric_dc["{}_RI".format(self._stage)].append(ri)
        self.metric_dc["{}_ARI".format(self._stage)].append(ari)
        self.metric_dc["{}_Homogeneity".format(self._stage)].append(homo)
        self.metric_dc["{}_Completness".format(self._stage)].append(compl)
        self.metric_dc["{}_V-measure".format(self._stage)].append(vmeas)
        self.metric_dc["{}_MI".format(self._stage)].append(mi)
        self.metric_dc["{}_NMI".format(self._stage)].append(nmi)
        self.metric_dc["{}_NMI t/t-1".format(self._stage)].append(nmi_past)
        self.metric_dc["{}_clust-acc".format(self._stage)].append(acc)
        self.metric_dc["{}_mean-entropy".format(self._stage)].append(mean_entrop)
        # self.metric_dc["{}_Calinski-harabasz".format(self._stage)].append(self.calinski_harabasz_score.avg())
        # self.metric_dc["{}_Silhouette".format(self._stage)].append(self.silhouette.avg)
        self.metric_dc["{}_Davies-bouldin".format(self._stage)].append(self.davies_bouldin_score.avg())
        if verbose:
            # print("{}_Calinski-harabasz : ".format(self._stage) + str(self.calinski_harabasz_score.avg()))
            # print("{}_Silhouette : ".format(self._stage) + str(self.silhouette.avg))
            print("{}_Davies-bouldin : ".format(self._stage) + str(self.davies_bouldin_score.avg()))

    def compute_metric_deepcluster(self, dataset, verbose=False, rd_eq_sampling=False):
        label = []
        pseudo_label = []
        for a in range(len(dataset.pseudo_labels)):
            pair = dataset.load(a)
            label.append(pair.y.cpu().numpy())
            pseudo_label.append(dataset.pseudo_labels[a].cpu().numpy())
        label = np.concatenate(label)
        pseudo_label = np.concatenate(pseudo_label)
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

        ri = skmetric.rand_score(label, pseudo_label)
        ari = adjusted_rand_score_manual(label, pseudo_label)
        # print(skmetric.adjusted_rand_score(label, pseudo_label))
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
        if verbose:
            print("{}_RI : ".format(self._stage) + str(ri))
            print("{}_ARI : ".format(self._stage) + str(ari))
            print("{}_Homogeneity : ".format(self._stage) + str(homo))
            print("{}_Completness : ".format(self._stage) + str(compl))
            print("{}_V-measure : ".format(self._stage) + str(vmeas))
            print("{}_MI : ".format(self._stage) + str(mi))
            print("{}_NMI : ".format(self._stage) + str(nmi))
            print("{}_NMI t/t-1 : ".format(self._stage) + str(nmi_past))
            print("{}_cluster acc : ".format(self._stage) + str(acc))
            print("{}_mean entropy : ".format(self._stage) + str(mean_entrop))
        return ri, ari, homo, compl, vmeas, mi, nmi, acc, nmi_past, mean_entrop

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

    def save_pseudo_label_map(self):
        if self.pseudolabel_label_map is None:
            self._calc_pseudo_label_map()
        np.savetxt(osp.join(os.getcwd(), "plabel_map.txt"), self.pseudolabel_label_map)

    def save_data(self, dataset, epoch, saving_path=None):
        if self.pseudolabel_label_map is None:
            self._calc_pseudo_label_map()

        if saving_path is None:
            saving_path = os.getcwd()
        for a in range(len(dataset.pseudo_labels)):
            pair = dataset._load_save(a)
            plab = dataset.pseudo_labels[a]
            lab = dataset.load(a).y.numpy()
            true_idx = np.where(lab == self.pseudolabel_label_map[plab])[0]
            truefalse_lab = np.zeros(plab.shape)
            truefalse_lab[true_idx] = 1
            to_ply(pair.pos_target,
                   self.pseudolabel_label_map[plab],
                   osp.join(saving_path, "pointcloud1_" + str(a) + "_ep" + str(epoch) + ".ply"),
                   color=self.class_color_plab, label2=dataset.pseudo_labels[a], label3=truefalse_lab,
                   sf=self.entropy[plab])

    # def finalise_dc(self, dataset, save_pc=False, name_test="", saving_path=None):
    #
    #     if save_pc:
    #         for a in range(dataset.size()):
    #             pair = dataset._load_save(a)
    #             if saving_path is None:
    #                 saving_path = os.path.join(os.path.dirname(dataset.filePaths), 'tp3D', 'res', name_test)
    #             if not os.path.exists(saving_path):
    #                 os.makedirs(saving_path)
    #             self._dataset.to_ply(pair.pos_target, dataset.pseudo_labels[a],
    #                                  os.path.join(saving_path, "dc_pointCloud" +
    #                                               os.path.dirname(dataset.filesPC0[a]).split('/')[
    #                                                   -1] + ".ply"),
    #                                  )


def to_ply(pos, label, file, color, label2=None, label3=None, sf=None):
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
    colors = color.get_rgb(np.asarray(label))
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("pred", "i1")]
    if label2 is not None:
        dtype.append(("pred2", "i1"))
    if label3 is not None:
        dtype.append(("compGT", "i1"))
    if sf is not None:
        dtype.append(("entropy", "f4"))
    ply_array = np.ones(
        pos.shape[0],
        dtype=dtype)
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    ply_array["pred"] = np.asarray(label)
    if label2 is not None:
        ply_array["pred2"] = np.asarray(label2)
    if label3 is not None:
        ply_array["compGT"] = np.asarray(label3)
    if sf is not None:
        ply_array["entropy"] = np.asarray(sf)
    el = PlyElement.describe(ply_array, "params")
    PlyData([el], byte_order=">").write(file)


def adjusted_rand_score_manual(labels_true, labels_pred):
    """Rand index adjusted for chance.
    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.
    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::
        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).
    ARI is a symmetric measure::
        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)
    Read more in the :ref:`User Guide <adjusted_rand_score>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate
    Returns
    -------
    ARI : float
       Similarity score between -1.0 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.
    Examples
    --------
    Perfectly matching labelings have a score of 1 even
      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized::
      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.57...
    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])
      0.57...
    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::
      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    References
    ----------
    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    .. [Steinley2004] D. Steinley, Properties of the Hubert-Arabie
      adjusted Rand index, Psychological Methods 2004
    .. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
    See Also
    --------
    adjusted_mutual_info_score : Adjusted Mutual Information.
    """
    (tn, fp), (fn, tp) = skmetric.cluster.pair_confusion_matrix(labels_true, labels_pred)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


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
