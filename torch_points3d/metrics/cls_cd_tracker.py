from typing import Dict, Any
import logging
import torch
from torch_geometric.nn.unpool import knn_interpolate
import numpy as np
import os
import os.path as osp
import _pickle as pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.change_detection_tracker import CDTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.change_detection import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class Cls_cd_tracker(CDTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 ignore_label: int = IGNORE_LABEL):
        super(Cls_cd_tracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, ignore_label)
        self.gt_tot = None
        self.pred_tot = None
        self.areas = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._stage == 'test':
            self._ds = self._dataset.test_data
        elif self._stage == 'val':
            self._ds = self._dataset.val_data
        else:
            self._ds = self._dataset.train_data
        self.gt_tot = None
        self.pred_tot = None
        self.outputs = None
        self.areas = []

    def track(self, model: model_interface.TrackerInterface, data=None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        outputs, targets = super().track(model)

        # Train mode or low res, nothing special to do
        if self._stage == "train":
            return

        if self.pred_tot is None:
            self.outputs = outputs
            self.pred_tot = outputs.argmax(axis=1)
            self.gt_tot = targets
            self.areas = data.area
        else:
            self.outputs = torch.cat((self.outputs, outputs))
            self.pred_tot = torch.cat((self.pred_tot, outputs.argmax(axis=1)))
            self.gt_tot = torch.cat((self.gt_tot, targets))
            # self.areas = torch.cat((self.areas,  data.area))
            self.areas += data.area

    def finalise_TTA(self, predictions, labels, stage_name="train"):
        self.reset(stage_name)
        pred_agg = torch.mean(predictions, axis=0)
        super()._compute_metrics(pred_agg, labels)

    def finalise(self, save_pc=False, name_test="", saving_path=None, conv_classes=None, **kwargs):
        per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]
        self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}
        saving_path = os.path.join(os.getcwd(), name_test)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        name_classes = [name for name, i in self._ds.class_labels.items()]
        self.save_metrics(name_test=name_test, saving_path=saving_path)
        self.save_pred(saving_path=saving_path)
        try:
            self.plot_confusion_matrix(self.gt_tot.cpu().numpy(), self.pred_tot.cpu().numpy(), normalize=True,
                                       saving_path=saving_path + "cm.png",
                                       name_classes=name_classes)
        except:
            pass
        try:
            self.plot_confusion_matrix(self.gt_tot.cpu().numpy(), self.pred_tot.cpu().numpy(), normalize=True,
                                       saving_path=saving_path + "cm2.png")
        except:
            pass

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        return metrics

    def save_pred(self, saving_path=None):
        if saving_path is None:
            saving_path = os.path.join(os.getcwd(), 'res', self._stage)
        with open(osp.join(saving_path, "pred.txt"), "w") as fi:
            fi.write("Area GT Pred \n")
            for sc in range(len(self.areas)):
                fi.write(str(self.areas[sc]) + " " + str(self.gt_tot[sc].cpu().numpy()) + " " + str(self.pred_tot[sc].cpu().numpy()) + "\n")

    def save_metrics(self, saving_path=None, name_test=""):
        metrics = self.get_metrics()
        if saving_path is None:
            saving_path = os.path.join(os.path.dirname(self._ds.filesPC0[i]), 'tp3D', 'res', name_test)
        with open(osp.join(saving_path, "res.txt"), "w") as fi:
            for met, val in metrics.items():
                fi.write(met + " : " + str(val) + "\n")
            fi.write("\n")

    def plot_confusion_matrix(self, y_true, y_pred,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues,
                              saving_path="",
                              name_classes=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = unique_labels(y_true, y_pred)
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        if name_classes == None:
            name_classes = classes
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=name_classes, yticklabels=name_classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.xlim(-0.5, len(np.unique(y_pred)) - 0.5)
        plt.ylim(len(np.unique(y_pred)) - 0.5, -0.5)
        if saving_path != "":
            plt.savefig(saving_path)
        return ax


def merge_avg_mappings(dicts):
    """ Merges an arbitrary number of dictionaries based on the
    average value in a given mapping.

    Parameters
    ----------
    dicts : Dict[Any, Comparable]

    Returns
    -------
    Dict[Any, Comparable]
        The merged dictionary
    """
    merged = {}
    cpt = {}
    for d in dicts:  # `dicts` is a list storing the input dictionaries
        for key in d:
            if key not in merged:
                if type(d[key]) == dict:
                    merged[key] = {}
                    for key2 in d[key]:
                        merged[key][key2] = float(d[key][key2])
                else:
                    merged[key] = d[key]
                cpt[key] = 1
            else:
                if type(d[key]) == dict:
                    for key2 in d[key]:
                        merged[key][key2] += float(d[key][key2])
                else:
                    merged[key] += d[key]
                cpt[key] += 1
    for key in merged:
        if type(merged[key]) == dict:
            for key2 in merged[key]:
                merged[key][key2] /= cpt[key]
        else:
            merged[key] /= cpt[key]
    return merged
