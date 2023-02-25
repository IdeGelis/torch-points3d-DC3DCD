from typing import Dict, Any
import torch
import numpy as np

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.change_detection import IGNORE_LABEL
from torch_points3d.models import model_interface


class CDTracker(BaseTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
    ):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(CDTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._ignore_label = ignore_label

        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {
            "miou": max,
            "miou_ch":max, #miou over classes of change
            "macc": max,
            "acc": max,
            "loss": min,
            "map": max,
        }  # Those map subsentences to their optimization functions

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)
        self._acc = 0
        self._macc = 0
        self._miou = 0
        self._miou_per_class = {}
        self._miou_ch = 0
        self._loss = 0

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: model_interface.TrackerInterface, conv_classes=None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_labels(self._stage):
            return

        super().track(model)
        outputs = model.get_output()
        targets = model.get_labels()
        try:
            self._loss = model.loss.item()
        except:
            self._loss = 0
        self._compute_metrics(outputs, targets, conv_classes)
        return outputs, targets



    def _compute_metrics(self, outputs, labels, conv_classes=None):
        if self._ignore_label != None:
            mask = labels != self._ignore_label
            outputs = outputs[mask]
            labels = labels[mask]

        outputs = self._convert(outputs)
        labels = self._convert(labels)
        pred = np.argmax(outputs, 1)
        if len(labels) == 0:
            return
        if conv_classes is not None:
            pred = torch.from_numpy(conv_classes[pred])
            # for key in conv_classes:
            #     pred[pred == key] = conv_classes[key]
        assert outputs.shape[0] == len(labels)
        self._confusion_matrix.count_predicted_batch(labels, pred)

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()
        class_iou, present_class = self._confusion_matrix.get_intersection_union_per_class()
        self._miou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(class_iou)
        }
        self._miou_ch = 100*np.mean(class_iou[1:])
        class_acc = self._confusion_matrix.confusion_matrix.diagonal() / self._confusion_matrix.confusion_matrix.sum(axis=1)
        self._acc_per_class = {
            k: "{:.2f}".format(100 * v)
            for k, v in enumerate(class_acc)
        }

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_loss".format(self._stage)] = self._loss
        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou
        metrics["{}_miou_ch".format(self._stage)] = self._miou_ch
        # if verbose:
        #     metrics["{}_iou_per_class".format(self._stage)] = self._miou_per_class
        #     metrics["{}_acc_per_class".format(self._stage)] = self._acc_per_class
        return metrics

    @property
    def metric_func(self):
        return self._metric_func
