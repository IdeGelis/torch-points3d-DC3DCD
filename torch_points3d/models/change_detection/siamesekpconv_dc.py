from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np
import os

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_geometric.data import Data
from torch_geometric.nn import knn

from torch_points3d.datasets.change_detection.pair import PairBatch, PairMultiScaleBatch

log = logging.getLogger(__name__)


class SiameseKPConv_DC(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        self.option = option
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        # No ponderation if weights for the corresponding number of class are available
        if len(self._weight_classes) != self._num_classes:
            self._weight_classes = None
        try:
            self._ignore_label = dataset.ignore_label
        except:
            self._ignore_label = None
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build final MLP
        if option.deepCluster_V2:
            self.last_mlp_opt = option.mlp_cls
            in_feat = 64  # self.last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(len(self.last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i + 1),
                    Sequential(
                        *[
                            Linear(in_feat, self.last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(self.last_mlp_opt.nn[i], momentum=self.last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))
            self.prototypes_cd = Sequential()
            self.prototypes_cd.add_module("prototypes_cd", Lin(in_feat, self._num_classes, bias=False))
            self.prototypes_cd.add_module("Softmax", nn.LogSoftmax(-1))
        else:
            self.last_mlp_opt = option.mlp_cls
            self.reset_final_layer(cuda=False)
        self.loss_names = ["loss_cd"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])
        self.last_feature = None
        self.visual_names = ["data_visual"]
        if self.option.downstreamTaskTraining:
            for param in self.parameters():
                if param.requires_grad:
                    param.requires_grad = False
            self.prototypes_cd.prototypes_cd.weight.requires_grad = True
            self.FC_layer[0][0].weight.requires_grad = True
            self.FC_layer[0][1].batch_norm.weight.requires_grad = True

    def print_nb_param(self, log):
        log.info(
            'total nb of trainable parameters: ' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        log.info('downconv : ' + str(sum(p.numel() for p in self.down_modules.parameters() if p.requires_grad)))
        log.info('upconv : ' + str(sum(p.numel() for p in self.up_modules.parameters() if p.requires_grad)))

    def set_class_weight(self, dataset):
        self._weight_classes = dataset.weight_classes
        # No ponderation if weights for the corresponding number of class are available
        if len(self._weight_classes) != self._num_classes:
            print('number of weights different of the number of classes')
            self._weight_classes = None

    def set_input(self, data, device):
        data = data.to(device)
        data.x = add_ones(data.pos, data.x, True)
        self.batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
        else:
            self.pre_computed = None
            self.upsample = None
        if getattr(data, "pos_target", None) is not None:
            data.x_target = add_ones(data.pos_target, data.x_target, True)
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = data.multiscale_target
                self.upsample_target = data.upsample_target
                del data.multiscale_target
                del data.upsample_target
            else:
                self.pre_computed_target = None
                self.upsample_target = None

            self.input0, self.input1 = data.to_data()
            self.batch_idx_target = data.batch_target
            self.labels = data.y.to(device)
        else:
            self.input = data
            self.labels = None

    def forward(self, compute_loss=True, epoch=0, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data0 = self.input0
        data1 = self.input1

        for i in range(len(self.down_modules) - 1):
            data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1, :], :]
            stack_down.append(diff)
        # 1024 : last layer
        data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1, :], :]
        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data1)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
        self.last_features_cd = self.FC_layer(data.x)
        if self.option.deepCluster_V2:
            self.output = self.prototypes_cd(self.last_features_cd)
        else:
            self.output = self.last_features_cd

        if self.labels is not None and compute_loss:
            self.compute_loss(epoch)

        self.data_visual = self.input1
        self.data_visual.pred = torch.max(self.output, -1)[1]

        return self.output

    def compute_loss(self, epoch=0):
        self.epoch = epoch
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        self.loss = 0
        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            print('lambda_internal_losses')
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # Final cross entrop loss
        if self._ignore_label is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels, weight=self._weight_classes,
                                       ignore_index=self._ignore_label)
        else:
            self.loss_seg = F.nll_loss(self.output, self.labels, weight=self._weight_classes)

        if torch.isnan(self.loss_seg).sum() == 1:
            print(self.loss_seg)
        self.loss += self.loss_seg
        if epoch <= self.option.deepCluster.pre_train_kmean:
            self.ep_pretrain_kmean = True
        else:
            self.ep_pretrain_kmean = False

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G
        if self.option.deepCluster_V2 and not self.ep_pretrain_kmean:
            layer = ["prototypes_cd"]
            for name, p in self.named_parameters():
                cond = any([i in name for i in layer])
                if cond:
                    p.grad = None

    def reset_final_layer(self, cuda=True):
        if self.option.deepCluster_V2:
            log.warning("Option DeepCluster V2 is set. Final FC_layer are not used, instead prototypes are used.")
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                self.last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=self.last_mlp_opt.dropout,
                bn_momentum=self.last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = self.last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(self.last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, self.last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(self.last_mlp_opt.nn[i], momentum=self.last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
            if cuda:
                self.FC_layer.cuda()

    def set_pretrained_weights(self):
        super(SiameseKPConv_DC, self).set_pretrained_weights()
        path_pretrained = getattr(self.opt, "path_pretrained_encoders", None)
        weight_name = getattr(self.opt, "weight_name", "latest")
        if path_pretrained is not None:
            if not os.path.exists(path_pretrained):
                log.warning("The path does not exist, it will not load any other encoder")
            else:
                log.info("load encoders pretrained weights from {}".format(path_pretrained))
                m = torch.load(path_pretrained, map_location="cpu")["models"][weight_name]
                self.load_state_dict_encoders(m, strict=False)

    def load_state_dict_encoders(self, weights, strict=False):
        model_state = self.state_dict()
        filtered_weights = {k: v for k, v in weights.items() if
                            'down_modules' in k and (k in model_state and v.size() == model_state[k].size())}
        log.info("Loading weights:" + ", ".join(filtered_weights.keys()))
        self.load_state_dict(filtered_weights, strict=strict)

