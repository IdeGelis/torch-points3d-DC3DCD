from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np
import copy
import sklearn.metrics as skmetric
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

import torch_points3d.clustering_dc as clustering_dc

log = logging.getLogger(__name__)


class BaseFactoryPSI:
    def __init__(self, module_name_down, module_name_up_cd, module_name_up_seg, modules_lib):
        self.module_name_down = module_name_down
        self.module_name_up_cd = module_name_up_cd
        self.module_name_up_seg = module_name_up_seg
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP_CONV_CD":
            return getattr(self.modules_lib, self.module_name_up_cd, None)
        if flow.upper() == "UP_CONV_SEG":
            return getattr(self.modules_lib, self.module_name_up_seg, None)
        else:
            return getattr(self.modules_lib, self.module_name_down, None)


####################DEEPCLUSTER SIAMESE KP CONV Dual Task############################
class SiameseKPConv_DC_dual(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        self.option = option
        # Extract parameters from the dataset
        self._num_classes_cd = dataset.nb_cluster_kmeans_cd
        self._num_classes_seg = dataset.nb_cluster_kmeans_seg
        self._weight_classes_cd = None
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
        opt = copy.deepcopy(option)
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}

        self._init_from_compact_format(opt, model_type, dataset, modules)

        # Build final MLPs
        # Last MLP + prototype CD
        self.last_mlp_opt_cd = option.mlp_cls_cd
        in_feat = self.last_mlp_opt_cd.nn[0] + self._num_categories
        self.FC_layer_cd = Sequential()
        for i in range(len(self.last_mlp_opt_cd.nn)):
            self.FC_layer_cd.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(in_feat, self.last_mlp_opt_cd.nn[i], bias=False),
                        FastBatchNorm1d(self.last_mlp_opt_cd.nn[i], momentum=self.last_mlp_opt_cd.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                ),
            )
            in_feat = self.last_mlp_opt_cd.nn[i]

        if self.last_mlp_opt_cd.dropout:
            self.FC_layer_cd.add_module("Dropout", Dropout(p=self.last_mlp_opt_cd.dropout))
        self.prototypes_cd = Sequential()
        self.prototypes_cd.add_module("prototypes_cd", Lin(in_feat, self._num_classes_cd, bias=False))
        self.prototypes_cd.add_module("Softmax", nn.LogSoftmax(-1))

        # Last MLP + prototype Seg
        self.last_mlp_opt_seg = option.mlp_cls_seg
        in_feat = self.last_mlp_opt_seg.nn[0] + self._num_categories
        self.FC_layer_seg = Sequential()
        for i in range(len(self.last_mlp_opt_seg.nn)):
            self.FC_layer_seg.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(in_feat, self.last_mlp_opt_seg.nn[i], bias=False),
                        FastBatchNorm1d(self.last_mlp_opt_seg.nn[i], momentum=self.last_mlp_opt_seg.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                ),
            )
            in_feat = self.last_mlp_opt_seg.nn[i]

        if self.last_mlp_opt_seg.dropout:
            self.FC_layer_seg.add_module("Dropout", Dropout(p=self.last_mlp_opt_seg.dropout))
        self.prototypes_seg = Sequential()
        self.prototypes_seg.add_module("prototypes_seg", Lin(in_feat, self._num_classes_seg, bias=False))
        self.prototypes_seg.add_module("Softmax", nn.LogSoftmax(-1))

        self.loss_names = ["loss_cd"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])
        self.last_feature = None
        self.visual_names = ["data_visual"]
        self.loss_opt = self.get_from_opt(option, ["loss_type"], default_value="triple")

    def print_nb_param(self, log):
        log.info(
            'total nb of trainable parameters: ' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        log.info('downconv : ' + str(sum(p.numel() for p in self.down_modules.parameters() if p.requires_grad)))
        log.info('upconv_cd : ' + str(sum(p.numel() for p in self.up_modules_cd.parameters() if p.requires_grad)))
        log.info('upconv_seg : ' + str(sum(p.numel() for p in self.up_modules_seg.parameters() if p.requires_grad)))

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.down_modules = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules_seg = nn.ModuleList()
        self.up_modules_cd = nn.ModuleList()

        self.save_sampling_id = opt.down_conv.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        up_conv_cd_cls_name = opt.up_conv_cd.module_name if opt.get('up_conv_cd') is not None else None
        up_conv_seg_cls_name = opt.up_conv_seg.module_name if opt.get('up_conv_seg') is not None else None
        self._factory_module = factory_module_cls(
            down_conv_cls_name, up_conv_cd_cls_name, up_conv_seg_cls_name, modules_lib
        )  # Create the factory object

        # Loal module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules.append(down_module)

        # Up modules
        if up_conv_cd_cls_name:
            for i in range(len(opt.up_conv_cd.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv_cd, i, "UP_CONV_CD")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules_cd.append(up_module)

        if up_conv_seg_cls_name:
            for i in range(len(opt.up_conv_seg.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv_seg, i, "UP_CONV_SEG")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules_seg.append(up_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

    def _get_factory(self, model_name, modules_lib) -> BaseFactoryPSI:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactoryPSI
        return factory_module_cls

    def set_class_weight(self, dataset):
        self._weight_classes_cd = dataset.weight_classes_cd
        if self._weight_classes_cd is not None:
            # No ponderation if weights for the corresponding number of class are available
            if len(self._weight_classes_cd) != self._num_classes_cd:
                print('number of weights different of the number of classes')
                self._weight_classes_cd = None

        self._weight_classes_seg = dataset.weight_classes_seg
        if self._weight_classes_seg is not None:
            # No ponderation if weights for the corresponding number of class are available
            if len(self._weight_classes_seg) != self._num_classes_seg:
                print('number of weights different of the number of classes')
                self._weight_classes_seg = None

    def set_input(self, data, device):
        self.data = data
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
            self.labels_seg0 = getattr(data, "y0", None)
            self.labels_seg1 = getattr(data, "y1", None)
            self.gt_change = getattr(data, "gt_change", None)
            if self.labels_seg0 is not None:
                self.labels_seg0.to(device)
            if self.labels_seg1 is not None:
                self.labels_seg1.to(device)
            if self.gt_change is not None:
                self.gt_change.to(device)
        else:
            self.input = data
            self.labels = None
            self.labels_seg0 = None
            self.labels_seg1 = None

    def forward(self, compute_loss=True, epoch=0, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down_input0 = []
        stack_down_input1 = []
        stack_down_CD = []

        data0 = self.input0
        data1 = self.input1

        for i in range(len(self.down_modules) - 1):
            data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            if i == 0:
                self.nn_list_init = nn_list
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1, :], :]
            stack_down_CD.append(diff)
            stack_down_input0.append(data0)
            stack_down_input1.append(data1)

        # 1024 : last layer
        data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1, :], :]
        innermost = False
        for i in range(len(self.up_modules_cd)):
            if i == 0 and innermost:
                data = self.up_modules_cd[i]((data, stack_down_CD.pop()))
            else:
                data = self.up_modules_cd[i]((data, stack_down_CD.pop()), precomputed=self.upsample_target)

        for i in range(len(self.up_modules_seg)):
            if i == 0 and innermost:
                data0 = self.up_modules_seg[i]((data0, stack_down_input0.pop()))
                data1 = self.up_modules_seg[i]((data1, stack_down_input1.pop()))
            else:
                data0 = self.up_modules_seg[i]((data0, stack_down_input0.pop()), precomputed=self.upsample)
                data1 = self.up_modules_seg[i]((data1, stack_down_input1.pop()), precomputed=self.upsample_target)
        # print(torch.all(data1.x == data0.x))
        self.last_features_cd = self.FC_layer_cd(data.x)
        self.output = self.prototypes_cd(self.last_features_cd)
        self.last_features_seg_d0 = self.FC_layer_seg(data0.x)
        self.output_seg_d0 = self.prototypes_seg(self.last_features_seg_d0)
        self.last_features_seg_d1 = self.FC_layer_seg(data1.x)
        self.output_seg_d1 = self.prototypes_seg(self.last_features_seg_d1)

        if self.labels is not None and compute_loss:
            self.compute_loss(epoch)

        self.data_visual = self.input1
        self.data_visual.pred = torch.max(self.output, -1)[1]

        return self.output

    def set_trainable_encoders(self):
        for p in self.down_modules.parameters():
            if p.requires_grad == False:
                p.requires_grad = True

    def compute_loss(self, epoch=0):
        self.cur_epoch = epoch
        if self._weight_classes_cd is not None:
            self._weight_classes_cd = self._weight_classes_cd.to(self.output.device)
        if self._weight_classes_seg is not None:
            self._weight_classes_seg = self._weight_classes_seg.to(self.output_seg_d1.device)
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
        self.loss_seg0 = F.nll_loss(self.output_seg_d0, self.labels_seg0, weight=self._weight_classes_seg)
        self.loss_seg1 = F.nll_loss(self.output_seg_d1, self.labels_seg1, weight=self._weight_classes_seg)

        self.ep_cd = False
        if epoch < self.option.deepCluster.n2_epoch:
            self.ep_cd = True
        if epoch < self.option.deepCluster.n1_epoch:
            self.loss += 0.5 * self.loss_seg0 + 0.5 * self.loss_seg1
            self.ep_seg = True
        else:
            self.loss += 0.25 * self.loss_seg0 + 0.25 * self.loss_seg1
            self.loss_cd = F.nll_loss(self.output, self.labels, weight=self._weight_classes_cd)
            self.loss += 0.25 * self.loss_cd
            # Contrastive loss
            self.loss_cont = self.get_max_margin_contrastive_loss()
            self.loss += 0.25 * self.loss_cont
            self.ep_seg = False

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G
        if self.ep_cd:
            layer = ["prototypes_cd", "prototypes_seg", "up_modules_seg", "down_modules", "FC_layer_seg"]
        elif self.ep_seg:
            layer = ["prototypes_cd", "prototypes_seg", "up_modules_cd", "FC_layer_cd"]
        else:
            layer = ["prototypes_cd", "prototypes_seg", "up_modules_seg", "FC_layer_seg"]
        for name, p in self.named_parameters():
            cond = any([i in name for i in layer])
            if cond:
                p.grad = None

    def get_max_margin_contrastive_loss_seg(self):
        zero = torch.zeros(self.last_features_seg_d1.shape[0]).to(self.device)
        y = torch.zeros(self.last_features_seg_d1.shape[0]).to(self.device)
        # if self.option.deepCluster.contrastive_on_label:
        #     y[self.labels_seg0[self.nn_list_init[1, :]] == self.labels_seg1] = 1
        if self.option.deepCluster.y_from_threshold:
            y = self.gt_change
        else:
            y[torch.argmax(self.output_seg_d0, dim=1)[self.nn_list_init[1, :]] == torch.argmax(self.output_seg_d1,
                                                                                               dim=1)] = 1
        # y[self.gt_change == 0] = 1
        beta = 1
        dFseg = torch.norm(self.last_features_seg_d1 - self.last_features_seg_d0[self.nn_list_init[1, :], :], p=2,
                           dim=1)

        loss_cont = beta * (
                    0.5 * y * dFseg ** 2 + (1 - y) * torch.max(self.option.deepCluster.alpha - dFseg, zero) ** 2)
        loss_cont = torch.mean(loss_cont)

        return loss_cont

    def get_max_margin_contrastive_loss(self):
        zero = torch.zeros(self.last_features_seg_d1.shape[0]).to(self.device)
        y = torch.zeros(self.last_features_seg_d1.shape[0]).to(self.device)
        # if self.option.deepCluster.contrastive_on_label:
        #     y[self.labels_seg0[self.nn_list_init[1, :]] == self.labels_seg1] = 1
        if self.option.deepCluster.y_from_threshold:
            y = self.gt_change
        else:
            y[torch.argmax(self.output_seg_d0, dim=1)[self.nn_list_init[1, :]] == torch.argmax(self.output_seg_d1,
                                                                                               dim=1)] = 1
        # y[self.gt_change == 0] = 1
        beta = 1
        gamma = 1
        dFseg = torch.norm(self.last_features_seg_d1 - self.last_features_seg_d0[self.nn_list_init[1, :], :], p=2,
                           dim=1)
        dFcd = torch.norm(self.last_features_cd, p=2, dim=1)

        loss_cont = beta * (
                    0.5 * y * dFseg ** 2 + (1 - y) * torch.max(self.option.deepCluster.alpha - dFseg, zero) ** 2)
        loss_cont += gamma * 0.5 * y * dFcd ** 2
        loss_cont = torch.mean(loss_cont)

        if self.option.deepCluster.lossContTotal:
            if self.option.deepCluster.contrastive_on_label:
                cij = torch.cat(
                    (torch.unsqueeze(self.labels_seg0[self.nn_list_init[1, :]], 1),
                     torch.unsqueeze(self.labels_seg1, 1)), 1)
            else:
                cij = torch.cat(
                    (torch.unsqueeze(torch.argmax(self.output_seg_d0, dim=1)[self.nn_list_init[1, :]], 1),
                     torch.unsqueeze(torch.argmax(self.output_seg_d1, dim=1), 1)),
                    1)

            val, inverse_indices, count = torch.unique(cij, return_inverse=True, return_counts=True, dim=0)
            idx_repeated = torch.where(count[inverse_indices] > 1)[0]
            idx_cor = []
            for i in idx_repeated:
                idx_similar_value = torch.where(inverse_indices == inverse_indices[i])[0]
                idx_similar_value = idx_similar_value[idx_similar_value != i]
                idx_cor.append(idx_similar_value[torch.randint(idx_similar_value.shape[0], (1,))][0])

            dFchange = torch.norm(self.last_features_cd[idx_repeated, :] - self.last_features_cd[idx_cor, :], p=2,
                                  dim=1)
            loss_cont += torch.mean(0.5 * dFchange ** 2)
        return loss_cont

    def set_pretrained_weights(self):
        super(SiameseKPConv_DC_dual, self).set_pretrained_weights()
        # Concerning encoders only now
        path_pretrained = getattr(self.opt, "path_pretrained_encoders", None)
        weight_name = getattr(self.opt, "weight_name", "latest")

        if path_pretrained is not None:
            if not os.path.exists(path_pretrained):
                log.warning("The path does not exist, it will not load any model")
            else:
                log.info("load encoders pretrained weights from {}".format(path_pretrained))
                m = torch.load(path_pretrained, map_location="cpu")["models"][weight_name]
                self.load_state_dict_encoders(m, strict=False)

    def load_state_dict_with_same_shape(self, weights, strict=False):
        model_state = self.state_dict()
        if "KPConvPaper.pt" in getattr(self.opt, "path_pretrained", None):
            filtered_weights = self.modif_name_dict(weights, model_state)
        else:
            filtered_weights = {k: v for k, v in weights.items() if
                                k in model_state and v.size() == model_state[k].size()}
        log.info("Loading weights:" + ", ".join(filtered_weights.keys()))
        self.load_state_dict(filtered_weights, strict=strict)

    def modif_name_dict(self, weights, model_state):
        filtered_weights = {}
        for k, v in weights.items():
            if k in model_state and v.size() == model_state[k].size():
                filtered_weights[k] = v
            elif "up_modules" in k:
                knew = k.replace("up_modules", "up_modules_seg")
                if knew in model_state and v.size() == model_state[knew].size():
                    filtered_weights[knew] = v
            elif "FC_layer.1" in k:
                knew = k.replace("FC_layer.1", "FC_layer_seg.0")
                if knew in model_state and v.size() == model_state[knew].size():
                    filtered_weights[knew] = v
            elif "FC_layer.Class" in k:
                knew = k.replace("FC_layer.Class", "prototypes_seg.prototypes_seg")
                if knew in model_state and v.size() == model_state[knew].size():
                    filtered_weights[knew] = v
        return filtered_weights

    def load_state_dict_encoders(self, weights, strict=False):
        model_state = self.state_dict()
        filtered_weights = {k: v for k, v in weights.items() if
                            'down_modules' in k and (k in model_state and v.size() == model_state[k].size())}
        log.info("Loading weights:" + ", ".join(filtered_weights.keys()))
        self.load_state_dict(filtered_weights, strict=strict)
        for p in self.down_modules.parameters():
            if p.requires_grad:
                p.requires_grad = False
