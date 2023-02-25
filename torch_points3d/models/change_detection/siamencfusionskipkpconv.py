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

class BaseFactoryPSI:
    def __init__(self, module_name_down_1, module_name_down_2, module_name_up, modules_lib):
        self.module_name_down_1 = module_name_down_1
        self.module_name_down_2 = module_name_down_2
        self.module_name_up = module_name_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)
        elif "1" in flow:
            return getattr(self.modules_lib, self.module_name_down_1, None)
        else:
            return getattr(self.modules_lib, self.module_name_down_2, None)


####################SIAMESE ENCODER FUSION KP CONV ############################
class SiamEncFusionSkipKPConv(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
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

        # Unshared weight :  2 down modules
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
        self.loss_names = ["loss_cd"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]


    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.down_modules_1 = nn.ModuleList()
        self.down_modules_2 = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        self.save_sampling_id_1 = opt.down_conv_1.get('save_sampling_id')
        self.save_sampling_id_2 = opt.down_conv_2.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name_1 = opt.down_conv_1.module_name
        down_conv_cls_name_2 = opt.down_conv_2.module_name
        up_conv_cls_name = opt.up_conv.module_name if opt.get('up_conv') is not None else None
        self._factory_module = factory_module_cls(
            down_conv_cls_name_1, down_conv_cls_name_2, up_conv_cls_name, modules_lib
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
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_1, i, "DOWN_1")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_1.append(down_module)
        for i in range(len(opt.down_conv_2.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_2, i, "DOWN_2")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_2.append(down_module)

        # Up modules
        if up_conv_cls_name:
            for i in range(len(opt.up_conv.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules.append(up_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

    def _get_factory(self, model_name, modules_lib) -> BaseFactoryPSI:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactoryPSI
        return factory_module_cls

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


    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data0 = self.input0
        data1 = self.input1

        data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        diff = data1.clone()
        diff.x = data1.x - data0.x[nn_list[1, :], :]
        data1.x = torch.cat((data1.x, diff.x), axis=1)
        stack_down.append(data1)

        for i in range(1, len(self.down_modules_1) - 1):
            data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1,:],:]
            data1.x = torch.cat((data1.x, diff.x), axis=1)
            stack_down.append(data1)
        #1024
        data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)

        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1,:],:]
        data.x = torch.cat((data1.x, data.x), axis=1)
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
        last_feature = data.x
        if self._use_category:
            self.output = self.FC_layer(last_feature, self.category)
        else:
            self.output = self.FC_layer(last_feature)

        if self.labels is not None:
            self.compute_loss()

        self.data_visual = self.input1
        self.data_visual.pred = torch.max(self.output, -1)[1]

        return self.output

    def compute_loss(self):
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
            self.loss_seg = F.nll_loss(self.output, self.labels, weight=self._weight_classes, ignore_index=self._ignore_label)
        else:
            self.loss_seg = F.nll_loss(self.output, self.labels, weight=self._weight_classes)

        if torch.isnan(self.loss_seg).sum() == 1:
            print(self.loss_seg)
        self.loss += self.loss_seg

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G
