import warnings
# warnings.filterwarnings("ignore")
import os
import os.path as osp
import copy
import torch
import hydra
import time
import logging
import numpy as np
# from tqdm.auto import tqdm
from tqdm import tqdm
import wandb
from sklearn.decomposition import PCA
import skimage.filters as skf
import sklearn.metrics as skmetric
# from openTSNE import TSNE
# import umap
import matplotlib.pyplot as plt

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.visualization import Visualizer

# deepCluster import
import torch_points3d.clustering_dc as clustering_dc

log = logging.getLogger(__name__)


class Trainer:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
        - A custom Visualizer
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):
        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn
        log.info(self._cfg.pretty())
        if not self.has_training:
            resume = False
            self._cfg.training = self._cfg
        else:
            resume = bool(self._cfg.training.checkpoint_dir)
        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.wandb.public and self.wandb_log)

        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )
        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
            self._dataset.set_nbclass(self._cfg.deepclustering.nb_cluster_cd, self._cfg.deepclustering.nb_cluster_seg)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
        else:
            self._cfg.data.nb_cluster_kmeans_cd = self._cfg.deepclustering.nb_cluster_cd
            self._cfg.data.nb_cluster_kmeans_seg = self._cfg.deepclustering.nb_cluster_seg
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._dataset.set_nbclass(self._cfg.deepclustering.nb_cluster_cd, self._cfg.deepclustering.nb_cluster_seg)
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            # self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            self._model.set_pretrained_weights()
            self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Upgrade class name for deep cluster
        self._dataset.INV_OBJECT_LABEL = {i: "class " + str(i) for i in range(self._dataset.num_classes)}
        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        log.info(self._dataset)

        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log,
                                                               self.tracker_options.full_pc,
                                                               self.tracker_options.full_res)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
            )

        self._model.print_nb_param(log)
        # Chkpt temporary
        path = osp.join(os.getcwd(), "segsem")
        if not osp.exists(path):
            os.makedirs(path)
        self._checkpoint_tmp_epoch: ModelCheckpoint = ModelCheckpoint(
            path,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=False,
        )
        self._checkpoint_tmp_epoch.dataset_properties = self._dataset.used_properties

    def train_deepCluster(self):
        self._is_training = True
        self.pca = None
        comp_clust = True
        self._tracker.reset_dc()
        # Set dataloaders
        train_kmeans_loader = self._dataset._dataloader(
            self._dataset.train_data_kmeans,
            self._dataset.train_pre_batch_collate_transform,
            self._model.conv_type,
            self.precompute_multi_scale,
            batch_size=self._cfg.training.batch_size,
            shuffle=self._cfg.training.shuffle and not self._dataset.train_sampler,
            num_workers=self._cfg.training.num_workers,
            sampler=self._dataset.train_sampler,
        )
        centroids_cd = None
        if self._cfg.deepclustering.dualTask:
            centroids_seg = None
            if self._model.option.deepCluster.y_from_threshold:
                centroids_segdiff = None

        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs + 1):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)
            seg_task = False
            cd_task = False
            segbin_task = False

            if epoch <= self._model.option.deepCluster.nSeg_epoch:
                seg_task = True
                self._dataset.train_dataset.unset_binaryCD_segsem()
            elif epoch == self._model.option.deepCluster.nSeg_epoch + 1:

                segbin_task = True
                self._dataset.train_dataset.set_binaryCD_segsem()
            elif epoch <= self._model.option.deepCluster.nSeg_epoch + 1 + self._model.option.deepCluster.nCD_epoch:
                cd_task = True
                self._dataset.train_dataset.unset_binaryCD_segsem()

            if seg_task or cd_task or segbin_task:
                # Compute feature and run mini-batch k-means
                centroids_cd, kmloss_cd, centroids_seg, kmloss_seg = self.run_mini_batch_kmeans(epoch,
                                                                                                train_kmeans_loader,
                                                                                                centroids_cd=None,
                                                                                                centroids_seg=None,
                                                                                                cd_task=cd_task,
                                                                                                seg_task=seg_task,
                                                                                                segbin_task=segbin_task)
                # Cluster deep features
                print('Cluster features')
                self.compute_labels(epoch, centroids_cd, centroids_seg, train_kmeans_loader, cd_task=cd_task,
                                    seg_task=seg_task, segbin_task=segbin_task)

                # Cluster assignment
                self._tracker.save_pseudo_label_map()
                # Modify classes weights for loss weighting into model definition according to pseudo-labels
                self._dataset.train_data.get_weights_pseudolabel()
                self._model.set_class_weight(self._dataset.train_data)

                if epoch == self._model.option.deepCluster.nSeg_epoch + 1:
                    # Re-initialize Lr and batch norm
                    self._model.instantiate_optimizers(self._cfg, "cuda" == self._device)

                with torch.no_grad():
                    if self._cfg.deepclustering.dualTask:
                        if seg_task:
                            centroids_n = torch.from_numpy(centroids_seg)
                            centroids_n = torch.nn.functional.normalize(centroids_n, dim=1, p=2)
                            self._model.prototypes_seg.prototypes_seg.weight.copy_(centroids_n)
                        elif cd_task:
                            centroids_n = torch.from_numpy(centroids_cd)
                            centroids_n = torch.nn.functional.normalize(centroids_n, dim=1, p=2)
                            self._model.prototypes_cd.prototypes_cd.weight.copy_(centroids_n)
                    else:
                        centroids_n = torch.from_numpy(centroids_cd)
                        centroids_n = torch.nn.functional.normalize(centroids_n, dim=1, p=2)
                        self._model.prototypes_cd.prototypes_cd.weight.copy_(centroids_n)

            if epoch != self._model.option.deepCluster.nSeg_epoch + 1:
                # no train for one epoch when binary change computed from kmean on difference of feats from seg
                # Set dataloaders
                train_loader = self._dataset._dataloader(
                    self._dataset.train_data,
                    self._dataset.train_pre_batch_collate_transform,
                    self._model.conv_type,
                    self.precompute_multi_scale,
                    batch_size=self._cfg.training.batch_size,
                    shuffle=self._cfg.training.shuffle and not self._dataset.train_sampler,
                    num_workers=self._cfg.training.num_workers,
                    sampler=self._dataset.train_sampler,
                )
                # train one epoch based on deep clusters
                self._train_epoch(epoch, train_loader)
            if epoch > 205:  # % self.eval_frequency == 0:
                val_loader = self._dataset._dataloader(
                    self._dataset.val_data,
                    self._dataset.val_pre_batch_collate_transform,
                    self._model.conv_type,
                    self.precompute_multi_scale,
                    batch_size=self._cfg.training.batch_size,
                    shuffle=self._cfg.training.shuffle and not self._dataset.val_sampler,
                    num_workers=self._cfg.training.num_workers,
                    sampler=self._dataset.train_sampler,
                )
                self._test_epoch(epoch, "val", val_loader)


    def run_mini_batch_kmeans(self, epoch, data_loader=None, centroids_cd=None, centroids_seg=None, cd_task=True,
                              seg_task=False, segbin_task = False):
        """
        num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
        num_batches     : (int) The number of batches/iterations to accumulate before the next update.
        """
        kmeans_loss_cd = clustering_dc.AverageMeter()
        kmeans_loss_seg = clustering_dc.AverageMeter()
        kmeans_loss_segdiff = clustering_dc.AverageMeter()
        data_count_cd = np.zeros(self._cfg.deepclustering.nb_cluster_cd)
        data_count_seg = np.zeros(self._cfg.deepclustering.nb_cluster_seg)
        data_count_segdiff = np.zeros(2)
        featslist_cd = []
        featslist_seg = []
        featslist_segdiff = []
        num_batches = 0

        if cd_task and centroids_cd is not None:
            centroids = True
        elif seg_task and centroids_seg is not None:
            centroids = True
        else:
            centroids = False

        self._model.eval()
        if data_loader is None:
            train_loader = self._dataset.train_dataloader
        else:
            train_loader = data_loader
        with Ctq(train_loader) as tq_train_loader:
            for i_batch, data in enumerate(tq_train_loader):
                with torch.no_grad():
                    # 1. Compute initial centroids from the first few batches.
                    self._model.set_input(data, self._device)
                    with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                        self._model.forward(epoch=epoch, compute_loss=False)
                    if cd_task:
                        feats_cd = self._model.last_features_cd
                        # Normalise L2 norm (optional)
                        # feats_cd = torch.nn.functional.normalize(feats_cd, dim=1, p=2)
                        featslist_cd.append(feats_cd)
                    if seg_task or segbin_task:
                        feats_seg0 = self._model.last_features_seg_d0
                        feats_seg1 = self._model.last_features_seg_d1
                    if seg_task:
                        # feats_seg0 = torch.nn.functional.normalize(feats_seg0, dim=1, p=2)
                        # feats_seg1 = torch.nn.functional.normalize(feats_seg1, dim=1, p=2)
                        featslist_seg.append(feats_seg0)
                        featslist_seg.append(feats_seg1)

                    if segbin_task:
                        fdiff = torch.norm(feats_seg1 - feats_seg0[self._model.nn_list_init[1, :], :], dim= 1, p=2)
                        featslist_segdiff.append(fdiff)
                    num_batches += 1
                    if num_batches == self._cfg.deepclustering.num_init_batches or (i_batch + 1) == len(train_loader):
                        if not centroids:
                            # Compute initial centroids.
                            # By doing so, we avoid empty cluster problem from mini-batch K-Means.
                            if cd_task:
                                featslist_cd = torch.cat(featslist_cd).cpu().numpy().astype('float32')
                                if self.pca is not None:
                                    featslist_cd = self.pca.transform(featslist_cd)
                                I_cd, loss_cd, centroids_cd, faiss_module_cd = clustering_dc.run_kmeans(featslist_cd,
                                                                                                        self._cfg.deepclustering.nb_cluster_cd)
                                kmeans_loss_cd.update(loss_cd)

                            if seg_task:
                                featslist_seg = torch.cat(featslist_seg).cpu().numpy().astype('float32')
                                I_seg, loss_seg, centroids_seg, faiss_module_seg = clustering_dc.run_kmeans(
                                    featslist_seg,
                                    self._cfg.deepclustering.nb_cluster_seg)
                                kmeans_loss_seg.update(loss_seg)
                            if segbin_task:
                                featslist_segdiff = torch.cat(featslist_segdiff).cpu().numpy().astype('float32')
                                featslist_segdiff = featslist_segdiff.reshape(featslist_segdiff.shape[0],1)
                                I_segdiff, loss_segdiff, centroids_segdiff, faiss_module_segdiff = clustering_dc.run_kmeans(
                                    featslist_segdiff, 2)
                                kmeans_loss_segdiff.update(loss_segdiff)
                            # log.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))

                            # Compute counts for each cluster.
                            if cd_task:
                                for k in np.unique(I_cd):
                                    data_count_cd[k] += len(np.where(I_cd == k)[0])
                            if seg_task:
                                for k in np.unique(I_seg):
                                    data_count_seg[k] += len(np.where(I_seg == k)[0])
                            if segbin_task:
                                for k in np.unique(I_segdiff):
                                    data_count_segdiff[k] += len(np.where(I_segdiff == k)[0])

                        else:
                            if cd_task:
                                featslist_cd = torch.cat(featslist_cd).cpu().numpy().astype('float32')

                                if self.pca is not None:
                                    featslist_cd = self.pca.transform(featslist_cd)
                                if faiss_module_cd is None:
                                    faiss_module_cd = clustering_dc.get_faiss_module(featslist_cd.shape[1])
                                faiss_module_cd = clustering_dc.module_update_centroids(faiss_module_cd, centroids_cd)
                                D_cd, I_cd = faiss_module_cd.search(featslist_cd, 1)
                                kmeans_loss_cd.update(D_cd.mean())

                                # Update centroids.
                                for k in np.unique(I_cd):
                                    idx_k = np.where(I_cd == k)[0]
                                    data_count_cd[k] += len(idx_k)
                                    centroid_lr_cd = len(idx_k) / (data_count_cd[k] + 1e-6)
                                    centroids_cd[k] = (1 - centroid_lr_cd) * centroids_cd[k] + centroid_lr_cd * np.mean(
                                        featslist_cd[idx_k], axis=0)

                            if seg_task:
                                featslist_seg = torch.cat(featslist_seg).cpu().numpy().astype('float32')
                                if faiss_module_seg is None:
                                    faiss_module_seg = clustering_dc.get_faiss_module(featslist_seg.shape[1])
                                faiss_module_seg = clustering_dc.module_update_centroids(faiss_module_seg,
                                                                                         centroids_seg)
                                D_seg, I_seg = faiss_module_seg.search(featslist_seg, 1)
                                kmeans_loss_seg.update(D_seg.mean())

                                # Update centroids.
                                for k in np.unique(I_seg):
                                    idx_k = np.where(I_seg == k)[0]
                                    data_count_seg[k] += len(idx_k)
                                    centroid_lr_seg = len(idx_k) / (data_count_seg[k] + 1e-6)
                                    centroids_seg[k] = (1 - centroid_lr_seg) * centroids_seg[
                                        k] + centroid_lr_seg * np.mean(
                                        featslist_seg[idx_k], axis=0)

                            if segbin_task:
                                featslist_segdiff = torch.cat(featslist_segdiff).cpu().numpy().astype('float32')
                                featslist_segdiff = featslist_segdiff.reshape(featslist_segdiff.shape[0], 1)
                                if faiss_module_segdiff is None:
                                    faiss_module_segdiff = clustering_dc.get_faiss_module(
                                        featslist_segdiff.shape[1])
                                faiss_module_segdiff = clustering_dc.module_update_centroids(faiss_module_segdiff,
                                                                                             centroids_segdiff)
                                D_segdiff, I_segdiff = faiss_module_segdiff.search(featslist_segdiff, 1)
                                kmeans_loss_segdiff.update(D_segdiff.mean())

                                # Update centroids.
                                for k in np.unique(I_segdiff):
                                    idx_k = np.where(I_segdiff == k)[0]
                                    data_count_seg[k] += len(idx_k)
                                    centroid_lr_segdiff = len(idx_k) / (data_count_segdiff[k] + 1e-6)
                                    centroids_segdiff[k] = (1 - centroid_lr_segdiff) * centroids_segdiff[
                                        k] + centroid_lr_segdiff * np.mean(
                                        featslist_segdiff[idx_k], axis=0)

                        # Empty.
                        featslist_cd = []
                        featslist_seg = []
                        featslist_segdiff = []
                        # label_gt = []
                        num_batches = self._cfg.deepclustering.num_init_batches - self._cfg.deepclustering.num_batches
        if segbin_task:
            log.info('[Binary K-Means Loss SEG DIFF]: {:.4f}'.format(kmeans_loss_segdiff.avg))
            self.centroids_segdiff = centroids_segdiff
        kmeans_loss_cd_avg = None
        kmeans_loss_seg_avg = None
        if cd_task:
            log.info('[K-Means Loss CD]: {:.4f}'.format(kmeans_loss_cd.avg))
            kmeans_loss_cd_avg = kmeans_loss_cd.avg
        if seg_task:
            log.info('[K-Means Loss SEG]: {:.4f}'.format(kmeans_loss_seg.avg))
            kmeans_loss_seg_avg = kmeans_loss_seg.avg
        return centroids_cd, kmeans_loss_cd_avg, centroids_seg, kmeans_loss_seg_avg


    def compute_labels(self, epoch, centroids_cd=None, centroids_seg=None, data_loader=None, cd_task=True,
                              seg_task=False, segbin_task = False):
        """
        Label all images for each view with the obtained cluster centroids.
        The distance is efficiently computed by setting centroids as convolution layer.
        """
        self._dataset.train_data.reset_pseudo_label()
        if cd_task:
            index_cd = clustering_dc.index_from_centroid(centroids_cd)
        else:
            index_cd = None
        if seg_task:
            index_seg = clustering_dc.index_from_centroid(centroids_seg)
        else:
            index_seg = None
        if segbin_task:
            # 1 unchange (nearest distance to 0) 0 change (farest distance to 0) --> for contrastive loss
            arg = np.argsort(np.linalg.norm(self.centroids_segdiff, axis=1), )[::-1]
            self.index_segdiff = clustering_dc.index_from_centroid(self.centroids_segdiff[arg, :])
        else:
            self.index_segdiff = None

        if data_loader is None:
            train_loader = self._dataset.train_dataloader
        else:
            train_loader = data_loader
        self._model.eval()
        self._compute_label(epoch, train_loader, self._dataset.train_data, index_cd, index_seg, stage='train')

        if epoch > self._model.option.deepCluster.nSeg_epoch + 1 + self._model.option.deepCluster.nCD_epoch:  # % self.eval_frequency == 0: #>= pour qu'au dernier kmean soit calculé les plab du val
            self._dataset.val_data.reset_pseudo_label()
            val_loader = self._dataset.val_dataloader
            self._compute_label(epoch, val_loader, self._dataset.val_data, index_cd, index_seg, stage='val')

    def _compute_label(self, epoch, data_loader, dataset, index_cd, index_seg, stage='train'):
        self._tracker.reset_unsup_met_epoch()
        self._tracker.set_stage(stage)
        labels_onehot_cd = None
        labels_onehot_seg0 = None
        labels_onehot_seg1 = None
        with Ctq(data_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                with torch.no_grad():
                    # 1. Compute initial centroids from the first few batches.
                    self._model.set_input(data, self._device)
                    with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                        self._model.forward(epoch=epoch, compute_loss=False)
                    if index_cd is not None:
                        # Normalize. L2 normalisation
                        # self._model.last_feature = torch.nn.functional.normalize(self._model.last_feature, dim=1, p=2)
                        feats_cd = self._model.last_features_cd.cpu().numpy().astype('float32')
                        if self.pca is not None:
                            feats_cd = self.pca.transform(feats_cd)
                        # feats_cd = torch.nn.functional.normalize(feats_cd, dim=1, p=2).cpu().numpy().astype('float32')
                        # Compute distance and assign label.
                        distances_cd, labels_cd = index_cd.search(np.ascontiguousarray(feats_cd), 1)
                        labels_onehot_cd = torch.nn.functional.one_hot(torch.from_numpy(labels_cd.T.squeeze()),
                                                                       self._cfg.deepclustering.nb_cluster_cd)

                    if index_seg is not None or self.index_segdiff is not None:
                        feats_seg0 = self._model.last_features_seg_d0.cpu().numpy().astype('float32')
                        # feats_seg0 = torch.nn.functional.normalize(feats_seg0, dim=1, p=2).cpu().numpy().astype(
                        #     'float32')
                        feats_seg1 = self._model.last_features_seg_d1.cpu().numpy().astype('float32')
                        # feats_seg1 = torch.nn.functional.normalize(feats_seg1, dim=1, p=2).cpu().numpy().astype(
                        #     'float32')
                    if index_seg is not None:
                        # Compute distance and assign label.
                        distances_seg0, labels_seg0 = index_seg.search(np.ascontiguousarray(feats_seg0), 1)
                        labels_onehot_seg0 = torch.nn.functional.one_hot(torch.from_numpy(labels_seg0.T.squeeze()),
                                                                         self._cfg.deepclustering.nb_cluster_seg)

                        distances_seg1, labels_seg1 = index_seg.search(np.ascontiguousarray(feats_seg1), 1)
                        labels_onehot_seg1 = torch.nn.functional.one_hot(torch.from_numpy(labels_seg1.T.squeeze()),
                                                                         self._cfg.deepclustering.nb_cluster_seg)

                    if index_seg is not None:
                        dataset.add_batch_pseudo_label(data, labels_onehot_cd, labels_onehot_seg0,
                                                       labels_onehot_seg1)
                    elif index_cd is not None:
                        dataset.add_batch_pseudo_label(data, labels_onehot_cd)

                    if self.index_segdiff is not None:
                        feats_segdiff = np.linalg.norm(feats_seg1 - feats_seg0[self._model.nn_list_init[1, :].cpu().numpy(), :], axis=1, ord=2)
                        feats_segdiff = feats_segdiff.reshape(feats_segdiff.shape[0], 1)
                        distances_segdiff, labels_segdiff = self.index_segdiff.search(
                            np.ascontiguousarray(feats_segdiff), 1)
                        labels_onehot_segdiff = torch.nn.functional.one_hot(
                            torch.from_numpy(labels_segdiff.T.squeeze()), 2)
                        dataset.add_batch_pseudo_label_change(data, labels_onehot_segdiff)
                    if index_cd is not None:
                        self._tracker.track_epoch_unsup_met(feats_cd, labels_cd)
        dataset.finalise_pseudo_label()

        self._tracker.track_dc(dataset, verbose=True, rd_eq_sampling=self.tracker_options.get("rd_eq_sampling", False))
        self._tracker.plot_save_metric_dc(plot=True, saving_path=os.path.join(os.getcwd(), stage),
                                          epoch=epoch)
        if self.index_segdiff:
            self._tracker.save_data_binSegSem(dataset, epoch=epoch, saving_path=os.path.join(os.getcwd(), stage))


    def eval(self, stage_name=""):
        self._is_training = False
        epoch = self._checkpoint.start_epoch
        if osp.isfile(self._cfg.deepclustering.map_pclust_gt):
            map_pclust_gt = np.loadtxt(self._cfg.deepclustering.map_pclust_gt).astype('int32')
            self._tracker.pseudolabel_label_map = map_pclust_gt
        else:
            self._test_epoch(epoch, "test", loader=self._dataset.test_dataloaders, track=False)
            gt_tot = []
            pred_tot = []
            for i, area in enumerate(self._tracker._areas):
                if area is not None:
                    # Complete for points that have a prediction
                    area = area.to("cpu")
                    has_prediction = area.prediction_count > 0
                    pred = torch.argmax(area.votes[has_prediction], 1)
                    pred = pred.numpy()
                    gt = area.y[has_prediction].numpy()
                    gt_tot.append(gt)
                    pred_tot.append(pred)

            label = np.concatenate(gt_tot)
            pred_tot = np.concatenate(pred_tot)
            self._tracker._get_repartition(label, pred_tot)
            self._tracker._calc_pseudo_label_map()
        self._model.set_class_weight(self._dataset.train_data)

        if self._dataset.has_test_loaders:
            if not stage_name or stage_name == "test":
                self._test_epoch(epoch, "test", loader=self._dataset.test_dataloaders, track=True,
                                 use_pseudo_label_map=True)

    def get_pclust(self, stage_name=""):
        self._is_training = False
        epoch = self._checkpoint.start_epoch
        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch(epoch, "val", loader=self._dataset.val_dataloader, track=True,
                                 use_pseudo_label_map=False)
        if self._dataset.has_test_loaders:
            if not stage_name or stage_name == "test":
                self._test_epoch(epoch, "test", loader=self._dataset.test_dataloaders, track=True,
                                 use_pseudo_label_map=False)

        train_all_loader = self._dataset._dataloader(
            self._dataset.train_data_kmeans,
            self._dataset.train_pre_batch_collate_transform,
            self._model.conv_type,
            self.precompute_multi_scale,
            batch_size=self._cfg.training.batch_size,
            shuffle=self._cfg.training.shuffle and not self._dataset.train_sampler,
            num_workers=self._cfg.training.num_workers,
            sampler=self._dataset.train_sampler,
        )
        self._test_epoch(epoch, "train", loader=train_all_loader, track=True,
                         use_pseudo_label_map=False)




    def _finalize_epoch(self, epoch, pseudo_label_map=None):
        if pseudo_label_map is not None:
            self._tracker.finalise(conv_classes=pseudo_label_map, num_class_cm=self._tracker._num_classes_real,
                                   **self.tracker_options)
        else:
            self._tracker.finalise(**self.tracker_options)
        if self._is_training:
            metrics = self._tracker.publish(epoch)
            self._checkpoint.save_best_models_under_current_metrics(self._model, metrics, self._tracker.metric_func)
            if self.wandb_log and self._cfg.wandb.public:
                Wandb.add_file(self._checkpoint.checkpoint_path)
            if self._tracker._stage == "train":
                log.info("Learning rate = %f" % self._model.learning_rate)
            if epoch == self._model.option.deepCluster.nSeg_epoch:
                self._checkpoint_tmp_epoch.save_best_models_under_current_metrics(self._model, metrics,
                                                                                  self._tracker.metric_func)

    def _train_epoch(self, epoch: int, train_loader=None):

        self._model.train()
        self._tracker.reset("train")
        self._visualizer.reset(epoch, "train")
        if train_loader is None:
            train_loader = self._dataset.train_dataloader

        iter_data_time = time.time()
        with Ctq(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                self._model.set_input(data, self._device)
                self._model.optimize_parameters(epoch, self._dataset.batch_size)
                if i % 10 == 0:
                    with torch.no_grad():
                        self._tracker.track(self._model, data=data, **self.tracker_options)

                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics_supervised(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                    color=COLORS.TRAIN_COLOR
                )

                if self._visualizer.is_active:
                    self._visualizer.save_visuals(self._model.get_current_visuals())

                iter_data_time = time.time()

                if self.early_break:
                    break

                if self.profiling:
                    if i > self.num_batches:
                        return 0

        self._finalize_epoch(epoch)

    def _test_epoch(self, epoch, stage_name: str, loader, track=True, use_pseudo_label_map=False):
        voting_runs = self._cfg.get("voting_runs", 1)
        # if stage_name == "test":
        #     loaders = self._dataset.test_dataloaders
        # else:
        #     loaders = [self._dataset.val_dataloader]
        if type(loader) != list:
            loaders = [loader]
        else:
            loaders = loader

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        for loader in loaders:
            # stage_name = loader.dataset.name
            self._tracker.reset(stage_name)
            if self.has_visualization:
                self._visualizer.reset(epoch, stage_name)
            if not self._dataset.has_labels(stage_name) and not self.tracker_options.get(
                    "make_submission", False
            ):  # No label, no submission -> do nothing
                log.warning("No forward will be run on dataset %s." % stage_name)
                continue

            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            self._model.set_input(data, self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model.forward(epoch=epoch, compute_loss=False)
                            if use_pseudo_label_map:
                                self._tracker.track(self._model, data=data,
                                                    conv_classes=self._tracker.pseudolabel_label_map,
                                                    **self.tracker_options)
                            else:
                                self._tracker.track(self._model, data=data, **self.tracker_options)
                        if track:
                            tq_loader.set_postfix(**self._tracker.get_metrics_supervised(), color=COLORS.TEST_COLOR)

                        if self.has_visualization and self._visualizer.is_active:
                            self._visualizer.save_visuals(self._model.get_current_visuals())

                        if self.early_break:
                            break

                        if self.profiling:
                            if i > self.num_batches:
                                return 0
            if track:
                if use_pseudo_label_map:
                    self._finalize_epoch(epoch, pseudo_label_map=self._tracker.pseudolabel_label_map)
                else:
                    self._finalize_epoch(epoch)
                self._tracker.print_summary()

    @property
    def early_break(self):
        return getattr(self._cfg.debugging, "early_break", False) and self._is_training

    @property
    def profiling(self):
        return getattr(self._cfg.debugging, "profiling", False)

    @property
    def num_batches(self):
        return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", True)

    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    @property
    def wandb_log(self):
        if getattr(self._cfg, "wandb", False):
            return getattr(self._cfg.wandb, "log", False)
        else:
            return False

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.tensorboard, "log", False)
        else:
            return False

    @property
    def tracker_options(self):
        return self._cfg.get("tracker_options", {})

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)
