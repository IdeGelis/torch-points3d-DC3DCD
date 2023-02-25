import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer_deepCluster import Trainer
import shutil
import os.path as osp

@hydra.main(config_path="conf", config_name="evalDC") #evalUrb3D configDCVA_3D
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    # trainer.get_pclust()
    trainer.eval()
    # pathRes = osp.join(trainer._dataset.dataset_opt.dataTestFile, 'tp3D', 'res', cfg['tracker_options']['name_test'], 'chkpt')
    # shutil.copytree(cfg['checkpoint_dir'], pathRes)
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
