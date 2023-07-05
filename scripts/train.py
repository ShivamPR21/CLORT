import os
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ColorJitter, RandomApply
from tqdm import tqdm

import wandb
from clort import ArgoCL, ArgoCl_collate_fxn, ArgoCLSampler
from clort.model import (
    ContrastiveLoss,
    CrossObjectEncoder,
    MemoryBank,
    MemoryBankInfer,
    MultiModalEncoder,
    MultiViewEncoder,
    PCLGaussianNoise,
    PCLRigidTransformNoise,
    PointCloudEncoder,
)


def flatten_cfg(cfg: DictConfig) -> Dict[str, Any]:
    dct = {}

    for key1 in cfg.keys():
        for key2 in cfg[key1].keys():
            dct.update({f'{key1}.{key2}': cfg[key1][key2]})

    return dct


def check_gradients(net):
    g = []
    for n, p in net.named_parameters():
        if p.grad is not None and 'weight' in n.split('.'):
            # print(f'{n} => {p.grad.norm()}')
            g.append(p.grad.detach().cpu().norm())
    return np.array(g)


class CLModel(nn.Module):

    def __init__(self, mv_features: int | None = None, mv_xo: bool = False,
                 pc_features: int | None = None, bbox_aug: bool = True, pc_xo: bool = False,
                 mm_features: int | None = None, mm_xo: bool = False,
                 mmc_features: int | None = None) -> None:
        super().__init__()

        self.out_dim: int | None = None
        if mmc_features is not None:
            self.out_dim = mmc_features
        elif mm_features is not None:
            self.out_dim = mm_features
        elif pc_features is not None:
            self.out_dim = pc_features
        elif mv_features is not None:
            self.out_dim = mv_features
        else:
            raise NotImplementedError("Encoder resolution failed.")

        print(f'Model Config: {mv_features = } \t {mv_xo = } \t {pc_features = } \t {bbox_aug = } \n'
              f'{pc_xo = } \t {mm_features = } \t {mm_xo = } \t {mmc_features = }')
        print(f'Model Out Dims: {self.out_dim = }')

        self.mv_enc = MultiViewEncoder(out_dim=mv_features,
                                       norm_2d=nn.InstanceNorm2d,
                                       norm_1d=nn.LayerNorm,
                                       enable_xo=mv_xo) if mv_features is not None else None

        self.pc_enc = PointCloudEncoder(out_dims=pc_features, bbox_aug=bbox_aug,
                                        norm_layer=nn.LayerNorm, activation_layer=nn.SELU,
                                        offloading=False, enable_xo=pc_xo) if pc_features is not None else None

        self.mm_enc = MultiModalEncoder(mv_features, pc_features, mm_features, norm_layer=nn.LayerNorm,
                                        activation_layer=nn.SELU, enable_xo=mm_xo) if (mv_features is not None and pc_features is not None and mm_features is not None) else None

        self.mmc_enc = CrossObjectEncoder(mm_features, mmc_features, norm_layer=nn.LayerNorm,
                                          activation_layer=nn.SELU) if (mm_features is not None and mmc_features is not None) else None

    def forward(self, pcls: torch.Tensor | List[Any], pcls_sz: np.ndarray | List[Any],
                imgs: torch.Tensor | List[Any], imgs_sz: torch.Tensor | List[Any],
                bboxs: torch.Tensor | List[Any], frame_sz: np.ndarray) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        mv_e = self.mv_enc(imgs, imgs_sz, frame_sz) if self.mv_enc is not None else None
        pc_e = self.pc_enc(pcls, pcls_sz, frame_sz, bboxs) if self.pc_enc is not None else None

        mm_e = self.mm_enc(mv_e, pc_e) if self.mm_enc is not None else None

        mmc_e = self.mmc_enc(mm_e, frame_sz) if (self.mmc_enc is not None and mm_e is not None) else None

        return mv_e, pc_e, mm_e, mmc_e


def train(epoch, enc: CLModel, train_dl, optimizer, criterion, mem_bank, log_step=100, wb = True, model_device = 'cuda', reset: bool = False):
    mem_bank.reset() if reset else None # No resetting as it slows down training efforts
    # It is a logical construct for validation stage as final model has to work in isolated epochs

    enc.train() # Enable training

    training_loss = []

    # Training loop
    for itr, (pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, _cls_idxs, frame_sz, n_tracks) in (t_bar := tqdm(enumerate(train_dl), total=len(train_dl))):
        if len(track_idxs) == 0:
            continue

        optimizer.zero_grad()

        pcls = pcls.to(model_device) if isinstance(pcls, torch.Tensor) else pcls
        imgs = imgs.to(model_device) if isinstance(imgs, torch.Tensor) else imgs
        bboxs = bboxs.to(model_device) if isinstance(bboxs, torch.Tensor) else bboxs
        track_idxs = torch.from_numpy(track_idxs.astype(np.int32))

        mv_e, pc_e, mm_e, mmc_e = enc(pcls, pcls_sz, imgs, imgs_sz, bboxs, frame_sz)

        encoding = None
        if mmc_e is not None:
            encoding = mmc_e
        elif mm_e is not None:
            encoding = mm_e
        elif pc_e is not None:
            encoding = pc_e
        elif mv_e is not None:
            encoding = mv_e
        else:
            raise NotImplementedError("Encoder resolution failed.")

        mem_bank.update(encoding.detach().cpu(), track_idxs) # Update memory bank

        loss = criterion(encoding, track_idxs, mem_bank.get_memory(), n_tracks)
        training_loss.append(loss.numpy(force=True).item())

        loss.backward()

        gradient = check_gradients(enc)

        if np.any(gradient <= 1e-9):
            t_bar.set_description(f'Problem in gradients : {np.min(gradient)}')

        optimizer.step()

        if itr%log_step == log_step-1:
            t_bar.set_description(f'{epoch+1 = } and {itr+1 = } : Mean Training loss : {np.mean(training_loss[-log_step:])}')

            if wb:
                wandb.log({'Training Epoch': epoch+1, 'Training Iteration': itr+1,
                            'Training Loss': np.mean(training_loss[-log_step:])
                          })

    return training_loss

def val(epoch, enc, val_dl, criterion, mem_bank, log_step=100, wb = True, model_device = 'cuda', reset: bool = True):
    mem_bank.reset() if reset else None

    enc.eval() # Enable training

    validation_loss = []

    # Validation loop
    with torch.no_grad():
        for itr, (pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, _cls_idxs, frame_sz, n_tracks) in (v_bar := tqdm(enumerate(val_dl), total=len(val_dl))):
            if len(track_idxs) == 0:
                continue

            pcls = pcls.to(model_device) if isinstance(pcls, torch.Tensor) else pcls
            imgs = imgs.to(model_device) if isinstance(imgs, torch.Tensor) else imgs
            bboxs = bboxs.to(model_device) if isinstance(bboxs, torch.Tensor) else bboxs
            track_idxs = torch.from_numpy(track_idxs.astype(np.int32))

            mv_e, pc_e, mm_e, mmc_e = enc(pcls, pcls_sz, imgs, imgs_sz, bboxs, frame_sz)

            encoding = None
            if mmc_e is not None:
                encoding = mmc_e
            elif mm_e is not None:
                encoding = mm_e
            elif pc_e is not None:
                encoding = pc_e
            elif mv_e is not None:
                encoding = mv_e
            else:
                raise NotImplementedError("Encoder resolution failed.")

            mem_bank.update(encoding.detach().cpu(), track_idxs) # Update memory bank

            loss = criterion(encoding, track_idxs, mem_bank.get_memory(), n_tracks)
            validation_loss.append(loss.numpy(force=True).item())

            if itr%(log_step) == log_step-1:
                v_bar.set_description(f'{epoch+1 = } and {itr+1 = } : Mean Validation loss : {np.mean(validation_loss[-log_step:])}')

                if wb:
                    wandb.log({'Validation Epoch': epoch+1, 'Validation Iteration': itr+1,
                                'Validation Loss': np.mean(validation_loss[-log_step:])
                              })

    return validation_loss

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    wandb.login()

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.wb.project,

        resume=cfg.wb.resume,
        id=cfg.wb.run_id,

        # track hyper-parameters and run metadata
        config=flatten_cfg(cfg)
    )

    train_dataset = ArgoCL(cfg.dataset.root,
                           temporal_horizon=cfg.dataset.temporal_horizon,
                            temporal_overlap=cfg.dataset.temporal_overlap,
                            max_objects=cfg.dataset.max_objects,
                            target_cls=cfg.dataset.target_cls,
                            distance_threshold=cfg.dataset.distance_threshold,
                            splits=cfg.dataset.train_splits,
                            img_size=tuple(cfg.dataset.img_shape),
                            point_cloud_size=cfg.dataset.pcl_quant,
                            in_global_frame=cfg.dataset.global_frame,
                            pivot_to_first_frame=cfg.dataset.pivot_to_first_frame,
                            image=cfg.dataset.imgs, pcl=cfg.dataset.pcl, bbox=cfg.dataset.bbox_aug,
                            vision_transform=RandomApply([ColorJitter(0.5, 0, 0, 0.1)],
                                                            p=cfg.dataset.img_aug_prob), # type: ignore
                            pcl_transform=RandomApply([PCLGaussianNoise(mean=0, std=1, tr_lim=2), # type: ignore
                                                        PCLRigidTransformNoise(mean=0, std=np.pi/12, rot_lim=np.pi/12, trns_lim=2)],
                                                        p = cfg.dataset.pc_aug_prob)
                                                        )

    val_dataset = ArgoCL(cfg.dataset.root,
                        temporal_horizon=cfg.dataset.temporal_horizon,
                        temporal_overlap=cfg.dataset.temporal_overlap,
                        max_objects=None,
                        target_cls=cfg.dataset.target_cls,
                        distance_threshold=cfg.dataset.distance_threshold,
                        splits=cfg.dataset.val_splits,
                        img_size=tuple(cfg.dataset.img_shape),
                        point_cloud_size=cfg.dataset.pcl_quant,
                        in_global_frame=cfg.dataset.global_frame,
                        pivot_to_first_frame=cfg.dataset.pivot_to_first_frame,
                        image=cfg.dataset.imgs, pcl=cfg.dataset.pcl, bbox=cfg.dataset.bbox_aug)

    train_dl = DataLoader(train_dataset, cfg.dataset.batch, shuffle=False,
                          sampler=ArgoCLSampler(train_dataset, cfg.dataset.shuffle),
                          collate_fn=ArgoCl_collate_fxn, num_workers=cfg.dataset.workers,
                          prefetch_factor=cfg.dataset.prefetch, persistent_workers=cfg.dataset.persistent)

    val_dl = DataLoader(val_dataset, cfg.dataset.batch, shuffle=False,
                        sampler=ArgoCLSampler(val_dataset, False),
                        collate_fn=ArgoCl_collate_fxn, num_workers=cfg.dataset.workers,
                        prefetch_factor=cfg.dataset.prefetch, persistent_workers=cfg.dataset.persistent)

    print(f'{len(train_dl) = } \t {len(val_dl) = }')

    ## Initiate Encoders
    enc = CLModel(cfg.model.mv_features, cfg.model.mv_xo, cfg.model.pc_features, cfg.dataset.bbox_aug,
                  cfg.model.pc_xo, cfg.model.mm_features, cfg.model.mm_xo, cfg.model.mmc_features)
    enc = enc.to(cfg.model.device)

    # Initiate MemoryBanks
    assert(enc.out_dim is not None)
    mb = MemoryBank(train_dataset.n_tracks, enc.out_dim, cfg.mb.n_track_centers,
                    alpha=torch.tensor(cfg.mb.track_center_momentum, dtype=torch.float32, device=cfg.mb.device),
                    device=cfg.mb.device, init=cfg.mb.init, init_dilation=cfg.mb.init_dilation,
                    init_density=cfg.mb.init_density)

    cl = ContrastiveLoss(temp=cfg.loss.temperature, global_contrast=cfg.loss.global_contrast,
                        separate_tracks=cfg.loss.separate_tracks, static_contrast=cfg.loss.static_contrast,
                        soft_condition=cfg.loss.soft_condition, global_horizon=cfg.loss.global_horizon, sim_type=cfg.loss.sim_type)

    mb_infer = None

    if cfg.val_mb.mb == 'Infer':
        mb_infer = MemoryBankInfer(val_dataset.n_tracks, enc.out_dim, cfg.mb.n_track_centers, t = min(2, enc.out_dim), device=cfg.mb.device) if enc.out_dim is not None else None
    else:
        mb_infer = MemoryBank(val_dataset.n_tracks, enc.out_dim, cfg.mb.n_track_centers,
                              alpha=torch.tensor(cfg.mb.track_center_momentum, dtype=torch.float32, device=cfg.mb.device),
                              device=cfg.mb.device)

    cl_infer = ContrastiveLoss(temp=cfg.loss.temperature, global_contrast=cfg.loss.global_contrast,
                               separate_tracks=cfg.loss.separate_tracks, static_contrast=cfg.loss.static_contrast,
                               soft_condition=cfg.loss.soft_condition, global_horizon=cfg.loss.global_horizon, sim_type=cfg.loss.sim_type)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
                        params=[
                            {'params' : enc.parameters(), 'lr': cfg.optimizer.lr, "weight_decay": cfg.optimizer.w_decay}
                        ], lr = cfg.optimizer.lr, weight_decay=cfg.optimizer.w_decay
                    )

    # Load model from file
    last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optimizer.decay_step,
                                                   gamma=cfg.optimizer.lr_decay, last_epoch=last_epoch)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 5, 6, 7], gamma=0.9, last_epoch=last_epoch)

    # training_loss: List[float] = []
    # validation_loss: List[float] = []

    assert(run is not None and mb is not None and mb_infer is not None)

    if cfg.model.restore:
        print(f'Loading model from file: {cfg.model.model_file = } \t {cfg.model.run_path = }')
        ckpt = torch.load(run.restore(name=cfg.model.model_file, run_path=cfg.model.run_path).name)
        print(f'{enc.load_state_dict(ckpt["enc"]) = }')
        print(f'{optimizer.load_state_dict(ckpt["optimizer"]) = }')
        print(f'{lr_scheduler.load_state_dict(ckpt["lr_scheduler"]) = }')
        print(f'{mb.load_state_dict(ckpt["mb"]) = }')
        print(f'{mb_infer.load_state_dict(ckpt["mb_infer"]) = }')

    last_epoch = lr_scheduler.last_epoch
    n_epochs = cfg.optimizer.n_epochs
    print(f'{last_epoch = } \t {n_epochs = }')

    print(f'{len(train_dl) = } \t {len(val_dl) = }')

    for epoch in range(last_epoch, n_epochs):
        model_fname = f'model_{epoch+1}.pth'
        model_path = os.path.join(run.dir, model_fname)

        _train_loss = train(epoch, enc, train_dl, optimizer,
                            cl, mb, log_step=cfg.optimizer.log_freq, wb=True, model_device=cfg.model.device, reset=cfg.mb.reset)

        ###################################################################################
        ### Validation loss
        if epoch%cfg.optimizer.val_freq == cfg.optimizer.val_freq-1:
            _val_loss = val(epoch, enc, val_dl, cl_infer,
                            mb_infer, log_step=cfg.optimizer.log_freq, wb=True, model_device=cfg.model.device, reset=cfg.val_mb.reset)
        ### Validation loss
        ###################################################################################
        lr_scheduler.step() # Step Learning rate

        model_info = {
            'EPOCH': epoch,
            'enc': enc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'mb': mb.state_dict(),
            'mb_infer': mb_infer.state_dict()
        }

        torch.save(model_info, model_path)

        wandb.save(os.path.join(run.dir, "./model*.pth"))

    # Final Validation loop

    run.finish()

if __name__ == '__main__':
    main()
