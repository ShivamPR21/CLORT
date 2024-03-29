import os
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ColorJitter, RandomApply
from tqdm import tqdm

import wandb
from clort import ArgoCL, ArgoCl_collate_fxn, ArgoCLSampler
from clort.model import (
    ContrastiveLoss,
    DLA34Encoder,
    MemoryBank,
    MemoryBankInfer,
    PCLGaussianNoise,
    PCLRigidTransformNoise,
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


def train(epoch, enc: DLA34Encoder, train_dl, optimizer, criterion, mem_bank, log_step=100, wb = True, model_device = 'cuda', reset: bool = False):
    mem_bank.reset() if reset else None # No resetting as it slows down training efforts
    # It is a logical construct for validation stage as final model has to work in isolated epochs

    enc.train() # Enable training

    training_loss = []

    # Training loop
    for itr, (_, _, imgs, imgs_sz, _, track_idxs, _cls_idxs, _, n_tracks) in (t_bar := tqdm(enumerate(train_dl), total=len(train_dl))):
        if len(track_idxs) == 0:
            continue

        optimizer.zero_grad()

        imgs = imgs.to(model_device) if isinstance(imgs, torch.Tensor) else imgs
        track_idxs = torch.from_numpy(track_idxs.astype(np.int32))

        encoding = enc(imgs, imgs_sz)

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
        for itr, (_, _, imgs, imgs_sz, _, track_idxs, _cls_idxs, _, n_tracks) in (v_bar := tqdm(enumerate(val_dl), total=len(val_dl))):
            if len(track_idxs) == 0:
                continue

            imgs = imgs.to(model_device) if isinstance(imgs, torch.Tensor) else imgs
            track_idxs = torch.from_numpy(track_idxs.astype(np.int32))

            encoding = enc(imgs, imgs_sz)

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

@hydra.main(version_base=None, config_path="./conf", config_name="config_dla")
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
                            point_cloud_scaling=cfg.dataset.pcl_scale,
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
                        point_cloud_scaling=cfg.dataset.pcl_scale,
                        in_global_frame=cfg.dataset.global_frame,
                        pivot_to_first_frame=cfg.dataset.pivot_to_first_frame,
                        image=cfg.dataset.imgs, pcl=cfg.dataset.pcl, bbox=cfg.dataset.bbox_aug)

    train_dl = DataLoader(train_dataset, cfg.dataset.batch, shuffle=False,
                          sampler=ArgoCLSampler(train_dataset, cfg.dataset.shuffle, True),
                          collate_fn=ArgoCl_collate_fxn, num_workers=cfg.dataset.workers,
                          prefetch_factor=cfg.dataset.prefetch, persistent_workers=cfg.dataset.persistent)

    val_dl = DataLoader(val_dataset, cfg.dataset.batch, shuffle=False,
                        sampler=ArgoCLSampler(val_dataset, False, True),
                        collate_fn=ArgoCl_collate_fxn, num_workers=cfg.dataset.workers,
                        prefetch_factor=cfg.dataset.prefetch, persistent_workers=cfg.dataset.persistent)

    # cfg.model.pc_features is not None

    print(f'{len(train_dl) = } \t {len(val_dl) = }')

    ## Initiate Encoders
    enc = DLA34Encoder(out_dim=cfg.model.sv_enc)
    enc = enc.to(cfg.model.device)

    # Initiate MemoryBanks
    assert(enc.out_dim is not None)
    mb = MemoryBank(train_dataset.n_tracks, enc.out_dim, cfg.mb.n_track_centers,
                    alpha=torch.tensor(cfg.mb.track_center_momentum, dtype=torch.float32, device=cfg.mb.device),
                    device=cfg.mb.device, init=cfg.mb.init, init_dilation=cfg.mb.init_dilation,
                    init_density=cfg.mb.init_density)

    cl = ContrastiveLoss(temp=cfg.loss.temperature, max_t=cfg.loss.max_t, global_contrast=cfg.loss.global_contrast,
                        separation=cfg.loss.separation, static_contrast=cfg.loss.static_contrast,
                        soft_condition=cfg.loss.soft_condition, global_horizon=cfg.loss.global_horizon,
                        sim_type=cfg.loss.sim_type, temperature_adaptation_policy=cfg.loss.temperature_adaptation_policy,
                        temperature_increase_coeff=cfg.loss.t_inc_coeff, pivot=cfg.loss.pivot)

    mb_infer = None

    if cfg.val_mb.mb == 'Infer':
        mb_infer = MemoryBankInfer(val_dataset.n_tracks, enc.out_dim, cfg.mb.n_track_centers, t = min(2, cfg.mb.n_track_centers),
                                   alpha_threshold=tuple(cfg.val_mb.alpha_t), beta_threshold=tuple(cfg.val_mb.beta_t), device=cfg.mb.device)
    else:
        mb_infer = MemoryBank(val_dataset.n_tracks, enc.out_dim, cfg.mb.n_track_centers,
                              alpha=torch.tensor(cfg.mb.track_center_momentum, dtype=torch.float32, device=cfg.mb.device),
                              device=cfg.mb.device)

    cl_infer = ContrastiveLoss(temp=cfg.loss.temperature, max_t=cfg.loss.max_t, global_contrast=cfg.loss.global_contrast,
                               separation=cfg.loss.separation, static_contrast=cfg.loss.static_contrast,
                               soft_condition=cfg.loss.soft_condition, global_horizon=cfg.loss.global_horizon,
                               sim_type=cfg.loss.sim_type, temperature_adaptation_policy=cfg.loss.temperature_adaptation_policy,
                               temperature_increase_coeff=cfg.loss.t_inc_coeff, pivot=cfg.loss.pivot)

    print(f'{train_dataset.n_tracks = } \t {val_dataset.n_tracks = }')

    # Initialize optimizer
    print(f'{cfg.optimizer.lr = } \t {cfg.optimizer.w_decay = }')
    optimizer = torch.optim.AdamW(params=enc.parameters(), lr = cfg.optimizer.lr, weight_decay=cfg.optimizer.w_decay)

    # Load model from file
    last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optimizer.decay_step,
                                                   gamma=cfg.optimizer.lr_decay, last_epoch=last_epoch)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 5, 6, 7], gamma=0.9, last_epoch=last_epoch)

    assert(run is not None and mb is not None and mb_infer is not None)

    if cfg.model.restore:
        print(f'Loading model from file: {cfg.model.model_file = } \t {cfg.model.run_path = }')
        ckpt = torch.load(run.restore(name=cfg.model.model_file, run_path=cfg.model.run_path).name)
        print(f'{enc.load_state_dict(ckpt["enc"], strict=False) = }') if cfg.restore.restore_model else print("Not restoring model parameters.")
        print(f'{optimizer.load_state_dict(ckpt["optimizer"]) = }') if cfg.restore.restore_optimizer else print("Not restoring optimizer parameters.")
        print(f'{lr_scheduler.load_state_dict(ckpt["lr_scheduler"]) = }') if cfg.restore.restore_scheduler else print("Not restoring learning rate scheduler parameters.")
        print(f'{mb.load_state_dict(ckpt["mb"]) = }') if cfg.restore.restore_mb else print("Not restoring memory bank parameters.")
        print(f'{mb_infer.load_state_dict(ckpt["mb_infer"]) = }') if cfg.restore.restore_mb else print("Not restoring memory bank parameters.")

    last_epoch = lr_scheduler.last_epoch
    n_epochs = cfg.optimizer.n_epochs
    print(f'{last_epoch = } \t {n_epochs = }')

    print(f'{len(train_dl) = } \t {len(val_dl) = }')

    if cfg.restore.restore_loss_t:
        # Restore temperature parameters
        for _ in range(last_epoch):
            cl._temp_step()
            cl_infer._temp_step()
            print(f'Restored temperature parameter: {cl.temp = } \t {cl_infer.temp = }')

    save_folder = os.path.join(run.dir, 'models')
    os.makedirs(save_folder, exist_ok=True)

    for epoch in range(last_epoch, n_epochs):
        model_fname = f'model_{epoch+1}.pth'
        model_path = os.path.join(save_folder, model_fname)

        wandb.log({'Training Loss Temperature': cl.temp,
                   'Validation Loss Temperature': cl_infer.temp})

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
        cl._temp_step()
        cl_infer._temp_step()

        model_info = {
            'EPOCH': epoch,
            'enc': enc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'mb': mb.state_dict(),
            'mb_infer': mb_infer.state_dict()
        }

        torch.save(model_info, model_path)
        wandb.save(os.path.join(save_folder, "./model*.pth"))

    run.finish()

if __name__ == '__main__':
    main()
