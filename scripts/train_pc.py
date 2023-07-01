import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import RandomApply
from tqdm import tqdm

from clort import ArgoCL, ArgoCl_collate_fxn
from clort.model import (
    ContrastiveLoss,
    MemoryBank,
    MemoryBankInfer,
    PCLGaussianNoise,
    PCLRigidTransformNoise,
    PointCloudEncoder,
)

cfg = {
    "architecture": "PCL Encoder : Graph based encoder",
    "data root": "/home/lidar-workstation/CLORT/ArgoCL",
    "training data": ["train1", "train2", "train3", "train4"],
    "validation data": ["val"],
    "restore": None,
    "restore run path": None,
    "saved model": None,
    "batch size": 4,
    "temporal horizon": 4,
    "temporal overlap": 0,
    "batch shuffle": True,
    "num workers": 8,
    "prefetch factor": 8,
    "max objects": None,
    "static contrast": True,
    "hard condition": False,
    "local contrast": True,
    "separate tracks": False,
    "horizon localization": True,
    "sim type": 'dot',
    "temperature": 0.05,
    "number of tracks center": 5,
    "update factors for track center": [0.5, 0.4, 0.3, 0.2, 0.1],
    "augmentation probability": 0.5,
    "Normalization": "InstanceNorm2D and LayerNorm1D",
    "Normalization momemtum": 0.1,
    "lr factor" : 1.0,
    "n epochs": 30,
    "log iter": 20,
    "validation epoch": 3,
    "validation memory bank": "Infer", # Train
    "bbox aug": True,
    "n features": 256,
    }

def check_gradients(net):
    g = []
    for n, p in net.named_parameters():
        if p.grad is not None and 'weight' in n.split('.'):
            # print(f'{n} => {p.grad.norm()}')
            g.append(p.grad.detach().cpu().norm())
    return np.array(g)

def train(epoch, pc_enc, train_dl, optimizer, criterion, mem_bank, log_step=100, wb = True, model_device = 'cuda'):
    mem_bank.reset()

    pc_enc.train() # Enable training

    training_loss = []

    # Training loop
    for itr, (pcls, pcls_sz, _, _, bboxs, track_idxs, _, n_objs, n_tracks) in (t_bar := tqdm(enumerate(train_dl), total=len(train_dl))):
        optimizer.zero_grad()

        pcls = pcls.to(model_device)/100.
        # imgs = imgs.to(model_device)
        track_idxs = torch.from_numpy(track_idxs.astype(np.int32))
        bboxs = bboxs.to(model_device)/100.

        pc_e = pc_enc(pcls, pcls_sz, n_objs, bboxs)

        mem_bank.update(pc_e.detach().cpu(), track_idxs) # Update memory bank

        loss = criterion(pc_e, track_idxs, mem_bank.get_memory(), n_tracks)
        training_loss.append(loss.numpy(force=True).item())

        loss.backward()

        gradient = check_gradients(pc_enc)

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

def val(epoch, pc_enc, val_dl, criterion, mem_bank, log_step=100, wb = True, model_device = 'cuda'):
    mem_bank.reset()

    pc_enc.eval() # Enable training

    validation_loss = []

    # Validation loop
    with torch.no_grad():
        for itr, (pcls, pcls_sz, _, _, bboxs, track_idxs, _, n_objs, n_tracks) in (v_bar := tqdm(enumerate(val_dl), total=len(val_dl))):

            pcls = pcls.to(model_device)/100.
            # imgs = imgs.to(model_device)
            track_idxs = torch.from_numpy(track_idxs.astype(np.int32))
            bboxs = bboxs.to(model_device)/100.

            pc_e = pc_enc(pcls, pcls_sz, n_objs, bboxs)

            mem_bank.update(pc_e.detach().cpu(), track_idxs) # Update memory bank

            loss = criterion(pc_e, track_idxs, mem_bank.get_memory(), n_tracks)
            validation_loss.append(loss.numpy(force=True).item())

            if itr%(log_step) == log_step-1:
                v_bar.set_description(f'{epoch+1 = } and {itr+1 = } : Mean Validation loss : {np.mean(validation_loss[-log_step:])}')

                if wb:
                    wandb.log({'Validation Epoch': epoch+1, 'Validation Iteration': itr+1,
                                'Validation Loss': np.mean(validation_loss[-log_step:])
                              })

    return validation_loss

def main():
    # trunk-ignore(gitleaks/generic-api-key)
    wandb.login(key="724cb09699cc6beb92d458355f42ca29e3a67137")

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="CLORT",

        resume=False,

        # track hyperparameters and run metadata
        config=cfg
    )

    root: str = run.config["data root"]
    train_splits: List[str] = run.config["training data"]
    val_splits: List[str] = run.config["validation data"]
    restore: bool = run.config["restore"]
    restore_run_path: str | None = run.config["restore run path"]
    model_file_name: str | None = run.config["saved model"]
    batch_size: int = run.config["batch size"]
    batch_shuffle: bool = run.config["batch shuffle"]
    max_objects: int = run.config["max objects"]
    th: int = run.config["temporal horizon"]
    to: int = run.config["temporal overlap"]
    nw: int = run.config["num workers"]
    pf: int = run.config["prefetch factor"]
    lrf: float = run.config["lr factor"]
    model_device: torch.device | str = 'cuda'
    memory_device: torch.device | str = 'cpu'
    static_contrast: bool = run.config["static contrast"]
    hard_cond: bool = run.config["hard condition"]
    sep_tracks: bool = run.config["separate tracks"]
    hrz_loc: bool = run.config["horizon localization"]
    local_cont: bool = run.config["local contrast"]
    sim_type: str = run.config["sim type"]
    temp: float = run.config["temperature"]
    n_epochs: int = run.config['n epochs']
    Q: int = run.config["number of tracks center"]
    val_epoch: int = run.config["validation epoch"]
    infer_val_mb: bool = run.config["validation memory bank"] == "Infer"
    augmentation_prob: float = run.config["augmentation probability"]
    tcuf: List[int] = run.config["update factors for track center"]
    log_iter: int = run.config["log iter"]
    n_features: int = run.config["n features"]
    bbox_aug: bool = run.config["bbox aug"]

    train_dataset = ArgoCL(root,
                       temporal_horizon=th,
                       temporal_overlap=to,
                       max_objects=max_objects,
                       distance_threshold=None,
                       splits=train_splits, img_size=(224, 224),
                       point_cloud_size=[50, 100, 200, 500],
                       in_global_frame=True, pivot_to_first_frame=False,
                       image=False, pcl=True, bbox=bbox_aug,
                       vision_transform=None,
                       pcl_transform=RandomApply([PCLGaussianNoise(mean=0, std=1, tr_lim=2), # type: ignore
                                                PCLRigidTransformNoise(mean=0, std=np.pi/12, rot_lim=np.pi/12, trns_lim=2)], p = augmentation_prob))

    val_dataset = ArgoCL(root,
                        temporal_horizon=th,
                        temporal_overlap=to,
                        distance_threshold=None,
                        splits=val_splits, img_size=(224, 224),
                        point_cloud_size=[50, 100, 200, 500],
                        in_global_frame=True, pivot_to_first_frame=False,
                        image=False, pcl=True, bbox=bbox_aug)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=batch_shuffle,
                    collate_fn=ArgoCl_collate_fxn, num_workers=nw, prefetch_factor=pf)

    val_dl = DataLoader(val_dataset, batch_size, shuffle=False,
                    collate_fn=ArgoCl_collate_fxn, num_workers=nw, prefetch_factor=pf)

    print(f'{len(train_dl) = } \t {len(val_dl) = }')

    ## Initiate Encoders
    pc_enc = PointCloudEncoder(n_features, bbox_aug, norm_layer=nn.LayerNorm,
                               activation_layer=nn.SELU, offloading=False)
    pc_enc = pc_enc.to(model_device)

    # Initiate MemoryBanks
    mb = MemoryBank(train_dataset.n_tracks, n_features, Q,
                    alpha=torch.tensor(tcuf, dtype=torch.float32, device=memory_device),
                    device=memory_device)

    cl = ContrastiveLoss(temp=temp, local_contrast=local_cont,
                        separate_tracks=sep_tracks, static_contrast=static_contrast,
                        use_hard_condition=hard_cond, localize_to_horizon=hrz_loc, sim_type=sim_type)

    mb_infer = None

    if infer_val_mb:
        mb_infer = MemoryBankInfer(val_dataset.n_tracks, n_features, Q, t = min(2, Q), device=memory_device)
    else:
        mb_infer = MemoryBank(val_dataset.n_tracks, n_features, Q,
                            alpha=torch.tensor(tcuf, dtype=torch.float32, device=memory_device),
                            device=memory_device)

    cl_infer = ContrastiveLoss(temp=temp, local_contrast=True,
                               separate_tracks=sep_tracks, static_contrast=True,
                               use_hard_condition=hard_cond, localize_to_horizon=hrz_loc, sim_type=sim_type)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
                        params=[
                            {'params' : pc_enc.parameters(), 'lr': 1e-4*lrf, "weight_decay":1e-3*lrf}
                        ], lr = 1e-4*lrf, weight_decay=1e-3*lrf
                    )

    # Load model from file
    last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, last_epoch=last_epoch)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 5, 6, 7], gamma=0.9, last_epoch=last_epoch)

    # training_loss: List[float] = []
    # validation_loss: List[float] = []

    if restore:
        print(f'Loading model from file: {model_file_name = } \t {restore_run_path = }')
        ckpt = torch.load(run.restore(name=model_file_name, run_path=restore_run_path).name)
        print(f'{pc_enc.load_state_dict(ckpt["pc_enc"]) = }')
        print(f'{optimizer.load_state_dict(ckpt["optimizer"]) = }')
        print(f'{lr_scheduler.load_state_dict(ckpt["lr_scheduler"]) = }')
        print(f'{mb.load_state_dict(ckpt["mb"]) = }')
        print(f'{mb_infer.load_state_dict(ckpt["mb_infer"]) = }')

    last_epoch = lr_scheduler.last_epoch
    print(f'{last_epoch = } \t {n_epochs = }')

    print(f'{len(train_dl) = } \t {len(val_dl) = }')

    for epoch in range(last_epoch, n_epochs):
        model_fname = f'model_{epoch+1}.pth'
        model_path = os.path.join(run.dir, model_fname)

        _ = train(epoch, pc_enc, train_dl, optimizer,
                  cl, mb, log_step=log_iter, wb=True, model_device=model_device)

        ###################################################################################
        ### Validation loss
        if epoch%val_epoch == val_epoch-1:
            _ = val(epoch, pc_enc, val_dl, cl_infer,
                mb_infer, log_step=log_iter, wb = True, model_device=model_device)
        ### Validation loss
        ###################################################################################
        lr_scheduler.step() # Step Learning rate

        model_info = {
            'EPOCH': epoch,
            'pc_enc': pc_enc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'mb': mb.state_dict(),
            'mb_infer': mb_infer.state_dict()
        }

        torch.save(model_info, model_path)

        wandb.save(os.path.join(run.dir, "./model*.pth"))

    # Final Validation loop
    epoch = lr_scheduler.last_epoch
    _ = val(epoch-1, pc_enc, val_dl, cl_infer,
                   mb_infer, log_step=log_iter, wb = True, model_device=model_device)

    run.finish()

if __name__ == '__main__':
    main()
