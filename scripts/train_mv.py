# import argparse
import os
from typing import List

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from clort import ArgoCL, ArgoCl_collate_fxn
from clort.model import ContrastiveLoss, MemoryBank, MemoryBankInfer, MultiViewEncoder

torch.autograd.set_detect_anomaly(True)


def train(root: str = "../../../datasets/argoverse-tracking/argov1_proc",
          splits: List[str] = ['train4'],
          model_save_dir: str = '~/.tmp/CLORT/',
          load_saved_model: str | None = None,
          batch_size: int = 1,
          th: int = 4, to: int = 2,
          nw: int = 0,
          model_device: torch.device | str = 'cuda',
          memory_device: torch.device | str = 'cpu',
          static_contrast: bool = False,
          n_epochs: int = 30):

    train_dataset = ArgoCL(root,
                       temporal_horizon=th,
                       temporal_overlap=to,
                       distance_threshold=(0, 100),
                       splits=splits, img_size=(224, 224),
                       point_cloud_size=[20, 50, 100, 250, 500, 1000, 1500],
                       in_global_frame=True, pivot_to_first_frame=True,
                       image=True, pcl=True, bbox=True)

    val_dataset = ArgoCL(root,
                       temporal_horizon=1,
                       temporal_overlap=0,
                       distance_threshold=(0, 100),
                       splits=['val'], img_size=(224, 224),
                       point_cloud_size=[20, 50, 100, 250, 500, 1000, 1500],
                       in_global_frame=True, pivot_to_first_frame=True,
                       image=True, pcl=True, bbox=True)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=False,
                      collate_fn=ArgoCl_collate_fxn, num_workers=nw)

    val_dl = DataLoader(val_dataset, 1, shuffle=False,
                    collate_fn=ArgoCl_collate_fxn, num_workers=nw)

    n_features = 256

    mv_enc = MultiViewEncoder(out_dim=n_features)
    mv_enc = mv_enc.to(model_device)

    mb = MemoryBank(train_dataset.n_tracks, n_features, 5,
                    alpha=torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32, device=memory_device),
                    device=memory_device)

    cl = ContrastiveLoss(temp=0.05, static_contrast=static_contrast)

    optimizer = torch.optim.AdamW(
        params=[
            {'mv:base' : mv_enc.sv_enc1.parameters(), 'lr': 1e-6},
            {'mv:linear2': mv_enc.sv_enc2.parameters(), 'lr': 1e-4},
            {'mv:linear3': mv_enc.sv_enc3.parameters(), 'lr': 1e-4},
            {'mv:gat': mv_enc.gat.parameters(), 'lr': 1e-4},
            {'mv:projection': mv_enc.projection_head.parameters(), 'lr': 1e-4}
            ], lr = 1e-4, weight_decay=1e-3
    )

    # Load model from file
    last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=last_epoch)

    training_loss: List[float] = []
    validation_loss: List[float] = []

    if load_saved_model is not None:
        ckpt = torch.load(load_saved_model)
        mv_enc.load_state_dict(ckpt['mv_enc'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        training_loss = ckpt['train_loss']
        validation_loss = ckpt['val_loss']

    for epoch in range(last_epoch+1, n_epochs):
        model_path = os.path.join(model_save_dir, f'model_{epoch}.pth')

        # Training loop
        for itr, (_, _, imgs, imgs_sz, _, track_idxs, _, _, _) in tqdm(enumerate(train_dl)):
            mv_enc.train() # Enable training
            optimizer.zero_grad()

            # pcls = pcls.to(model_device)
            imgs = imgs.to(model_device)
            track_idxs = torch.from_numpy(track_idxs.astype(np.int32))
            # bboxs = bboxs.to(model_device)

            mv_e = mv_enc(imgs, imgs_sz)

            loss = cl(mv_e, track_idxs, mb.get_memory())
            training_loss.append(loss.numpy(force=True).item())

            loss.backward()

            optimizer.step()

            if itr%10 == 9:
                print(f'{epoch = } and {itr = } : Mean Training loss : {np.mean(training_loss[-10:])}')

        ###################################################################################
                ### Validation loss
                mb_infer = MemoryBankInfer(val_dataset.n_tracks, n_features, 5, 3, 'cpu')

                cl_infer = ContrastiveLoss(static_contrast=False)

                val_loss = 0.0

                mv_enc.eval() # Enable inference

                with torch.no_grad():
                    for _, (_, _, imgs, imgs_sz, _, track_idxs, _, _, _) in tqdm(enumerate(val_dl)):
                        # pcls = pcls.to(model_device)
                        imgs = imgs.to(model_device)
                        track_idxs = torch.from_numpy(track_idxs.astype(np.int32))
                        # bboxs = bboxs.to(model_device)

                        mv_e : torch.Tensor = mv_enc(imgs, imgs_sz)
                        loss : torch.Tensor = cl_infer(mv_e, track_idxs, mb_infer.get_memory())

                        val_loss += loss.detach().cpu().item()

                        mb_infer.update(mv_e.detach().cpu(), track_idxs)

                val_loss /= len(val_dl)
                validation_loss.append(val_loss)

                print(f'{epoch = } and {itr = } : Mean Validation loss : {val_loss = }')

                wandb.log({'epoch': epoch+1, 'itr': itr+1,
                           'training_loss': np.mean(training_loss[-10:]),
                           'val_loss': val_loss})
                ### Validation loss
        ###################################################################################
        lr_scheduler.step() # Step Learning rate

        model_info = {
            'EPOCH': epoch,
            'mv_enc': mv_enc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss
        }

        torch.save(model_info, model_path)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('root', )

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="clort",

        # track hyperparameters and run metadata
        config={
        "architecture": "Multi_View Encoder",
        "dataset": "ArgoCL : Train1",
        "epochs": 30,
        }
    )
