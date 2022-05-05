import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from clort.clearn.data import ContrastiveLearningTracking
from clort.clearn.models import FeatureMixer, PointCloudEncoder, VisualEncoder
from mzLosses.contrastive import SoftNearestNeighbourLoss
from torch.utils.data import DataLoader


def compute_loss(dl, vis_model, pcl_model, feature_mixer, criterion):
    total_loss = np.zeros(4, dtype=np.float32)
    n = len(dl)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for imgs, pcls, track_ids in dl:
            b, n_view, C, H, W = imgs.size()
            _, _, d, n = pcls.size()

            imgs, pcls, track_ids = imgs.view(b*n_view, C, H, W), pcls.view(b*n_view, d, n), track_ids.flatten()
            imgs, pcls, track_ids = imgs.to(device), pcls.to(device), track_ids.to(device)

            vis_enc = vis_model(imgs)
            pcls_enc = pcl_model(pcls)
            final_enc = feature_mixer(vis_enc, pcls_enc)

            vis_loss = criterion(vis_enc, track_ids)
            pcl_loss = criterion(pcls_enc, track_ids)
            enc_loss = criterion(final_enc, track_ids)

            loss = w_vis * vis_loss + w_pcl * pcl_loss + w_enc * enc_loss

            total_loss += np.array([loss.item(), vis_loss.item(), pcl_loss.item(), enc_loss.item()], dtype=np.float32)

    return total_loss/n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root directory path to data.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batchsize to be applied.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of times to iterate complete data.')
    parser.add_argument('--n_itr_logs', type=int, default=10, help='Number of times to iterate complete log id.')
    parser.add_argument('--log_id', type=int, default=0, help='Log id to be used for training.')
    parser.add_argument('--n_frames', type=int, default=10, help='Number of track frames to preload.')
    parser.add_argument('--n_augs', type=int, default=5, help='Number of augmentations to be loaded from each track.')
    parser.add_argument('--vis_loss_w', type=float, default=1., help='Weight to put on vision contrastive loss.')
    parser.add_argument('--pcl_loss_w', type=float, default=1., help='Weight to put on geometric contrastive loss.')
    parser.add_argument('--enc_loss_w', type=float, default=1., help='Weight to put on feature encoding contrastive loss.')
    parser.add_argument('--vis_lr', type=float, default=0.0007, help='Visual model learning rate.')
    parser.add_argument('--pcl_lr', type=float, default=0.0007, help='Geometric model learning rate.')
    parser.add_argument('--enc_lr', type=float, default=0.0001, help='Visual model learning rate.')
    parser.add_argument('--itr_log_ln', type=int, default=5, help='Number of time logging in a single epoch.')
    parser.add_argument('--epoch_log_ln', type=int, default=20, help='Number of time logging in a single epoch.')
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.getenv('HOME'), 'clearn_models'))
    parser.add_argument('--preload_model_path', type=str, default='')
    parser.add_argument('--cuda', action='store_true', help='If given use GPU if cuda is available.')

    args = parser.parse_args()

    root_dir: str = args.root # Root directory
    batch_size: int = args.batch_size # Batch size
    n_epochs: int = args.n_epochs # Number of epochs
    n_itr_logs: int = args.n_itr_logs # Number of times complete log list to interate
    log_id: int = args.log_id # Log id
    n_frames: int = args.n_frames # Number of preloaded frames
    n_augs: int = args.n_augs # Number of augmentations to be loaded per track

    print(f'Root directory : {root_dir} \nBatch size : {batch_size} \nn_epochs : {n_epochs} \nlog_id : {log_id} \nn_frames : {n_frames} \nn_augs : {n_augs}')

    w_vis, w_pcl, w_enc = np.float32(args.vis_loss_w), np.float32(args.pcl_loss_w), np.float32(args.enc_loss_w)
    w_s = (w_vis + w_pcl + w_enc)
    w_vis, w_pcl, w_enc = w_vis/w_s, w_pcl/w_s, w_enc/w_s

    print(f'Loss weights : Visual, Geometric, Final enc : {[w_vis, w_pcl, w_enc]}')

    vis_lr, pcl_lr, enc_lr = args.vis_lr, args.pcl_lr, args.enc_lr

    print(f'Learning rates : Visual, Geometric, feature mixer : {[vis_lr, pcl_lr, enc_lr]}')

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print(f'Results directory : {results_dir}')

    dataset = ContrastiveLearningTracking(root_dir,
                                          aug_per_track=n_augs,
                                          occlusion_thresh = 30.,
                                          central_crop=True,
                                          img_tr_ww = (0.9, 0.9),
                                          image_size_threshold=100,
                                          img_reshape = (256, 256),
                                          ids_repeat=batch_size//10)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Create models
    vis_model = VisualEncoder()
    pcl_model = PointCloudEncoder(10)
    feature_mixer = FeatureMixer(vis_size=512, pcl_size=60, embed_dim=128)

    # Move models to device
    vis_model.to(device)
    pcl_model.to(device)
    feature_mixer.to(device)

    if args.preload_model_path != '':
        print(f'Preloading model from path : {args.preload_model_path}')
        vis_model.load_state_dict(torch.load(os.path.join(args.preload_model_path, 'vis_model.pth')))
        pcl_model.load_state_dict(torch.load(os.path.join(args.preload_model_path, 'pcl_model.pth')))
        feature_mixer.load_state_dict(torch.load(os.path.join(args.preload_model_path, 'feature_mixer.pth')))
    else:
        print('No preload instruction provided.')

    # Define criterion and optimizers
    criterion = SoftNearestNeighbourLoss()
    vis_optim = optim.AdamW(vis_model.parameters(), lr=vis_lr, weight_decay=0.0001)
    pcl_optim = optim.AdamW(pcl_model.parameters(), lr=pcl_lr, weight_decay=0.0001)
    feature_optim = optim.AdamW(feature_mixer.parameters(), lr=enc_lr, weight_decay=0.0001)

    for _ in range(n_itr_logs):
        for log_id in range(len(dataset.log_list)):
            print(f'Training on log_id : {dataset.log_list[log_id]}')

            # Preload n_frames
            dataset.dataset_init(log_id, n_frames)

            # Training dataloader
            train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            n_itrs = len(train_dl)
            itr_log_ln = n_itrs//args.itr_log_ln
            epoch_log_ln = 1

            print(f'Num iterations : {n_itrs} \t Iteration logging@ : {itr_log_ln} \t Epoch logging@ : {epoch_log_ln}')

            # Run the training loop
            rll, tll = [], []

            for epoch in range(1, n_epochs+1):
                running_loss = np.zeros(4, dtype=np.float32)
                for i, (imgs, pcls, track_ids) in enumerate(train_dl):
                    itr = i+1
                    b, n_view, C, H, W = imgs.size()
                    _, _, d, n = pcls.size()

                    imgs, pcls, track_ids = imgs.view(b*n_view, C, H, W), pcls.view(b*n_view, d, n), track_ids.flatten()
                    imgs, pcls, track_ids = imgs.to(device), pcls.to(device), track_ids.to(device)

                    vis_optim.zero_grad()
                    pcl_optim.zero_grad()
                    feature_optim.zero_grad()

                    vis_enc = vis_model(imgs)
                    pcls_enc = pcl_model(pcls)
                    final_enc = feature_mixer(vis_enc, pcls_enc)

                    vis_loss = criterion(vis_enc, track_ids)
                    pcl_loss = criterion(pcls_enc, track_ids)
                    enc_loss = criterion(final_enc, track_ids)

                    loss = w_vis * vis_loss + w_pcl * pcl_loss + w_enc * enc_loss


                    loss.backward()

                    vis_optim.step()
                    pcl_optim.step()
                    feature_optim.step()

                    running_loss += np.array([loss.item(), vis_loss.item(), pcl_loss.item(), enc_loss.item()], dtype=np.float32)

                    if itr%itr_log_ln == 0:
                        rll += [[epoch, itr] + list(running_loss/itr_log_ln)]
                        print(f'Epoch : {epoch} \t Iteration : {itr} \t Running loss : {running_loss/itr_log_ln}')
                        running_loss = np.zeros(4, dtype=np.float32)

                if epoch%epoch_log_ln == 0:
                    total_loss = compute_loss(train_dl, vis_model, pcl_model, feature_mixer, criterion)
                    tll += [list(total_loss)]
                    print(f'Epoch : {epoch} \t Total loss : {total_loss}')

                # Save model state
                model_state_path = os.path.join(results_dir, f'model_state_at_log_itr_{_}_log_id_{log_id}_epoch_{epoch}')
                os.makedirs(model_state_path, exist_ok=True)
                torch.save(vis_model.state_dict(), os.path.join(model_state_path, 'vis_model.pth'))
                torch.save(pcl_model.state_dict(), os.path.join(model_state_path, 'pcl_model.pth'))
                torch.save(feature_mixer.state_dict(), os.path.join(model_state_path, 'feature_mixer.pth'))

                # Save rll and tll
                rll_df_path = os.path.join(results_dir, f'rll_at_log_itr_{_}_log_id_{log_id}_epoch_{epoch}.csv')
                rll_df = pd.DataFrame(rll, columns=['Epoch', 'Iteration', 'TRL', 'VRL', 'PRL', 'FRL'])
                rll_df.to_csv(rll_df_path, index=False)

                tll_df_path = os.path.join(results_dir, f'tll_at_log_itr_{_}_log_id_{log_id}_epoch_{epoch}.csv')
                tll_df = pd.DataFrame(tll, columns=['TRL', 'VRL', 'PRL', 'FRL'])
                tll_df.to_csv(tll_df_path, index=False)

        # Save model state
        model_state_path = os.path.join(results_dir, f'model_state_at_logs_itr_{_}')
        os.makedirs(model_state_path, exist_ok=True)
        torch.save(vis_model.state_dict(), os.path.join(model_state_path, 'vis_model.pth'))
        torch.save(pcl_model.state_dict(), os.path.join(model_state_path, 'pcl_model.pth'))
        torch.save(feature_mixer.state_dict(), os.path.join(model_state_path, 'feature_mixer.pth'))
