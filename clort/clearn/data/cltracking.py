'''
Copyright (C) 2021  Shiavm Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from .argoversedataset import ArgoverseDataset


class ContrastiveLearningTracking(ArgoverseDataset):

    def __init__(self, root: str,
                 aug_per_track: int = 5,
                 img_dist_transforms: List[Callable] = None,
                 log_id: str = "",
                 max_tracklets: int = 10,
                 occlusion_thresh: float = 80,
                 lidar_points_thresh: int = 30,
                 image_size_threshold: int = 50,
                 n_img_view_thresh: int = 1,
                 n_img_view_aug: int = 7,
                 aug_transforms: List[Callable] = None,
                 central_crop: bool = True,
                 img_tr_ww: float = 0.7,
                 discard_invalid_dfs: bool = True,
                 dataframe_augmentation: bool = True,
                 augmentation_frames: int = 4,
                 img_reshape: Tuple[int, int] = (200, 200),
                 ids_repeat: int = 200) -> None:
        super().__init__(root, log_id, max_tracklets, occlusion_thresh, lidar_points_thresh, image_size_threshold,
                         n_img_view_thresh, n_img_view_aug, aug_transforms, central_crop, img_tr_ww,
                         discard_invalid_dfs, dataframe_augmentation, augmentation_frames, img_reshape)

        self.aug_per_track = aug_per_track
        self.img_dist_transforms = img_dist_transforms
        self.n = None
        self.track_ids = None
        self.valid_track_ids = None
        self.n_valid = None
        self.ids_repeat = ids_repeat
        self.valid_track_ids_int_map: Dict[str, int] = {}

    def dataset_init(self, log_idx: int, frames_upto:int = None) -> None:
        super().dataset_init(log_idx)
        self.pre_load_tracking_frames(frames_upto)
        self.n = self.count_examples()
        self.track_ids = list(self.tracking_queue.keys())
        self.valid_track_ids = []
        self.discard_invalid_tracks()
        self.valid_track_ids_int_map = {track_id: i for i, track_id in enumerate(self.valid_track_ids)}
        self.valid_track_ids = self.valid_track_ids*self.ids_repeat
        self.n_valid = len(self.valid_track_ids)

    def repeat_tracks(self, ids_repeat: int):
        self.valid_track_ids = self.valid_track_ids*ids_repeat
        self.n_valid = len(self.valid_track_ids)

    def __getitem__(self, index: Any) -> Any:
        # Get the track ids
        img_data, pcd_data = [], []

        track_id = self.valid_track_ids[index]
        rand_track_ids = np.arange(len(self.tracking_queue[track_id]))

        if self.aug_per_track < len(rand_track_ids):
            rand_track_ids = np.random.choice(rand_track_ids, self.aug_per_track, replace=False)

        for id in rand_track_ids:
            df = self.tracking_queue[track_id][id]
            imgs_ = torch.cat(tuple(df.img_data), dim=0).unsqueeze(dim=0)
            pcd = df.get_lidar()
            idx = np.random.choice(len(pcd), self.lidar_points_thresh, replace=False)
            pcd = torch.as_tensor(pcd[idx], dtype=torch.float32).unsqueeze(dim=0)
            img_data += [imgs_]
            pcd_data += [pcd]

        img_data = torch.cat(tuple(img_data), dim=0)
        pcd_data = torch.cat(tuple(pcd_data), dim=0)
        pcd_data = pcd_data.transpose(2, 1)

        track_id = torch.as_tensor([self.valid_track_ids_int_map[track_id]]*len(rand_track_ids), dtype=torch.int32)
        return img_data, pcd_data, track_id

    def __len__(self) -> int:
        return self.n_valid

    def discard_invalid_tracks(self) -> None:
        valid_track_ids = []
        for key, val in self.tracking_queue.items():
            if val.__len__() >= self.aug_per_track:
                valid_track_ids += [key]

        self.valid_track_ids = valid_track_ids
