import os
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ArgoCL(Dataset):

    def __init__(self, root: str,
                 temporal_horizon: int = 8,
                 temporal_overlap: int = 4,
                 # trunk-ignore(ruff/B006)
                 splits: List[str] = ['train1', 'train2', 'train3', 'train4'],
                 image : bool = True, pcl : bool = True, bbox : bool = True,
                 in_global_frame : bool = True,
                 distance_threshold: Tuple[float, float] = (0, 50),
                 # trunk-ignore(ruff/B006)
                 point_batch_quantization : List[int] = [0, 150, 500, 1000, 1500, 2000],
                 point_cloud_size: int = 2048) -> None:
        super().__init__()

        assert(temporal_horizon > temporal_overlap and temporal_horizon != temporal_overlap)

        self.root = root # Dataset Root
        self.th = temporal_horizon # Temporal Horizon
        self.to = temporal_overlap # Temporal Overlap
        self.im, self.pc, self.bx, self.glc = image, pcl, bbox, in_global_frame
        self.dt = distance_threshold
        self.pbq = point_batch_quantization
        self.pcs = point_cloud_size

        # Get all log files
        self.log_files: List[Union[h5py.File, h5py.Group]] = []

        for split in splits:
            self.log_files += [h5py.File(f'{self.root}/{split}/{log}', mode='r', track_order=True) for log in os.listdir(f'{self.root}/{split}')]

        for i, log_file in enumerate(self.log_files):
            self.log_files[i] = log_file[list(log_file.keys())[0]] # type: ignore

        # All tracks
        self.tracks : Dict[str, Dict[str, int]] = {}
        self.n_tracks = 0
        for log in self.log_files:
            tracks_in_log = self.list_all_tracks_in_log(log)
            self.tracks.update({
                log.name : {track_id : i + self.n_tracks for i, track_id in enumerate(tracks_in_log)} # type: ignore
                })
            self.n_tracks += len(tracks_in_log)

        self.obj_cls : List[str] = []

        # All frames and corresponding logs
        self.frames : Dict[str, List[str]] = {}
        self.sorted_logs = []
        for log in self.log_files:
            self.frames.update({log.name : list(log.keys())}) # type: ignore
            self.sorted_logs.append(log.name)

        self.n = [len(self.frames[log])//(self.th - self.to) for log in self.sorted_logs]
        self.N = np.sum(self.n)

    def __del__(self):
        for f in self.log_files:
            f.close()
        print("All files are closed and destructor called.")

    @staticmethod
    def list_all_tracks_in_log(log: h5py.Group) -> List[str]:

        tracks : List[str] = []

        log_id = log.name

        for frame in log[log_id].keys(): # type: ignore
            for in_frame_key in log[f'{log_id}/{frame}'].keys(): # type: ignore
                if not in_frame_key.startswith('det'):
                    continue

                track_id = np.asanyarray(log[f'{log_id}/{frame}/{in_frame_key}/track_id'], dtype=str).item()

                if track_id not in tracks:
                    tracks.append(track_id)

        return tracks

    def get_reduced_index(self, index:int) -> Tuple[int, int]:
        items = -1

        for i in range(len(self.n)):
            items += self.n[i]
            if index <= items:
                return i, index-items+self.n[i]

        raise KeyError(f'Index ({index}) out of bound')

    def extract_frame_info(self, log_id: str, frame_log: h5py.Group,
                           im : bool = True, pc : bool = True,
                           bx : bool = True, glc : bool = True) -> List[Dict[str, Union[np.ndarray, int]]]:
        frame_data : List[Dict[str, Union[np.ndarray, int]]] = []

        R, t = np.eye(3), np.zeros((3, ))
        if glc:
            tr = np.asanyarray(frame_log['local_to_global_transform'], dtype=np.float32)
            R, t = tr[:, :3], tr[:, 3]

        for det in frame_log.keys():
            if not det.startswith('det'):
                continue

            bbox = np.asanyarray(frame_log[f'{det}/bbox'], dtype=np.float32)
            center_distance = np.linalg.norm(np.mean(bbox, axis=0))
            if not (center_distance > self.dt[0] and center_distance < self.dt[1]):
                continue

            det_data : Dict[str, Union[np.ndarray, int]] = {
                'pcl' : np.empty((0, 3)),
                'imgs' : np.empty((0, 250, 250)),
                'bbox' : np.empty((0, 3)),
                'track_idx' : -1,
                'cls_idx' : -1,
            }

            if pc:
                point_cloud = np.asanyarray(frame_log[f'{det}/pcl'], dtype=np.float32)
                if glc:
                    point_cloud = point_cloud @ R + t
                det_data.update({'pcl' : point_cloud})

            if bx:
                # bbox = np.asanyarray(frame_log[f'{det}/bbox'], dtype=np.float32)
                if glc:
                    bbox = bbox @ R + t
                det_data.update({'bbox' : bbox})

            if im:
                img_data = []
                for im_key in frame_log[det].keys(): # type: ignore
                    if not im_key.startswith('img'):
                        continue
                    img_data.append(np.asanyarray(frame_log[f'{det}/{im_key}'], dtype=np.uint8))

                det_data.update({'imgs' : np.concatenate(img_data, axis=-1).transpose((2, 0, 1))})

            track_id = np.asanyarray(frame_log[f'{det}/track_id'], dtype=str).item()
            det_data.update(
                {'track_idx' : self.tracks[log_id][track_id]}
                         )

            obj_cls = np.asanyarray(frame_log[f'{det}/cls'], dtype=str).item()
            cls_idx = -1
            try:
                cls_idx = self.obj_cls.index(obj_cls)
            except ValueError:
                self.obj_cls.append(obj_cls)
                cls_idx = len(self.obj_cls)-1

            det_data.update(
                {'cls_idx' : cls_idx}
            )

            frame_data.append(det_data)

        return frame_data

    def __getitem__(self, index) -> Any:
        i, index = self.get_reduced_index(index)
        log_id : str = self.log_files[i].name # type: ignore

        # Point-Cloud Data
        pcls : Union[torch.Tensor, np.ndarray, List[np.ndarray]] = []
        pcls_sz : List[int] = [] # Point-Cloud sizes (for slicing)

        # Multi-View Image Data
        imgs : Union[torch.Tensor, np.ndarray, List[np.ndarray]] = []
        imgs_sz : List[int] = [] # MV Image sizes (for slicing)

        bboxs : Union[torch.Tensor, np.ndarray, List[np.ndarray]] = [] # Bounding boxes, fixed size of 8 corner bbox
        track_idxs : List[int] = [] # Track id list as integers one per detection
        cls_idxs : List[int] = [] # Class id list as integer, one per detection

        n = len(self.frames[self.sorted_logs[i]])
        for frame in self.frames[log_id][index:min(index+self.th, n)]:
            frame_data = self.extract_frame_info(
                log_id,
                self.log_files[i][frame], # type: ignore
                self.im, self.pc, self.bx, self.glc)

            for det in frame_data:
                if self.pc:
                    pcls.append(det['pcl']) # type: ignore
                    pcls_sz.append(det['pcl'].shape[0]) # type: ignore

                if self.im:
                    imgs.append(det['imgs'])  # type: ignore
                    imgs_sz.append(det['imgs'].shape[0]//3) # type: ignore

                if self.bx:
                    bboxs.append(det['bbox']) # type: ignore

                track_idxs.append(det['track_idx']) # type: ignore
                cls_idxs.append(det['cls_idx']) # type: ignore

        # Aggregate informations from all frames in temporal horizon
        if self.pc:
            pcls = torch.from_numpy(np.concatenate(pcls, axis=0)) # Concatenate on points dimension

        if self.im:
            imgs = torch.from_numpy(np.concatenate(imgs, axis=0).astype(np.float32)/255.) # Concatenate images on channel dimension # [_, 250, 250]
            imgs = torch.cat(
                imgs.unsqueeze(dim=0).split(split_size=3, dim=1)   # ([1, 3, 250, 250]...)
                , dim=0)                                           # [_//3, 3, 250, 250] # dimension 1 is number of views which
                                                                                                            # can be extracted with $imgs_sz$ split

        if self.bx:
            bboxs = torch.cat(
                torch.from_numpy(np.concatenate(bboxs, axis=0)).unsqueeze(dim=0).split(8, dim=1),
                dim=0) # [num_dets, 8, 3] # Concatenation on points dimension

        pcls_sz = np.array(pcls_sz, dtype=np.uint16)
        imgs_sz = np.array(imgs_sz, dtype=np.uint8)
        track_idxs = np.array(track_idxs, dtype=np.uint16)
        cls_idxs = np.array(cls_idxs, dtype=np.uint8)

        return pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs

    def __len__(self) -> int:
        return self.N

def ArgoCl_collate_fxn(batch:torch.tensor):

    pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs = [], [], [], [], [], [], []

    for sample in batch:
        pcls.append(sample[0])
        pcls_sz.append(sample[1])
        imgs.append(sample[2])
        imgs_sz.append(sample[3])
        bboxs.append(sample[4])
        track_idxs.append(sample[5])
        cls_idxs.append(sample[6])

    pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs = \
        torch.cat(pcls, dim=0), np.concatenate(pcls_sz, axis=0), torch.cat(imgs, dim=0), \
            np.concatenate(imgs_sz, axis=0), torch.cat(bboxs, dim=0), np.concatenate(track_idxs, axis=0), \
                np.concatenate(cls_idxs, axis=0)

    return pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs
