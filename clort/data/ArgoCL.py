import os
from typing import Any, Dict, List, Tuple, Type

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ArgoCL(Dataset):

    def __init__(self, root: str,
                 temporal_horizon: int = 8,
                 temporal_overlap: int = 4,
                 max_objects: int | None = None,
                 # trunk-ignore(ruff/B006)
                 splits: List[str] = ['train1', 'train2', 'train3', 'train4'],
                 image : bool = True, pcl : bool = True, bbox : bool = True,
                 in_global_frame : bool = True,
                 distance_threshold: Tuple[float, float] | None = None,
                 img_size : Tuple[int, int] = (224, 224),
                 point_cloud_size: List[int] | int | None = None,
                 pivot_to_first_frame: bool = False,
                 vision_transform: Type[nn.Module] | None = None,
                 pcl_transform: Type[nn.Module] | None = None) -> None:
        super().__init__()

        assert(temporal_horizon > temporal_overlap and temporal_horizon != temporal_overlap)

        self.root = root # Dataset Root
        self.th = temporal_horizon # Temporal Horizon
        self.to = temporal_overlap # Temporal Overlap

        assert((self.th - self.to) > 0)

        self.max_objects = max_objects
        self.im, self.pc, self.bx, self.glc = image, pcl, bbox, in_global_frame
        self.dt = distance_threshold
        self.img_size = img_size
        self.pcs = point_cloud_size
        self.pvff = pivot_to_first_frame
        self.vt = vision_transform
        self.pcl_tr = pcl_transform

        # Get all log files
        self.log_files: List[h5py.File | h5py.Group] = []

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

        self.n = [int(np.ceil(len(self.frames[log])/(self.th - self.to))) for log in self.sorted_logs]
        self.N = np.sum(self.n)

    # def __del__(self):
    #     for f in self.log_files:
    #         f.parent.close()
    #     print("All files are closed and destructor called.")

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
                return i, (index-items+self.n[i]-1)*(self.th-self.to)

        raise KeyError(f'Index ({index}) out of bound')

    def sample_point_cloud(self, pc: np.ndarray) -> np.ndarray:

        if self.pcs is None:
            return pc

        sz = len(pc)
        if isinstance(self.pcs, int):
            sz = self.pcs
        else:
            curr_sz = sz
            for i in sorted(self.pcs):
                if i < sz:
                    curr_sz = i
                else:
                    break

            sz = curr_sz

        idx = np.random.randint(0, len(pc), size=sz, dtype=int)

        return pc[idx, :]

    def extract_frame_info(self, log_id: str, frame_log: h5py.Group,
                           im : bool = True, pc : bool = True,
                           bx : bool = True, glc : bool = True) -> List[Dict[str, np.ndarray | int]]:
        frame_data : List[Dict[str, np.ndarray | int]] = []

        R, t = np.eye(3), np.zeros((3, ))
        if glc:
            tr = np.asanyarray(frame_log['local_to_global_transform'], dtype=np.float32)
            R, t = tr[:, :3], tr[:, 3]

        n_det = len(list(frame_log.keys()))
        idxs_ = np.arange(n_det)

        if self.max_objects is not None and n_det > self.max_objects:
            idxs_ = np.random.randint(0, n_det, size=self.max_objects)

        dets = list(frame_log.keys())
        for det_idx in idxs_:
            det = dets[det_idx]
            if not det.startswith('det'):
                continue

            bbox = np.asanyarray(frame_log[f'{det}/bbox'], dtype=np.float32) if self.dt is not None or self.bx else None

            if self.dt is not None:
                center_distance = np.linalg.norm(np.mean(bbox, axis=0))
                if not (center_distance > self.dt[0] and center_distance < self.dt[1]):
                    continue

            det_data : Dict[str, np.ndarray | int] = {
                'pcl' : np.empty((0, 3)),
                'imgs' : np.empty((0, 250, 250)),
                'bbox' : np.empty((0, 3)),
                'track_idx' : -1,
                'cls_idx' : -1,
            }

            if pc:
                point_cloud = self.sample_point_cloud(np.asanyarray(frame_log[f'{det}/pcl'], dtype=np.float32))
                if glc:
                    point_cloud = point_cloud @ R + t
                if self.pcl_tr is not None:
                    point_cloud = self.pcl_tr(point_cloud)
                det_data.update({'pcl' : point_cloud})

            if bx:
                # bbox = np.asanyarray(frame_log[f'{det}/bbox'], dtype=np.float32)
                if glc:
                    bbox = bbox @ R + t # type: ignore
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
        pcls : torch.Tensor | np.ndarray | List[np.ndarray] = []
        pcls_sz : List[int] | np.ndarray = [] # Point-Cloud sizes (for slicing)

        # Multi-View Image Data
        imgs : torch.Tensor | np.ndarray | List[np.ndarray] = []
        imgs_sz : List[int] | np.ndarray = [] # MV Image sizes (for slicing)

        bboxs : torch.Tensor | np.ndarray | List[np.ndarray] = [] # Bounding boxes, fixed size of 8 corner bbox
        track_idxs : List[int] | np.ndarray = [] # Track id list as integers one per detection
        cls_idxs : List[int] | np.ndarray = [] # Class id list as integer, one per detection

        frame_sz : List[int] | np.ndarray = []

        n = len(self.frames[self.sorted_logs[i]])

        pivot: np.ndarray | None = None

        # print(f'{index, min(index+self.th, n) = }')

        for frame in self.frames[log_id][index:min(index+self.th, n)]:
            frame_data = self.extract_frame_info(
                log_id,
                self.log_files[i][frame], # type: ignore
                self.im, self.pc, self.bx, self.glc)

            frame_sz.append(len(frame_data))

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

            if self.pvff and self.pc and pivot is None:
                pivot = np.concatenate(pcls, axis=0).mean(axis=0, keepdims=True)

        # Aggregate informations from all frames in temporal horizon
        if self.pc:
            pcls = torch.from_numpy(np.concatenate(pcls, axis=0) - (pivot if pivot is not None else 0)) # Concatenate on points dimension

        if self.im:
            imgs = torch.from_numpy(np.concatenate(imgs, axis=0).astype(np.float32)/255.) # Concatenate images on channel dimension # [_, 250, 250]
            imgs = imgs.view(-1, 3, imgs.shape[-2], imgs.shape[-1])
            # torch.cat(
            #     imgs.unsqueeze(dim=0).split(split_size=3, dim=1)   # ([1, 3, 250, 250]...)
            #     , dim=0)                                           # [_//3, 3, 250, 250] # dimension 1 is number of views which
            #                                                                                                 # can be extracted with $imgs_sz$ split
            imgs = F.interpolate(imgs, self.img_size, mode='bilinear')
            if self.vt is not None:
                imgs = self.vt(imgs) # type: ignore

        if self.bx:
            # bboxs = torch.cat(
            #     torch.from_numpy(np.concatenate(bboxs, axis=0)).unsqueeze(dim=0).split(8, dim=1),
            #     dim=0) # [num_dets, 8, 3] # Concatenation on points dimension
            bboxs = torch.from_numpy(np.concatenate(bboxs, axis=0) - (pivot if pivot is not None else 0)).view(-1, 8, 3)

        pcls_sz = np.array(pcls_sz, dtype=np.uint16) if len(pcls_sz) != 0 else []
        imgs_sz = np.array(imgs_sz, dtype=np.uint8) if len(imgs_sz) != 0 else []
        track_idxs = np.array(track_idxs, dtype=np.uint16)
        cls_idxs = np.array(cls_idxs, dtype=np.uint8)
        frame_sz = np.array(frame_sz, dtype=np.uint16)

        return pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs, frame_sz

    def __len__(self) -> int:
        return self.N

def ArgoCl_collate_fxn(batch:Any):

    pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs, frame_sz, sample_sz = \
        [], [], [], [], [], [], [], [], []

    for sample in batch:
        if isinstance(sample[0], torch.Tensor):
            pcls.append(sample[0])
        if isinstance(sample[1], np.ndarray):
            pcls_sz.append(sample[1])
        if isinstance(sample[2], torch.Tensor):
            imgs.append(sample[2])
        if isinstance(sample[3], np.ndarray):
            imgs_sz.append(sample[3])
        if isinstance(sample[4], torch.Tensor):
            bboxs.append(sample[4])

        track_idxs.append(sample[5])
        cls_idxs.append(sample[6])
        frame_sz.append(sample[7])
        sample_sz.append(len(sample[7]))

    pcls = torch.cat(pcls, dim=0) if len(pcls) != 0 else []
    pcls_sz = np.concatenate(pcls_sz, axis=0) if len(pcls_sz) != 0 else []

    imgs = torch.cat(imgs, dim=0) if len(imgs) != 0 else []
    imgs_sz = np.concatenate(imgs_sz, axis=0) if len(imgs_sz) != 0 else []

    bboxs = torch.cat(bboxs, dim=0) if len(bboxs) != 0 else []

    track_idxs = np.concatenate(track_idxs, axis=0)
    cls_idxs = np.concatenate(cls_idxs, axis=0)
    frame_sz = np.concatenate(frame_sz, axis=0)
    sample_sz = np.array(sample_sz, dtype=int)

    return pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs, frame_sz, sample_sz
