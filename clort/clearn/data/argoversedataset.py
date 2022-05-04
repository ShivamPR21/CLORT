import os
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.frustum_clipping import generate_frustum_planes
from matplotlib.pyplot import axis
from numpy import float32
from torch.utils.data import Dataset

from .dataframe import ArgoverseDataFrame, ArgoverseObjectDataFrame
from .utils import get_object_patch_from_image


class ArgoverseDataset(Dataset):

    def __init__(self,
                 root : str,
                 log_id : str = "",
                 max_tracklets : int = 10,
                 occlusion_thresh : float32 = 80.,
                 lidar_points_thresh : int = 30,
                 image_size_threshold : int = 50,
                 n_img_view_thresh : int = 1,
                 n_img_view_aug : int = 7,
                 aug_transforms : List[Callable] = None,
                 central_crop : bool = True,
                 img_tr_ww : Tuple[float, float] = (0.7, 0.7),
                 discard_invalid_dfs : bool = True,
                 dataframe_augmentation : bool = True,
                 augmentation_frames : int = 4,
                 img_reshape : Tuple[int, int] = (200, 200)) -> None:
        n_cam_ = len(RING_CAMERA_LIST + STEREO_CAMERA_LIST)
        assert(n_img_view_thresh <= n_cam_ and augmentation_frames <=n_cam_)

        self.root = root
        self.log_id = log_id
        self.max_tracklets = max_tracklets
        self.occlusion_thresh = occlusion_thresh
        self.lidar_points_thresh = lidar_points_thresh
        self.image_size_threshold = image_size_threshold
        self.n_img_view_thresh = n_img_view_thresh
        self.n_img_view_aug = n_img_view_aug
        self.aug_transforms = aug_transforms
        self.tracking_loader = ArgoverseTrackingLoader(root)
        self.log_list = self.tracking_loader.log_list
        self.central_crop = central_crop
        self.img_tr_ww = img_tr_ww
        self.discard_invalid_dfs = discard_invalid_dfs

        self.dataframe_augmentation = dataframe_augmentation
        self.augmentation_frames = augmentation_frames
        self.img_reshape = img_reshape

        self.dataset = None
        self.n_frames = None
        self.am = ArgoverseMap()

        if self.log_id:
            try:
                self.dataset = self.tracking_loader.get(self.log_id)
                self.n_frames = self.dataset.num_lidar_frame
            except:
                print(f'The data log_id {self.log_id} not available in dataset\n'
                      'Use `dataset_init` function to specify the dataset log.')

        # Tracking queue containts the track ids and
        # their corresponding data for each frame
        self.tracking_queue : Dict[str, List[ArgoverseObjectDataFrame]] = {}

    def dataset_init(self, log_idx : int) -> None:
        assert(log_idx < len(self.log_list) and log_idx >= 0)

        self.log_id = self.log_list[log_idx]
        self.dataset = self.tracking_loader.get(self.log_id)
        self.n_frames = self.dataset.num_lidar_frame

    def pre_load_tracking_frames(self, frames_upto:int):
        n_frames = self.n_frames
        if frames_upto is not None:
            if frames_upto != -1:
                n_frames = min(frames_upto, n_frames)

        for i in range(n_frames):
            self.load_data_frame(i)

    def load_data_frame(self, idx : int):
        assert(not self.dataset is None)
        assert(not self.n_frames is None and 0<=idx and idx < self.n_frames)
        timestamp = int(os.path.basename(self.dataset.lidar_list[idx]).split('.')[0].split('_')[1])
        pcloud = self.dataset.get_lidar(idx)
        pcloud = self.__prune_point_cloud__(idx, pcloud)
        dataframe = ArgoverseDataFrame(timestamp)
        dataframe.set_lidar(pcloud)
        dataframe.set_valid()

        for i, camera in enumerate(RING_CAMERA_LIST + STEREO_CAMERA_LIST):
            img = self.dataset.get_image_sync(idx, camera)
            dataframe.set_iamge(img, i)

        objects = self.dataset.get_label_object(idx)
        for obj in objects:
            if obj.occlusion >= self.occlusion_thresh:
                continue

            df_obj = ArgoverseObjectDataFrame(timestamp, obj.track_id, augmentation=True,
                                              img_resize=self.img_reshape, n_images=self.n_img_view_aug)
            if (self.__populate_object_dataframe__(dataframe, obj, df_obj, self.lidar_points_thresh)):
                df_obj.set_valid()

            if self.discard_invalid_dfs and not df_obj.is_valid():
                continue

            df_obj.generate_trainable_img_data()
            if df_obj.track_id in self.tracking_queue and len(self.tracking_queue[df_obj.track_id]) < self.max_tracklets:
                self.tracking_queue[df_obj.track_id] += [df_obj]
            else:
                self.tracking_queue[df_obj.track_id] = [df_obj]

    def __populate_object_dataframe__(self,
                                df : ArgoverseDataFrame,
                                obj : ObjectLabelRecord,
                                obj_df : ArgoverseObjectDataFrame,
                                cloud_thresh : int = 30) -> bool:
        obj_df.bbox = obj.as_3d_bbox()
        # For lidar dataframe
        pcloud = self.__segment_cloud__(obj_df.bbox, df.lidar)
        if pcloud.shape[0] <= cloud_thresh:
            return False

        pcloud -= pcloud.mean(axis=0, keepdims=True)

        obj_df.set_lidar(pcloud)

        # For camera dataframe
        img_cnt = 0;
        for i, cam in enumerate(obj_df.cam_list):
            calib = self.dataset.get_calibration(cam)
            img = df.get_image(i)
            planes = generate_frustum_planes(calib.K, calib.camera)
            uv_cam = calib.project_ego_to_cam(obj_df.bbox)
            obj_patch = get_object_patch_from_image(img,
                                                    uv_cam[:, :3],
                                                    planes.copy(),
                                                    deepcopy(calib.camera_config),
                                                    self.central_crop,
                                                    self.img_tr_ww)
            # if not obj_patch is None:
            #     print(f'Object patch shapre : {obj_patch.shape}')

            if ((not obj_patch is None) and (obj_patch.shape[0] >= 50) and (obj_patch.shape[1] >= 50)):
                img_cnt += 1
                obj_df.set_iamge(obj_patch, i)

        if img_cnt < self.n_img_view_thresh:
            return False

        return True

    def __segment_cloud__(self,
                        box : np.ndarray,
                        lidar : np.ndarray) -> np.ndarray:
        p, p1, p2, p3 = box[0], box[1], box[3], box[4]

        d = lidar - p
        dnorm = np.linalg.norm(d, axis = 1, keepdims=True)
        mask = np.ones((1, len(lidar)), dtype=np.bool8)
        for p_ in [p1, p2, p3]:
            d1 = np.expand_dims(p_ - p, axis = 0)
            d1norm = np.linalg.norm(d1)
            cost = d1.dot(d.T)/(dnorm.T*d1norm)
            dist = dnorm.T*cost
            tmp_mask = np.logical_and(dist >= 0, dist <= d1norm)
            mask = np.logical_and(mask, tmp_mask)

        return lidar[mask[0]]

    def __prune_point_cloud__(self,
                            idx : int,
                            lidar : np.ndarray,
                            prune_non_roi : bool = True,
                            prune_ground : bool = True) -> np.ndarray:
        city_to_egovehicle_se3 = self.dataset.get_pose(idx)
        city_name = self.dataset.city_name
        roi_area_pts = deepcopy(lidar)
        roi_area_pts = city_to_egovehicle_se3.transform_point_cloud(
            roi_area_pts
        )
        if prune_non_roi:
            roi_area_pts = self.am.remove_non_roi_points(roi_area_pts, city_name)

        if prune_ground:
            roi_area_pts = self.am.remove_ground_surface(roi_area_pts, city_name)

        roi_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(
            roi_area_pts
        )
        return roi_area_pts

    def count_examples(self):
        n = list(self.tracking_queue.keys()).__len__()
        return n
