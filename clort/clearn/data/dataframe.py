from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from PIL import Image
from torchvision.transforms import RandomPerspective, RandomResizedCrop


class ArgoverseDataFrame:
    def __init__(self, timestamp: int) -> None:
        self.timestamp = timestamp
        self.lidar: np.ndarray = None
        self.bbox: np.ndarray = None
        self.valid: bool = False
        self.images: Dict[str, np.ndarray] = {}
        self.cam_list = RING_CAMERA_LIST + STEREO_CAMERA_LIST
        for cam_name in self.cam_list:
            self.images.update({cam_name: None})

    def set_iamge(self, image: np.ndarray, cam_idx: int = 0) -> None:
        camera = self.cam_list[cam_idx]
        self.images.update({camera: image})

    def set_lidar(self, pcloud: np.ndarray) -> None:
        self.lidar = pcloud

    def get_image(self, cam_idx: int = 0) -> np.ndarray:
        camera = self.cam_list[cam_idx]
        return self.images[camera]

    def get_lidar(self) -> np.ndarray:
        return self.lidar

    def is_valid(self) -> bool:
        return self.valid

    def set_valid(self, flag: bool = True):
        self.valid = flag


class ArgoverseObjectDataFrame(ArgoverseDataFrame):
    def __init__(
        self,
        timestamp: int,
        track_id: Optional[str],
        augmentation: bool,
        img_resize: Tuple[int, int],
        n_images: int,
    ) -> None:
        super().__init__(timestamp)
        self.track_id = track_id
        self.augmentation = augmentation
        self.img_resize = img_resize
        self.n_images = n_images
        self.img_data: List[Union[np.ndarray, torch.Tensor]] = []
        self.bbox_global: np.ndarray = None  # Required for inference
        self.object_type: str = None
        self.dims: np.ndarray = None
        self.score: float = None
        self.n_orig_imgs: int = -1
        self.global_pcl: np.ndarray = None
        self.local_to_global_transform: np.ndarray = None

    def generate_trainable_img_data(self, transforms: List[Callable] = None):
        assert self.is_valid()

        if transforms is None:
            transforms = [
                RandomPerspective(distortion_scale=0.6, p=1.0),
                RandomResizedCrop(size=self.img_resize),
            ]

        for _, img in self.images.items():
            if img is None:
                continue

            loc_img = Image.fromarray(img)
            loc_img = (
                np.asarray(loc_img.resize(self.img_resize, resample=Image.BICUBIC))
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            loc_img /= 255.0
            self.img_data += [torch.as_tensor(loc_img, dtype=torch.float32)]

        self.n_orig_imgs = len(self.img_data)
        if len(self.img_data) > self.n_images:
            idx = np.arange(len(self.img_data))
            idx = np.random.choice(idx, self.n_images, replace=False)
            self.img_data = self.img_data[idx]
        elif len(self.img_data) < self.n_images:
            n_imgs = len(self.img_data)
            while len(self.img_data) < self.n_images:
                # Convert to tensor
                loc_img = torch.as_tensor(
                    self.img_data[np.random.randint(0, n_imgs)], dtype=torch.float32
                )
                tr = transforms[np.random.randint(0, len(transforms))]
                loc_img = tr(loc_img)
                self.img_data += [loc_img]

    def generate_inference_img_data(self):
        assert self.is_valid()
        self.img_data = []

        for _, img in self.images.items():
            if img is None:
                continue

            loc_img = Image.fromarray(img)
            loc_img = (
                np.asarray(loc_img.resize(self.img_resize, resample=Image.BICUBIC))
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            loc_img /= 255.0
            self.img_data += [torch.as_tensor(loc_img, dtype=torch.float32)]

        self.n_orig_imgs = len(self.img_data)
        if len(self.img_data) > self.n_images:
            idx = np.arange(len(self.img_data))
            idx = np.random.choice(idx, self.n_images, replace=False)
            self.img_data = self.img_data[idx]
        elif len(self.img_data) < self.n_images:
            n_imgs = len(self.img_data)
            while len(self.img_data) < self.n_images:
                # Convert to tensor
                loc_img = torch.as_tensor(
                    self.img_data[np.random.randint(0, n_imgs)], dtype=torch.float32
                )
                self.img_data += [loc_img]

    def trackid(self):
        return self.track_id
