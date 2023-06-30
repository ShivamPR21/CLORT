from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import GraphConv
from scipy.spatial.transform import Rotation as R

from . import MinimalCrossObjectEncoder


class BboxEncoder(nn.Module):

    def __init__(self, out_dim: int = 64,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU,
                 offloading: bool = True) -> None:
        super().__init__()

        self.graph_conv1 = GraphConv(3, 64, k=4, reduction='max',
                                    features='local+global',
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dynamic_batching=False,
                                    enable_offloading=offloading)

        self.xo_enc1 = MinimalCrossObjectEncoder(64, 64,
                                                norm_layer=norm_layer,
                                                activation_layer=activation_layer)

        self.graph_conv2 = GraphConv(64, 64, k=4, reduction='max',
                                    features='local+global',
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dynamic_batching=False,
                                    enable_offloading=offloading)

        self.xo_enc2 = MinimalCrossObjectEncoder(64, 64,
                                                norm_layer=norm_layer,
                                                activation_layer=activation_layer)

        self.projection_head = LinearNormActivation(64*2, out_dim, bias=True,
                                                    norm_layer=norm_layer, activation_layer=activation_layer)

    def forward(self, x: torch.Tensor, n_nodes: np.ndarray) -> torch.Tensor:
        # x -> [n_obj, 8, 3]
        x1 = self.graph_conv1(x) # [n_obj, 8, 64]
        x2 = self.graph_conv2(x1) # [n_obj, 8, 64]

        x1, x2 = x1.max(dim=1).values, x2.max(dim=1).values # max aggregation

        x1, x2 = self.xo_enc1(x1, n_nodes), self.xo_enc2(x2, n_nodes)

        x2 = torch.cat([x1, x2], dim=-1)

        x2 = self.projection_head(x2) # projection head

        return x2

class PointCloudEncoder(nn.Module):

    def __init__(self, out_dims : int = 128, bbox_aug : bool = True,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU,
                 offloading = True) -> None:
        super().__init__()

        self.eps = 1e-9
        self.graph_conv1 = GraphConv(3, 64, k=10, reduction='max',
                                    features='local+global',
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dynamic_batching=True,
                                    enable_offloading=offloading)

        self.xo_enc1 = MinimalCrossObjectEncoder(64, 64, norm_layer=norm_layer,
                                                 activation_layer=activation_layer)

        self.graph_conv2 = GraphConv(64, 64, k=10, reduction='max',
                                    features='local+global',
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dynamic_batching=True,
                                    enable_offloading=offloading)

        self.xo_enc2 = MinimalCrossObjectEncoder(64, 64, norm_layer=norm_layer,
                                            activation_layer=activation_layer)

        self.graph_conv3 = GraphConv(64, 64, k=10, reduction='max',
                                    features='local+global',
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dynamic_batching=True,
                                    enable_offloading=offloading)

        self.xo_enc3 = MinimalCrossObjectEncoder(64, 64, norm_layer=norm_layer,
                                            activation_layer=activation_layer)

        self.graph_conv4 = GraphConv(64, 64, k=10, reduction='max',
                                    features='local+global',
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dynamic_batching=True,
                                    enable_offloading=offloading)

        self.xo_enc4 = MinimalCrossObjectEncoder(64, 64, norm_layer=norm_layer,
                                            activation_layer=activation_layer)

        self.bbox_enc = BboxEncoder(64, norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    offloading=offloading) if bbox_aug else None

        self.projection_head = LinearNormActivation(64*4 + (64 if bbox_aug else 0), out_dims, bias=True, activation_layer=None)

    def aggregate(self, f: torch.Tensor, sz_arr: List[List[int]]) -> torch.Tensor:
        out = torch.zeros((len(sz_arr), f.shape[-1]), dtype=torch.float32, device=f.device)

        for i, sz_idxs in enumerate(sz_arr):
            out[i, :] = f[sz_idxs, :].max(dim=0).values

        return out

    def forward(self, x: torch.Tensor, n_pts: np.ndarray, n_objects: np.ndarray, bbox: torch.Tensor | None = None) -> torch.Tensor:

        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_pts.tolist(), dim=0)]

        f1 = self.graph_conv1(x, n_pts) # [N, 64]

        f2 = self.graph_conv2(f1, n_pts) # [N, 64]

        f3 = self.graph_conv3(f2, n_pts) # [N, 64]

        f4 = self.graph_conv4(f3, n_pts) # [N, 64]

        f1, f2, f3, f4 = \
            self.aggregate(f1, sz_arr), \
                self.aggregate(f2, sz_arr), \
                    self.aggregate(f3, sz_arr), \
                        self.aggregate(f4, sz_arr)

        f1, f2, f3, f4 = \
            self.xo_enc1(f1, n_objects), \
                self.xo_enc2(f2, n_objects), \
                    self.xo_enc3(f3, n_objects), \
                        self.xo_enc4(f4, n_objects)

        f = torch.cat([f1, f2, f3, f4], dim=1)

        f_bbox = self.bbox_enc(bbox, n_objects) if bbox is not None and self.bbox_enc is not None else None
        f = torch.cat([f, f_bbox], dim=1) if f_bbox is not None else f

        f = self.projection_head(f)

        f = f/(f.norm(dim=1, keepdim=True) + self.eps)

        return f

class PCLGaussianNoise(nn.Module):

    def __init__(self, mean: float = 0,
                 std: float = 1,
                 tr_lim: float = 2) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.limit = tr_lim

    def forward(self, pcl:np.ndarray) -> np.ndarray:
        perturb = np.clip(np.random.randn(*pcl.shape) * self.std + self.mean,
                          -self.limit, self.limit, dtype=np.float32)
        pcl = pcl + perturb
        return pcl

class PCLRigidTransformNoise(nn.Module):

    def __init__(self, mean: Tuple[float, float] | float = 0,
                 std: Tuple[float, float] | float = 1,
                 rot_lim: float = np.pi/12,
                 trns_lim: float = 2) -> None:
        super().__init__()
        self.mean_r, self.mean_t = mean if isinstance(mean, tuple) else (mean, mean)
        self.std_r, self.std_t = std if isinstance(std, tuple) else (std, std)
        self.rot_lim, self.trns_lim = rot_lim, trns_lim

    def forward(self, pcl: np.ndarray) -> np.ndarray:
        rand_rot = R.from_euler('zxy',
                                np.clip(np.random.randn(3,) * self.std_r + self.mean_r,
                                        -self.rot_lim, self.rot_lim)).as_matrix()

        rand_t = np.clip(np.random.randn(1, 3) * self.std_t + self.mean_t,
                         -self.trns_lim, self.trns_lim)

        pcl = (pcl @ rand_rot + rand_t).astype(np.float32)

        return pcl
