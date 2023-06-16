from typing import List

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import GraphConv


class BboxEncoder(nn.Module):

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()

        self.graph_conv1 = GraphConv(3, 64, k=4, reduction='max',
                                    features='local+global',
                                    norm_layer=nn.BatchNorm1d,
                                    activation_layer=nn.ReLU6,
                                    dynamic_batching=False)

        self.graph_conv2 = GraphConv(64, 64, k=4, reduction='max',
                                    features='local+global',
                                    norm_layer=nn.BatchNorm1d,
                                    activation_layer=nn.ReLU6,
                                    dynamic_batching=False)

        self.projection_head = LinearNormActivation(64*2, out_dim, bias=True, activation_layer=nn.ReLU6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [n_obj, 8, 3]
        x1 = self.graph_conv1(x) # [n_obj, 8, 64]
        x2 = self.graph_conv2(x1) # [n_obj, 8, 64]

        x2 = torch.cat([x1, x2], dim=-1).max(dim=1).values # max aggregation
        x2 = self.projection_head(x2) # projection head

        return x2

class PointCloudEncoder(nn.Module):

    def __init__(self, out_dims : int = 128, bbox_aug : bool = True) -> None:
        super().__init__()

        self.eps = 1e-9
        self.graph_conv1 = GraphConv(3, 64, k=10, reduction='max',
                                    features='local+global',
                                    norm_layer=nn.BatchNorm1d,
                                    activation_layer=nn.ReLU6,
                                    dynamic_batching=True)

        self.graph_conv2 = GraphConv(64, 64, k=10, reduction='max',
                            features='local+global',
                            norm_layer=None,
                            activation_layer=nn.ReLU6,
                            dynamic_batching=True)

        self.graph_conv3 = GraphConv(64, 64, k=10, reduction='max',
                            features='local+global',
                            norm_layer=nn.BatchNorm1d,
                            activation_layer=nn.ReLU6,
                            dynamic_batching=True)

        self.graph_conv4 = GraphConv(64, 64, k=10, reduction='max',
                            features='local+global',
                            norm_layer=None,
                            activation_layer=nn.ReLU6,
                            dynamic_batching=True)

        self.bbox_enc = BboxEncoder(64) if bbox_aug else None

        self.projection_head = LinearNormActivation(64*4 + (64 if bbox_aug else 0), out_dims, bias=True, activation_layer=nn.Tanh)

    def aggregate(self, f: torch.Tensor, sz_arr: List[List[int]]) -> torch.Tensor:
        out = torch.zeros((len(sz_arr), f.shape[-1]), dtype=torch.float32, device=f.device)

        for i, sz_idxs in enumerate(sz_arr):
            out[i, :] = f[sz_idxs, :].max(dim=0).values

        return out

    def forward(self, x: torch.Tensor, n_pts: np.ndarray, bbox: torch.Tensor | None = None) -> torch.Tensor:

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

        f = torch.cat([f1, f2, f3, f4], dim=1)

        f_bbox = self.bbox_enc(bbox) if bbox is not None and self.bbox_enc is not None else None
        f = torch.cat([f, f_bbox], dim=1) if f_bbox is not None else f

        f = self.projection_head(f)

        f = f/(f.norm(dim=1, keepdim=True) + self.eps)

        return f
