from typing import List

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.graphs import GraphConv


class PointCloudEncoder(nn.Module):

    def __init__(self, out_dims : int = 128) -> None:
        super().__init__()

        self.eps = 1e-9
        self.graph_conv1 = GraphConv(3, 64, k=10, reduction='max',
                                    features='local+global',
                                    norm_layer=nn.BatchNorm1d,
                                    activation_layer=nn.ReLU)

        self.graph_conv2 = GraphConv(64, 64, k=10, reduction='max',
                            features='local+global',
                            norm_layer=None,
                            activation_layer=nn.ReLU)

        self.graph_conv3 = GraphConv(64, 64, k=10, reduction='max',
                            features='local+global',
                            norm_layer=nn.BatchNorm1d,
                            activation_layer=nn.ReLU)

        self.graph_conv4 = GraphConv(64, 64, k=10, reduction='max',
                            features='local+global',
                            norm_layer=None,
                            activation_layer=nn.ReLU)

        self.projection_head = nn.Sequential([nn.Linear(256, out_dims, bias=True), nn.Tanh()])

    def aggregate(self, f: torch.Tensor, sz_arr: List[List[int]]) -> torch.Tensor:
        out = torch.zeros((len(sz_arr), f.shape[-1]), dtype=torch.float32)

        for i, sz_idxs in enumerate(sz_arr):
            out[i, :] = f[sz_idxs, :].max(dim=0)[0]

        return out

    def forward(self, x: torch.Tensor, n_pts: np.ndarray) -> torch.Tensor:

        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_pts.tolist(), dim=0)]

        f1 = self.graph_conv1(x) # [N, 64]

        f2 = self.graph_conv2(f1) # [N, 64]

        f3 = self.graph_conv3(f2) # [N, 64]

        f4 = self.graph_conv4(f3) # [N, 64]

        f1, f2, f3, f4 = \
            self.aggregate(f1, sz_arr), \
                self.aggregate(f2, sz_arr), \
                    self.aggregate(f3, sz_arr), \
                        self.aggregate(f4, sz_arr)

        f = torch.cat([f1, f2, f3, f4], dim=1)

        f = self.projection_head(f)

        f /= (f.norm(dim=1, keepdim=False) + self.eps)

        return f
