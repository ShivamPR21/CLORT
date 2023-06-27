from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import GraphConv, SelfGraphAttentionLinear


class CrossObjectEncoder(nn.Module):

    def __init__(self, in_dim : int = 256, out_dim : int = 128,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU) -> None:
        super().__init__()

        self.eps = 1e-9
        self.in_dim, self.out_dim = in_dim, out_dim

        enc_layers = [128, 64, 128]

        self.gat1 = SelfGraphAttentionLinear(self.in_dim, None, residual=True, dynamic_batching=True)
        self.conv1 = GraphConv(self.in_dim, enc_layers[0], bias=True, k=10,
                               reduction='max', features='local+global',
                               norm_layer=norm_layer, activation_layer=activation_layer,
                               dynamic_batching=True, enable_offloading=False)

        self.gat2 = SelfGraphAttentionLinear(enc_layers[0], None, residual=True, dynamic_batching=True)
        self.conv2 = GraphConv(enc_layers[0], enc_layers[1], bias=True, k=10,
                               reduction='max', features='local+global',
                               norm_layer=norm_layer, activation_layer=activation_layer,
                               dynamic_batching=True, enable_offloading=False)

        self.gat3 = SelfGraphAttentionLinear(enc_layers[1], None, residual=True, dynamic_batching=True)
        self.conv3 = GraphConv(enc_layers[1], enc_layers[2], bias=True, k=10,
                               reduction='max', features='local+global',
                               norm_layer=norm_layer, activation_layer=activation_layer,
                               dynamic_batching=True, enable_offloading=False)

        self.projection_head = LinearNormActivation(np.sum(enc_layers), self.out_dim, bias=True,
                                                    norm_layer=None, activation_layer=None)

    def forward(self, obj_encs: torch.Tensor, n_nodes: np.ndarray) -> torch.Tensor:
        obj_encs1 = self.conv1(self.gat1(obj_encs, n_nodes), n_nodes)
        obj_encs2 = self.conv2(self.gat2(obj_encs1, n_nodes), n_nodes)
        obj_encs3 = self.conv3(self.gat3(obj_encs2, n_nodes), n_nodes)

        obj_encs = torch.cat([obj_encs1, obj_encs2, obj_encs3], dim=1)

        obj_encs = self.projection_head(obj_encs)

        obj_encs = obj_encs/(obj_encs.norm(dim=1, keepdim=True) + self.eps)

        return obj_encs
