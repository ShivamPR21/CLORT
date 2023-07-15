from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import GraphConv, SelfGraphAttentionLinear


class MinimalCrossObjectEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, k: int,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU,
                 use_attention: bool = True,
                 red_factor: int = 2,
                 similarity: str = 'euclidean') -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.hidden_dim = self.in_dim//red_factor

        self.gat = SelfGraphAttentionLinear(self.in_dim, self.hidden_dim, residual=True, dynamic_batching=True) if use_attention else None
        self.conv = GraphConv(self.hidden_dim, self.out_dim, bias=True, k=k,
                               reduction='max', features='local+global', similarity=similarity,
                               norm_layer=norm_layer, activation_layer=activation_layer,
                               dynamic_batching=True, enable_offloading=False)

    def forward(self, obj_encs:torch.Tensor, n_nodes: np.ndarray) -> torch.Tensor:
        x = self.gat(obj_encs, n_nodes) if self.gat is not None else obj_encs
        return self.conv(x, n_nodes)

class CrossObjectEncoder(nn.Module):

    def __init__(self, in_dim : int = 256, out_dim : int = 128,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU) -> None:
        super().__init__()

        self.eps = 1e-9
        self.in_dim, self.out_dim = in_dim, out_dim

        enc_layers = [512, 256, 128]

        # Block 1
        self.f_p1 = LinearNormActivation(self.in_dim, enc_layers[0],
                                         norm_layer=norm_layer, activation_layer=activation_layer)
        self.xo1 = MinimalCrossObjectEncoder(enc_layers[0], enc_layers[0], k = 10, norm_layer=norm_layer,
                                             activation_layer=activation_layer, similarity='cosine')
        self.p1 = LinearNormActivation(enc_layers[0], enc_layers[0], norm_layer=norm_layer, activation_layer=activation_layer)

        # Block 2
        self.f_p2 = LinearNormActivation(enc_layers[0], enc_layers[1],
                                         norm_layer=norm_layer, activation_layer=activation_layer)
        self.xo2 = MinimalCrossObjectEncoder(enc_layers[1], enc_layers[1], k = 10, norm_layer=norm_layer,
                                             activation_layer=activation_layer, similarity='cosine')
        self.p2 = LinearNormActivation(enc_layers[1], enc_layers[1], norm_layer=norm_layer, activation_layer=activation_layer)

        # Block 3
        self.f_p3 = LinearNormActivation(enc_layers[1], enc_layers[2],
                                         norm_layer=norm_layer, activation_layer=activation_layer)
        self.xo3 = MinimalCrossObjectEncoder(enc_layers[2], enc_layers[2], k = 10, norm_layer=norm_layer,
                                             activation_layer=activation_layer, similarity='cosine')
        self.p3 = LinearNormActivation(enc_layers[2], enc_layers[2], norm_layer=norm_layer, activation_layer=activation_layer)

        self.projection_head1 = LinearNormActivation(np.sum(enc_layers) + enc_layers[0], 512, bias=True,
                                                    norm_layer=None, activation_layer=None)

        self.projection_head2 = LinearNormActivation(512, self.out_dim, bias=True,
                                                    norm_layer=None, activation_layer=None)

    def forward(self, obj_encs: torch.Tensor, n_nodes: np.ndarray) -> torch.Tensor:
        obj_encs = self.f_p1(obj_encs)

        obj_encs1 = self.xo1(obj_encs, n_nodes) #self.conv1(self.gat1(obj_encs, n_nodes), n_nodes)
        obj_encs2 = self.xo2(self.f_p2(obj_encs1), n_nodes) #self.conv2(self.gat2(obj_encs1, n_nodes), n_nodes)
        obj_encs3 = self.xo3(self.f_p3(obj_encs2), n_nodes) #self.conv3(self.gat3(obj_encs2, n_nodes), n_nodes)

        obj_encs = torch.cat([obj_encs, self.p1(obj_encs1), self.p2(obj_encs2), self.p3(obj_encs3)], dim=1)

        obj_encs = self.projection_head1(obj_encs)

        obj_encs = self.projection_head2(obj_encs)

        obj_encs = obj_encs/(obj_encs.norm(dim=1, keepdim=True) + self.eps)

        return obj_encs
