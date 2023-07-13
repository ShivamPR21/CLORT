from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.attention import MultiHeadSelfAttentionLinear
from moduleZoo.dense import LinearNormActivation

from . import MinimalCrossObjectEncoder


class MultiModalEncoder(nn.Module):

    def __init__(self, mv_in_dim = 256, pc_in_dim = 128, out_dim: int = 128,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU,
                 enable_xo: bool = False) -> None:
        super().__init__()

        self.eps = 1e-9
        self.enable_xo = enable_xo
        self.udim = max(mv_in_dim, pc_in_dim)

        self.mv_linear = LinearNormActivation(mv_in_dim, self.udim, norm_layer=norm_layer, activation_layer=activation_layer)
        self.pc_linear = LinearNormActivation(pc_in_dim, self.udim, norm_layer=norm_layer, activation_layer=activation_layer)

        self.gat_1 = MultiHeadSelfAttentionLinear(self.udim, None, n_heads=2, residual=True)

        self.xo_gat = MinimalCrossObjectEncoder(self.udim, self.udim, k=10,
                                                norm_layer=norm_layer, activation_layer=activation_layer, similarity='cosine') if self.enable_xo else None

        self.projection_head1 = LinearNormActivation(self.udim + (self.udim if self.enable_xo else 0), self.udim, bias=True,
                                                    norm_layer=norm_layer,
                                                    activation_layer=activation_layer)

        self.projection_head2 = LinearNormActivation(self.udim, out_dim, bias=True,
                                                    norm_layer=None,
                                                    activation_layer=None)

    def forward(self, mv_enc: torch.Tensor, pc_enc: torch.Tensor, n_nodes: np.ndarray | None = None) -> torch.Tensor:
        # mv_enc -> [n_obj, N_mv]
        # pc_enc -> [n_obj, N_pc]

        mv_enc, pc_enc = self.mv_linear(mv_enc).unsqueeze(dim=1), self.pc_linear(pc_enc).unsqueeze(dim=1)
        enc = self.gat_1(torch.cat([mv_enc, pc_enc], dim=1)).flatten(1, 2).max(dim=1).values

        enc = torch.cat([enc, self.xo_gat(enc, n_nodes)], dim=-1) if (self.xo_gat is not None and n_nodes is not None) else enc

        enc = self.projection_head1(enc)
        enc = self.projection_head2(enc)

        enc = enc/(enc.norm(dim=1, keepdim=True) + self.eps)
        return enc
