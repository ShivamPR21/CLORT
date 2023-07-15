from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.attention import MultiHeadSelfAttentionLinear
from moduleZoo.dense import LinearNormActivation

from . import MinimalCrossObjectEncoder


class MultiModalEncoder(nn.Module):

    def __init__(self, mv_in_dim: int = 256, pc_in_dim: int = 128, udim: int = 256, out_dim: int = 128,
                 norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
                 activation_layer: Callable[..., nn.Module] | None = nn.SELU,
                 enable_xo: bool = False, features_only: bool = False) -> None:
        super().__init__()

        self.eps = 1e-9
        self.enable_xo = enable_xo
        self.features_only = features_only
        self.udim = udim

        self.mv_linear = LinearNormActivation(mv_in_dim, self.udim, norm_layer=norm_layer, activation_layer=activation_layer)
        self.pc_linear = LinearNormActivation(pc_in_dim, self.udim, norm_layer=norm_layer, activation_layer=activation_layer)

        enc_layers = [512, 256, 128]

        # Block 1
        self.f_p1 = LinearNormActivation(self.udim, enc_layers[0], norm_layer=norm_layer, activation_layer=activation_layer)
        self.gat_1 = MultiHeadSelfAttentionLinear(enc_layers[0], None, n_heads=2, residual=True)
        self.xo1 = MinimalCrossObjectEncoder(enc_layers[0], enc_layers[0], k=10,
                                            norm_layer=norm_layer, activation_layer=activation_layer,
                                            similarity='cosine') if self.enable_xo else None
        self.p1 = LinearNormActivation(enc_layers[0], enc_layers[0], norm_layer=norm_layer, activation_layer=activation_layer)

        # Block 2
        self.f_p2 = LinearNormActivation(enc_layers[0], enc_layers[1], norm_layer=norm_layer, activation_layer=activation_layer)
        self.gat_2 = MultiHeadSelfAttentionLinear(enc_layers[1], None, n_heads=2, residual=True)
        self.xo2 = MinimalCrossObjectEncoder(enc_layers[1], enc_layers[1], k=10,
                                            norm_layer=norm_layer, activation_layer=activation_layer,
                                            similarity='cosine') if self.enable_xo else None
        self.p2 = LinearNormActivation(enc_layers[1], enc_layers[1], norm_layer=norm_layer, activation_layer=activation_layer)

        # Block 3
        self.f_p3 = LinearNormActivation(enc_layers[1], enc_layers[2], norm_layer=norm_layer, activation_layer=activation_layer)
        self.gat_3 = MultiHeadSelfAttentionLinear(enc_layers[2], None, n_heads=2, residual=True)
        self.xo3 = MinimalCrossObjectEncoder(enc_layers[2], enc_layers[2], k=10,
                                            norm_layer=norm_layer, activation_layer=activation_layer,
                                            similarity='cosine') if self.enable_xo else None
        self.p3 = LinearNormActivation(enc_layers[2], enc_layers[2], norm_layer=norm_layer, activation_layer=activation_layer)

        self.projection_head1 = LinearNormActivation(np.sum(enc_layers) + enc_layers[0], 512, bias=True,
                                                    norm_layer=norm_layer,
                                                    activation_layer=activation_layer)

        self.projection_head2 = LinearNormActivation(512, out_dim, bias=True,
                                                    norm_layer=None,
                                                    activation_layer=None)

        self.out_dim = 512 if self.features_only else out_dim

    def forward(self, mv_enc: torch.Tensor, pc_enc: torch.Tensor, n_nodes: np.ndarray | None = None) -> torch.Tensor:
        # mv_enc -> [n_obj, N_mv]
        # pc_enc -> [n_obj, N_pc]

        mv_enc, pc_enc = self.mv_linear(mv_enc).unsqueeze(dim=1), self.pc_linear(pc_enc).unsqueeze(dim=1)
        enc = torch.cat([mv_enc, pc_enc], dim=1) # [N, 2, udim]

        enc = self.f_p1(enc)

        enc1 = self.gat_1(enc).max(dim=2).values
        enc2 = self.gat_2(self.f_p2(enc1)).max(dim=2).values
        enc3 = self.gat_3(self.f_p3(enc2)).max(dim=2).values

        enc, enc1, enc2, enc3 = enc.max(dim=1).values, enc1.max(dim=1).values, \
            enc2.max(dim=1).values, enc3.max(dim=1).values

        if (self.xo1 is not None and self.xo2 is not None and self.xo3 is not None and self.enable_xo):
            enc1, enc2, enc3 = self.xo1(enc1, n_nodes), self.xo2(enc2, n_nodes), self.xo3(enc3, n_nodes)

        enc = torch.cat([enc, self.p1(enc1), self.p2(enc2), self.p3(enc3)], dim=1)

        enc = self.projection_head1(enc)

        if not self.features_only:
            enc = self.projection_head2(enc)
            enc = enc/(enc.norm(dim=1, keepdim=True) + self.eps)

        return enc
