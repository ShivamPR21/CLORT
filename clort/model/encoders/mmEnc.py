from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.attention import SelfAttentionLinear
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

        self.gat_mv1 = SelfAttentionLinear(mv_in_dim, 128,
                                            residual=True)


        self.gat_pc1 = SelfAttentionLinear(pc_in_dim, 128,
                                            residual=True)

        self.combined_gat = SelfAttentionLinear(128, 128,
                                                residual=True)

        self.xo_gat = MinimalCrossObjectEncoder(128, 128, k=10,
                                                norm_layer=norm_layer, activation_layer=activation_layer) if self.enable_xo else None

        self.projection_head = LinearNormActivation(128, out_dim, bias=True,
                                                    norm_layer=None,
                                                    activation_layer=None)

    def forward(self, mv_enc: torch.Tensor, pc_enc: torch.Tensor, n_nodes: np.ndarray | None = None) -> torch.Tensor:
        # mv_enc -> [n_obj, N_mv]
        # pc_enc -> [n_obj, N_pc]

        mv_enc, pc_enc = mv_enc.unsqueeze(dim=1), pc_enc.unsqueeze(dim=1)
        q_mv, k_mv, v_mv = self.gat_mv1.extract_qkv(mv_enc, mv_enc) # [n_obj, 1, q/k/v]
        q_pc, k_pc, v_pc = self.gat_pc1.extract_qkv(pc_enc, pc_enc) # [n_obj, 1, q/k/v]

        res_mv = self.gat_mv1(mv_enc, q_mv.repeat(1, 2, 1), k_mv.repeat(1, 2, 1), torch.cat([v_mv, v_pc], dim=1)).squeeze(dim=1).max(dim=1, keepdims=True).values

        res_pc = self.gat_pc1(pc_enc, q_pc.repeat(1, 2, 1), k_pc.repeat(1, 2, 1), torch.cat([v_mv, v_pc], dim=1)).squeeze(dim=1).max(dim=1, keepdims=True).values

        res = self.combined_gat(torch.cat([res_mv, res_pc], dim=1)).squeeze(dim=1).max(dim=1).values

        res = self.xo_gat(res, n_nodes) if (self.xo_gat is not None and n_nodes is not None) else res

        res = self.projection_head(res)

        res = res/(res.norm(dim=1, keepdim=True) + self.eps)
        return res
