import torch
import torch.nn as nn
from moduleZoo.attention import SelfAttentionLinear


class MultiModalEncoder(nn.Module):

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()

        self.eps = 1e-9
        self.act = nn.Tanh()

        self.gat_mv1 = SelfAttentionLinear(256, 128,
                                            residual=True)


        self.gat_pc1 = SelfAttentionLinear(128, 128,
                                            residual=True)

        self.combined_gat = SelfAttentionLinear(128, 128,
                                                residual=True)

        self.projection_head = nn.Sequential([nn.Linear(128, out_dim), nn.Tanh()])

    def forward(self, mv_enc: torch.Tensor, pc_enc: torch.Tensor) -> torch.Tensor:
        # mv_enc -> [n_obj, N_mv]
        # pc_enc -> [n_obj, N_pc]

        mv_enc, pc_enc = mv_enc.unsqueeze(dim=1), pc_enc.unsqueeze(dim=1)
        q_mv, k_mv, v_mv = self.gat_mv1.extract_qkv(mv_enc) # [n_obj, 1, q/k/v]
        q_pc, k_pc, v_pc = self.gat_pc1.extract_qkv(pc_enc) # [n_obj, 1, q/k/v]

        res_mv = self.gat_mv1(mv_enc, q_mv.repeat(1, 2, 1), k_mv.repeat(1, 2, 1), torch.cat([v_mv, v_pc], dim=1)).max(dim=1, keepdims=True).values

        res_pc = self.gat_mv1(pc_enc, q_pc.repeat(1, 2, 1), k_pc.repeat(1, 2, 1), torch.cat([v_mv, v_pc], dim=1)).max(dim=1, keepdims=True).values

        res = self.combined_gat(torch.cat([res_mv, res_pc], dim=1)).max(dim=1).values

        res = self.projection_head(res)

        res /= (res.norm(dim=1, keepdim=False) + self.eps)
        return res
