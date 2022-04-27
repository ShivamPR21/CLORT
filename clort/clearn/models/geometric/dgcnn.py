from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from modelZoo.graphs import DGCNN


class PointCloudEncoder(DGCNN):

    def __init__(self, k: int,
                 embed_dim: int = 512,
                 cfg: Optional[List[Tuple[int, int, int, bool]]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.SELU) -> None:
        super().__init__(k, embed_dim, cfg, activation_layer)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x) # [B, D, n]

        x = x.transpose(2, 1) # [B, n, D]

        x_max = self.adaptive_max_pool(x).squeeze(dim=-1) # [B, n]
        x_mean = self.adaptive_avg_pool(x).squeeze(dim=-1) # [B, n]

        x = torch.cat((x_max, x_mean), dim=-1) # [B, 2*n]

        return x
