from typing import Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.graphs import SelfGraphAttentionLinear


class MultiViewEncoder(nn.Module):

    def __init__(self, sv_enc: Type[nn.Module], image_shape : Tuple[int, int] = (224, 224)) -> None:
        super().__init__()

        self.eps = 1e-9
        self.image_shape = image_shape

        self.sv_enc = sv_enc
        self.projection_head1 = nn.Sequential([nn.Linear(128, 256, bias=True), nn.Tanh()])
        self.gat = SelfGraphAttentionLinear(256, None, residual=True, dynamic_batching=True)
        self.projection_head2 = nn.Sequential([nn.Linear(256, 128, bias=True), nn.Tanh()])

    def forward(self, x : torch.Tensor, n_views: np.ndarray) -> torch.Tensor:
        # x -> [N, 3, W, H]
        # n_views -> [nv....]

        # Atomic encodings
        x = self.sv_enc(x) # [N, 128]
        x = self.gat(x, n_views) # [N, 128]

        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_views.tolist(), dim=0)]

        output = torch.zeros((len(sz_arr), 128), dtype=torch.float32)

        for i, sz_idxs in enumerate(sz_arr):
            output[i, :] = x[sz_idxs, :].max(dim=0)[0]

        output = self.projection_head2(output)

        output /= (output.norm(dim=1, keepdim=False) + self.eps)

        return output
