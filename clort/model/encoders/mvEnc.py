from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import SelfGraphAttentionLinear
from timm import create_model


class MultiViewEncoder(nn.Module):

    def __init__(self, image_shape : Tuple[int, int] = (224, 224), out_dim: int = 256) -> None:
        super().__init__()

        self.eps = 1e-9
        self.image_shape = image_shape

        self.sv_enc1 = create_model('dla34', features_only=True, pretrained=True)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.sv_enc2 = LinearNormActivation(512, 256, norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU)
        self.sv_enc3 = LinearNormActivation(256, 128, activation_layer=nn.ReLU)

        self.gat = SelfGraphAttentionLinear(128, None, residual=True, dynamic_batching=True)
        self.gat_act = nn.ReLU()
        self.projection_head = LinearNormActivation(128, out_dim, activation_layer=None)

    def forward(self, x : torch.Tensor, n_views: np.ndarray) -> torch.Tensor:
        # x -> [N, 3, W, H]
        # n_views -> [nv....]

        # Atomic encodings
        x = self.max_pool(self.sv_enc1(x)[-1]).flatten(start_dim=1) # [N, 512]
        x = self.sv_enc3(self.sv_enc2(x)) # [N, 128]

        x = self.gat_act(self.gat(x, n_views)) # [N, 128]

        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_views.tolist(), dim=0)]

        output = torch.zeros((len(sz_arr), 128), dtype=torch.float32, device=x.device)

        for i, sz_idxs in enumerate(sz_arr):
            output[i, :] = x[sz_idxs, :].max(dim=0).values

        output = self.projection_head(output)

        output = output/(output.norm(dim=1, keepdim=True) + self.eps)

        return output
