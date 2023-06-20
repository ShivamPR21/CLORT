from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.convolution import ConvNormActivation2d
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import SelfGraphAttentionLinear
from moduleZoo.resblocks import (
    ConvBottleNeckResidualBlock2d,
    ConvInvertedResidualBlock2d,
)

# from timm import create_model


class SingleViewEncoder(nn.Module):

    def __init__(self, image_shape: Tuple[int, int] = (224, 224), in_dim:int = 3, out_dim: int = 256) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.in_dim, self.out_dim = in_dim, out_dim

        self.conv1 = ConvNormActivation2d(self.in_dim, 64, kernel_size=5, stride=2,
                                          norm_layer=None, activation_layer=nn.SELU)

        self.conv2 = ConvNormActivation2d(64, 64, kernel_size=5, stride=2,
                                          norm_layer=None, activation_layer=nn.SELU)

        self.res3 = ConvBottleNeckResidualBlock2d(64, 4, 64, kernel_size=3, stride=1,
                                                  norm_layer=None, activation_layer=nn.SELU)

        self.res4 = ConvBottleNeckResidualBlock2d(64, 4, 128, kernel_size=3, stride=2,
                                                  norm_layer=None, activation_layer=nn.SELU)

        self.res5 = ConvInvertedResidualBlock2d(128, 2, 256, kernel_size=3, stride=2,
                                                norm_layer=None, activation_layer=nn.SELU)

        self.res6 = ConvInvertedResidualBlock2d(256, 2, self.out_dim, kernel_size=3, stride=2,
                                                norm_layer=None, activation_layer=nn.SELU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        return x

class MultiViewEncoder(nn.Module):

    def __init__(self, image_shape : Tuple[int, int] = (224, 224), out_dim: int = 256) -> None:
        super().__init__()

        self.eps = 1e-9
        self.image_shape = image_shape

        # self.sv_enc1 = create_model('dla34', features_only=True, pretrained=True)
        self.sv_enc1 = SingleViewEncoder(self.image_shape, 3, 512)

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.sv_enc2 = LinearNormActivation(512, 256, norm_layer=None, activation_layer=nn.SELU)
        self.sv_enc3 = LinearNormActivation(256, 128, activation_layer=nn.SELU)

        self.gat = SelfGraphAttentionLinear(128, None, residual=True, dynamic_batching=True)
        self.gat_act = nn.SELU()
        self.projection_head = LinearNormActivation(128, out_dim, activation_layer=None)

    def forward(self, x : torch.Tensor, n_views: np.ndarray) -> torch.Tensor:
        # x -> [N, 3, W, H]
        # n_views -> [nv....]

        # Atomic encodings
        # x = self.max_pool(self.sv_enc1(x)[-1]).flatten(start_dim=1) # [N, 512]
        x = self.max_pool(self.sv_enc1(x)).flatten(start_dim=1) # [N, 512]
        x = self.sv_enc3(self.sv_enc2(x)) # [N, 128]

        x = self.gat_act(self.gat(x, n_views)) # [N, 128]

        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_views.tolist(), dim=0)]

        output = torch.zeros((len(sz_arr), 128), dtype=torch.float32, device=x.device)

        for i, sz_idxs in enumerate(sz_arr):
            output[i, :] = x[sz_idxs, :].max(dim=0).values

        output = self.projection_head(output)

        output = output/(output.norm(dim=1, keepdim=True) + self.eps)

        return output
