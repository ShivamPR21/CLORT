from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.convolution import ConvNormActivation2d
from moduleZoo.dense import LinearNormActivation
from moduleZoo.graphs import SelfGraphAttentionLinear
from moduleZoo.resblocks import (
    ConvBottleNeckResidualBlock2d,
    ConvInvertedResidualBlock2d,
    ConvResidualBlock2d,
)

from . import MinimalCrossObjectEncoder


class CrossViewAttention(SelfGraphAttentionLinear):

    def __init__(self, in_dim: int, out_dim: int | None = None,
                 residual: bool = True, dynamic_batching: bool = False,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None):
        super().__init__(in_dim, out_dim, residual, dynamic_batching)

        self.norm_layer = norm_layer(out_dim) if norm_layer is not None else None
        self.activation_layer = activation_layer() if activation_layer is not None else None

    def forward(self, x: torch.Tensor, n_views: np.ndarray) -> torch.Tensor:
        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_views.tolist(), dim=0)]

        output = torch.zeros((len(sz_arr), self.out_dim), dtype=torch.float32, device=x.device)

        for i, sz_idxs in enumerate(sz_arr):
            output[i, :] = x[sz_idxs, :].max(dim=0).values

        output = self.norm_layer(output) if self.norm_layer is not None else output
        output = self.activation_layer(output) if self.activation_layer is not None else output

        return output



class MultiViewEncoder(nn.Module):

    def __init__(self, image_shape: Tuple[int, int] = (224, 224), in_dim:int = 3, out_dim: int = 256,
                 norm_2d : Callable[..., nn.Module] | None=None,
                 norm_1d : Callable[..., nn.Module] | None=None,
                 activation_layer: Callable[..., nn.Module] | None=None,
                 enable_xo: bool = False) -> None:
        super().__init__()

        self.eps = 1e-9

        self.image_shape = image_shape
        self.in_dim, self.out_dim = in_dim, out_dim
        self.enable_xo = enable_xo

        self.conv1 = ConvNormActivation2d(self.in_dim, 64, kernel_size=5, stride=2,
                                          norm_layer=norm_2d, activation_layer=activation_layer)

        self.conv2 = ConvNormActivation2d(64, 128, kernel_size=5, stride=2,
                                          norm_layer=norm_2d, activation_layer=activation_layer)

        self.conv2_proj = ConvNormActivation2d(128, 128, kernel_size=5, stride=5,
                                               bias=False, norm_layer=None, activation_layer=None)
        self.xv2 = CrossViewAttention(128, 128, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo2 = MinimalCrossObjectEncoder(128, 128, k = 10, norm_layer=norm_1d,
                                             activation_layer=activation_layer, red_factor=2) if self.enable_xo else None


        self.res3 = ConvBottleNeckResidualBlock2d(128, 4, 64, kernel_size=3, stride=1,
                                                  norm_layer=norm_2d, activation_layer=activation_layer)

        self.res4 = ConvBottleNeckResidualBlock2d(64, 4, 128, kernel_size=3, stride=2,
                                                  norm_layer=norm_2d, activation_layer=activation_layer)

        self.res4_proj = ConvNormActivation2d(128, 128, kernel_size=3, stride=2,
                                            bias=False, norm_layer=None, activation_layer=None)
        self.xv4 = CrossViewAttention(128, 128, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo4 = MinimalCrossObjectEncoder(128, 128, k = 10, norm_layer=norm_1d,
                                            activation_layer=activation_layer, red_factor=2) if self.enable_xo else None

        self.res5 = ConvInvertedResidualBlock2d(128, 2, 256, kernel_size=3, stride=1,
                                                norm_layer=norm_2d, activation_layer=activation_layer)

        self.res6 = ConvInvertedResidualBlock2d(256, 2, 128, kernel_size=3, stride=2,
                                                norm_layer=norm_2d, activation_layer=activation_layer)

        self.res6_proj = ConvNormActivation2d(128, 128, kernel_size=3, stride=2,
                                               bias=False, norm_layer=None, activation_layer=None)
        self.xv6 = CrossViewAttention(128, 128, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo6 = MinimalCrossObjectEncoder(128, 128, k = 10, norm_layer=norm_1d,
                                            activation_layer=activation_layer, red_factor=2) if self.enable_xo else None

        self.res7 = ConvResidualBlock2d(128, 512, kernel_size=3, stride=2,
                                         norm_layer=norm_2d, activation_layer=activation_layer)

        self.xv7 = CrossViewAttention(512, 512, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo7 = MinimalCrossObjectEncoder(512, 512, k = 10, norm_layer=norm_1d,
                                            activation_layer=activation_layer, red_factor=2) if self.enable_xo else None

        self.linear8 = LinearNormActivation(512+128*3, 128, bias=True, norm_layer=norm_1d, activation_layer=activation_layer)
        self.projection_head = LinearNormActivation(128, self.out_dim, bias=True, norm_layer=None, activation_layer=None)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor, n_views: np.ndarray, n_nodes: np.ndarray | None = None) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2_p = self.conv2_proj(x2)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x4_p = self.res4_proj(x4)
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        x6_p = self.res6_proj(x6)
        x7 = self.res7(x6)

        x2_p, x4_p, x6_p, x7 = \
            self.max_pool(x2_p).flatten(1), self.max_pool(x4_p).flatten(1), \
            self.max_pool(x6_p).flatten(1), self.max_pool(x7).flatten(1)

        x2_p, x4_p, x6_p, x7 = \
            self.xv2(x2_p, n_views), self.xv4(x4_p, n_views), \
                self.xv6(x6_p, n_views), self.xv7(x7, n_views)

        if self.enable_xo and n_nodes is not None:
            assert(self.xo2 is not None and self.xo4 is not None and self.xo6 is not None and self.xo7 is not None)
            x2_p, x4_p, x6_p, x7 = \
                self.xo2(x2_p, n_nodes), self.xo4(x4_p, n_nodes), \
                    self.xo6(x6_p, n_nodes), self.xo7(x7, n_nodes)

        x = torch.cat([x2_p, x4_p, x6_p, x7], dim=1)
        x = self.linear8(x)

        x= self.projection_head(x)

        x = x/(x.norm(dim=1, keepdim=True) + self.eps)

        return x
