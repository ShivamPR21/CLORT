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
from timm import create_model

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


class DLA34Encoder(nn.Module):

    def __init__(self, img_shape = (128, 128), out_dim: int = 256) -> None:
        super().__init__()

        self.enc = create_model('dla34', pretrained=False, features_only=True)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.linear1 = LinearNormActivation(32+64+128+256+512, 512,
                                            norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU)

        self.linear2 = LinearNormActivation(512, out_dim,
                                            norm_layer=None, activation_layer=None)

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, img_sz: np.ndarray) -> torch.Tensor:
        enc1, enc2, enc3, enc4, enc5 = self.enc(x)
        enc1, enc2, enc3, enc4, enc5 = self.max_pool(enc1).flatten(1), self.max_pool(enc2).flatten(1), \
            self.max_pool(enc3).flatten(1), self.max_pool(enc4).flatten(1), self.max_pool(enc5).flatten(1)

        enc = torch.cat([enc1, enc2, enc3, enc4, enc5], dim=1)

        enc = self.linear1(enc)
        enc = self.linear2(enc)

        enc = torch.cat([spl.max(dim=0, keepdim=True).values for spl in enc.split(img_sz.tolist(), dim=0)], dim=0)

        enc = enc/(enc.norm(dim=1, keepdim=True)+1e-9)

        return enc

class DLA34EncoderMV(DLA34Encoder):

    def __init__(self, img_shape=(128, 128), out_dim: int = 256, enable_mv: bool = True, enable_xo: bool = True, features_only: bool = False) -> None:
        super().__init__(img_shape, out_dim)

        self.features_only = features_only

        self.xv1 = CrossViewAttention(32, 32, residual=True, dynamic_batching=True,
                                      norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU) if enable_mv else None
        self.xo1 = MinimalCrossObjectEncoder(32, 32, k = 10, norm_layer=nn.BatchNorm1d,
                                             activation_layer=nn.ReLU, red_factor=2) if enable_xo else None

        self.xv2 = CrossViewAttention(64, 64, residual=True, dynamic_batching=True,
                                      norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU) if enable_mv else None
        self.xo2 = MinimalCrossObjectEncoder(64, 64, k = 10, norm_layer=nn.BatchNorm1d,
                                             activation_layer=nn.ReLU, red_factor=2) if enable_xo else None

        self.xv3 = CrossViewAttention(128, 128, residual=True, dynamic_batching=True,
                                      norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU) if enable_mv else None
        self.xo3 = MinimalCrossObjectEncoder(128, 128, k = 10, norm_layer=nn.BatchNorm1d,
                                             activation_layer=nn.ReLU, red_factor=2) if enable_xo else None

        self.xv4 = CrossViewAttention(256, 256, residual=True, dynamic_batching=True,
                                      norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU) if enable_mv else None
        self.xo4 = MinimalCrossObjectEncoder(256, 256, k = 10, norm_layer=nn.BatchNorm1d,
                                             activation_layer=nn.ReLU, red_factor=2) if enable_xo else None

        self.xv5 = CrossViewAttention(512, 512, residual=True, dynamic_batching=True,
                                      norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU) if enable_mv else None
        self.xo5 = MinimalCrossObjectEncoder(512, 512, k = 10, norm_layer=nn.BatchNorm1d,
                                             activation_layer=nn.ReLU, red_factor=2) if enable_xo else None

        self.out_dim = 512 if self.features_only else out_dim

    def forward(self, x: torch.Tensor, img_sz: np.ndarray, n_nodes: np.ndarray | None = None) -> torch.Tensor:
        enc1, enc2, enc3, enc4, enc5 = self.enc(x)
        enc1, enc2, enc3, enc4, enc5 = self.max_pool(enc1).flatten(1), self.max_pool(enc2).flatten(1), \
            self.max_pool(enc3).flatten(1), self.max_pool(enc4).flatten(1), self.max_pool(enc5).flatten(1)

        if (self.xv1 is not None and self.xv2 is not None and \
            self.xv3 is not None and self.xv4 is not None and self.xv5 is not None):
            enc1, enc2, enc3, enc4, enc5 = self.xv1(enc1, img_sz), self.xv2(enc2, img_sz), \
                self.xv3(enc3, img_sz), self.xv4(enc4, img_sz), self.xv5(enc5, img_sz)

        if (self.xo1 is not None and self.xo2 is not None and \
            self.xo3 is not None and self.xo4 is not None and self.xo5 is not None) and n_nodes is not None:
                enc1, enc2, enc3, enc4, enc5 = self.xo1(enc1, n_nodes), self.xo2(enc2, n_nodes), \
                    self.xo3(enc3, n_nodes), self.xo4(enc4, n_nodes), self.xo5(enc5, n_nodes)

        enc = torch.cat([enc1, enc2, enc3, enc4, enc5], dim=1)

        enc = self.linear1(enc)

        if not self.features_only:
            enc = self.linear2(enc)

            enc = enc/(enc.norm(dim=1, keepdim=True)+1e-9)

        return enc


class MultiViewEncoder(nn.Module):

    def __init__(self, image_shape: Tuple[int, int] = (224, 224), in_dim:int = 3, out_dim: int = 256,
                 norm_2d : Callable[..., nn.Module] | None=None,
                 norm_1d : Callable[..., nn.Module] | None=None,
                 activation_layer: Callable[..., nn.Module] | None=None,
                 enable_xo: bool = False,
                 features_only: bool = False,
                 size: str = 'small') -> None:
        super().__init__()

        assert(size in ['small', 'medium', 'large'])

        self.eps = 1e-9

        self.image_shape = image_shape
        self.in_dim, self.out_dim = in_dim, out_dim
        self.enable_xo = enable_xo
        self.features_only = features_only
        self.medium = (size in ['medium', 'large'])
        self.large = (size == 'large')

        self.conv1 = ConvNormActivation2d(self.in_dim, 64, kernel_size=5, stride=2,
                                          norm_layer=norm_2d, activation_layer=activation_layer)

        self.conv2 = ConvNormActivation2d(64, 128, kernel_size=5, stride=2,
                                          norm_layer=norm_2d, activation_layer=activation_layer)
        self.res2_m = ConvResidualBlock2d(128, 128, kernel_size=5, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.medium else None
        self.res2_l = ConvResidualBlock2d(128, 128, kernel_size=5, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.large else None

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
        self.res4_m = ConvResidualBlock2d(128, 128, kernel_size=3, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.medium else None
        self.res4_l = ConvResidualBlock2d(128, 128, kernel_size=3, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.large else None

        self.res4_proj = ConvNormActivation2d(128, 128, kernel_size=3, stride=2,
                                            bias=False, norm_layer=None, activation_layer=None)
        self.xv4 = CrossViewAttention(128, 128, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo4 = MinimalCrossObjectEncoder(128, 128, k = 10, norm_layer=norm_1d,
                                            activation_layer=activation_layer, red_factor=2) if self.enable_xo else None

        self.res5 = ConvInvertedResidualBlock2d(128, 2, 256, kernel_size=3, stride=1,
                                                norm_layer=norm_2d, activation_layer=activation_layer)

        self.res6 = ConvInvertedResidualBlock2d(256, 2, 256, kernel_size=3, stride=2,
                                                norm_layer=norm_2d, activation_layer=activation_layer)
        self.res6_m = ConvResidualBlock2d(256, 256, kernel_size=3, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.medium else None
        self.res6_l = ConvResidualBlock2d(256, 256, kernel_size=3, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.large else None

        self.res6_proj = ConvNormActivation2d(256, 256, kernel_size=3, stride=2,
                                               bias=False, norm_layer=None, activation_layer=None)
        self.xv6 = CrossViewAttention(256, 256, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo6 = MinimalCrossObjectEncoder(256, 256, k = 10, norm_layer=norm_1d,
                                            activation_layer=activation_layer, red_factor=2) if self.enable_xo else None

        self.res7 = ConvResidualBlock2d(256, 512, kernel_size=3, stride=2,
                                         norm_layer=norm_2d, activation_layer=activation_layer)
        self.res7_m = ConvResidualBlock2d(512, 512, kernel_size=3, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.medium else None
        self.res7_l = ConvResidualBlock2d(512, 512, kernel_size=3, stride=1,
                                            norm_layer=norm_2d, activation_layer=activation_layer) if self.large else None

        self.xv7 = CrossViewAttention(512, 512, residual=True, dynamic_batching=True,
                                      norm_layer=norm_1d, activation_layer=activation_layer)
        self.xo7 = MinimalCrossObjectEncoder(512, 512, k = 10, norm_layer=norm_1d,
                                            activation_layer=activation_layer, red_factor=2) if self.enable_xo else None

        self.linear8 = LinearNormActivation(512+256*2, 512, bias=True, norm_layer=norm_1d, activation_layer=activation_layer)
        self.projection_head = LinearNormActivation(512, self.out_dim, bias=True, norm_layer=None, activation_layer=None)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.out_dim = 512 if self.features_only else self.out_dim

    def forward(self, x: torch.Tensor, n_views: np.ndarray, n_nodes: np.ndarray | None = None) -> torch.Tensor:
        x1 = self.conv1(x)

        # Level 2
        x2 = self.conv2(x1)
        if self.res2_m is not None:
            x2 = self.res2_m(x2)
        if self.res2_l is not None:
            x2 = self.res2_l(x2)
        x2_p = self.conv2_proj(x2)

        x3 = self.res3(x2)

        # Level 4
        x4 = self.res4(x3)
        if self.res4_m is not None:
            x4 = self.res4_m(x4)
        if self.res4_l is not None:
            x4 = self.res4_l(x4)
        x4_p = self.res4_proj(x4)

        x5 = self.res5(x4)

        x6 = self.res6(x5)
        if self.res6_m is not None:
            x6 = self.res6_m(x6)
        if self.res6_l is not None:
            x6 = self.res6_l(x6)
        x6_p = self.res6_proj(x6)

        x7 = self.res7(x6)
        if self.res7_m is not None:
            x7 = self.res7_m(x7)
        if self.res7_l is not None:
            x7 = self.res7_l(x7)

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

        if not self.features_only:
            x = self.projection_head(x)

            x = x/(x.norm(dim=1, keepdim=True) + self.eps)

        return x
