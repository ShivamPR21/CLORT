from typing import Callable, Tuple, Type

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


class SingleViewEncoderDLA34(nn.Module):

    def __init__(self, image_shape: Tuple[int, int] = (224, 224), in_dim:int = 3, out_dim: int = 256,
                 norm_layer : Callable[..., nn.Module] | None=None) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.in_dim, self.out_dim = in_dim, out_dim

        self.dla34 = create_model('dla34', features_only=True, pretrained=True)

        self.conv1 = ConvNormActivation2d(64, 64, kernel_size=8, stride=8,
                                          padding="stride_effective",
                                          norm_layer=norm_layer, activation_layer=nn.SELU)

        self.conv2 = ConvNormActivation2d(128, 128, kernel_size=4, stride=4,
                                          padding="stride_effective",
                                          norm_layer=norm_layer, activation_layer=nn.SELU)

        self.conv3 = ConvNormActivation2d(256, 256, kernel_size=2, stride=2,
                                        padding="stride_effective",
                                        norm_layer=norm_layer, activation_layer=nn.SELU)

        self.conv4 = ConvNormActivation2d(512, 512, kernel_size=1, stride=1,
                                          padding="stride_effective",
                                          norm_layer = norm_layer, activation_layer=nn.SELU)

        self.linear5 = LinearNormActivation(512*2, self.out_dim, bias=True)

        self.max_pool = nn.MaxPool2d((7, 7))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_ = self.dla34(x)
        x = torch.cat([self.conv1(x_[1]), self.conv2(x_[2]), self.conv3(x_[3]), self.conv4(x_[4])], dim=1)

        x = self.linear5(self.max_pool(x).flatten(1))

        return x

class SingleViewEncoder(nn.Module):

    def __init__(self, image_shape: Tuple[int, int] = (224, 224), in_dim:int = 3, out_dim: int = 256,
                 norm_layer : Callable[..., nn.Module] | None=None) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.in_dim, self.out_dim = in_dim, out_dim

        self.conv1 = ConvNormActivation2d(self.in_dim, 64, kernel_size=5, stride=2,
                                          norm_layer=norm_layer, activation_layer=nn.SELU)

        self.conv2 = ConvNormActivation2d(64, 128, kernel_size=5, stride=2,
                                          norm_layer=norm_layer, activation_layer=nn.SELU)

        self.conv2_proj = ConvNormActivation2d(128, 128, kernel_size=5, stride=5,
                                          norm_layer=None, activation_layer=None)

        self.res3 = ConvBottleNeckResidualBlock2d(128, 4, 64, kernel_size=3, stride=1,
                                                  norm_layer=norm_layer, activation_layer=nn.SELU)

        self.res4 = ConvBottleNeckResidualBlock2d(64, 4, 128, kernel_size=3, stride=2,
                                                  norm_layer=norm_layer, activation_layer=nn.SELU)

        self.res4_proj = ConvNormActivation2d(128, 128, kernel_size=3, stride=2,
                                          norm_layer=None, activation_layer=None)

        self.res5 = ConvInvertedResidualBlock2d(128, 2, 256, kernel_size=3, stride=1,
                                                norm_layer=norm_layer, activation_layer=nn.SELU)

        self.res6 = ConvInvertedResidualBlock2d(256, 2, 128, kernel_size=3, stride=2,
                                                norm_layer=norm_layer, activation_layer=nn.SELU)

        self.res6_proj = ConvNormActivation2d(128, 128, kernel_size=3, stride=2,
                                            norm_layer=None, activation_layer=None)

        self.res7 = ConvResidualBlock2d(128, 512, kernel_size=3, stride=2,
                                         norm_layer=norm_layer, activation_layer=nn.SELU)

        self.linear8 = LinearNormActivation(512+128*3, self.out_dim, bias=True)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        x = torch.cat([self.max_pool(x2_p), self.max_pool(x4_p), self.max_pool(x6_p), self.max_pool(x7)], dim=1).flatten(1)
        x = self.linear8(x)

        return x

class MultiViewEncoder(nn.Module):

    def __init__(self,
                 image_shape : Tuple[int, int] = (224, 224), out_dim: int = 256,
                 norm_2d: Callable[..., nn.Module] | None = None,
                 norm_1d: Callable[..., nn.Module] | None = None,
                 sv_backbone: Type[nn.Module] | None = None) -> None:
        super().__init__()

        self.eps = 1e-9
        self.image_shape = image_shape

        self.sv_enc1 = SingleViewEncoder(self.image_shape, 3, 512, norm_layer=norm_2d) if sv_backbone is None else sv_backbone

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.sv_enc2 = LinearNormActivation(512, 256, norm_layer=norm_1d, activation_layer=nn.SELU)
        self.sv_enc3 = LinearNormActivation(256, 128, activation_layer=nn.SELU)

        self.gat = SelfGraphAttentionLinear(128, None, residual=True, dynamic_batching=True)
        self.gat_act = nn.SELU()
        self.projection_head = LinearNormActivation(128, out_dim, activation_layer=None)

    def forward(self, x : torch.Tensor, n_views: np.ndarray) -> torch.Tensor:
        # x -> [N, 3, W, H]
        # n_views -> [nv....]

        # Atomic encodings
        # x = self.max_pool(self.sv_enc1(x)[-1]).flatten(start_dim=1) # [N, 512]
        x = self.sv_enc1(x) # [N, 512, n, n] or [N, 512]

        if not x.ndim == 2:
            x = self.max_pool(x).flatten(start_dim=1) # [N, 512]

        x = self.sv_enc3(self.sv_enc2(x)) # [N, 128]

        x = self.gat_act(self.gat(x, n_views)) # [N, 128]

        sz_arr = [_.numpy().tolist() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(n_views.tolist(), dim=0)]

        output = torch.zeros((len(sz_arr), 128), dtype=torch.float32, device=x.device)

        for i, sz_idxs in enumerate(sz_arr):
            output[i, :] = x[sz_idxs, :].max(dim=0).values

        output = self.projection_head(output)

        output = output/(output.norm(dim=1, keepdim=True) + self.eps)

        return output
