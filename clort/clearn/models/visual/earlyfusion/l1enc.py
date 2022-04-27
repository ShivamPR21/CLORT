from typing import Tuple

import torch
import torch.nn as nn
from moduleZoo import (
    ConvBottleNeckResidualBlock2d,
    ConvInvertedResidualBlock2d,
    ConvNormActivation2d,
    ConvResidualBlock2d,
    MultiHeadSelfAttention2d,
)


class Level1Encoder(nn.Module):

    def __init__(self, in_channels: int = 21, img_size: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()

        self.in_shape = img_size
        self.out_shape = img_size

        # Layer 1 : Conv2d
        self.conv_1 = ConvNormActivation2d(in_channels,
                                           32,
                                           kernel_size=5,
                                           stride=2,
                                           padding='stride_effective',
                                           activation_layer=nn.SELU)

        # Layer 2 : BottleNeckResidual2d
        self.bottle_neck_res_2 = ConvBottleNeckResidualBlock2d(32,
                                                                2,
                                                                kernel_size=5,
                                                                stride=1,
                                                                activation_layer=nn.SELU)

        # Layer 3 : InvertedResidual2d
        self.inverted_res_3 = ConvInvertedResidualBlock2d(32,
                                                  expansion_ratio=2,
                                                  kernel_size=5,
                                                  stride=1,
                                                  activation_layer=nn.SELU)

        # Layer 4 : Residual2d
        self.res_4 = ConvResidualBlock2d(32,
                                         64,
                                         kernel_size=5,
                                         stride=2,
                                         activation_layer=nn.SELU)

        # Layer 5 : BottleNeckResidual2d
        self.bottle_neck_res_5 = ConvBottleNeckResidualBlock2d(64,
                                                                4,
                                                                kernel_size=3,
                                                                stride=1,
                                                                activation_layer=nn.SELU)

        # Layer 6 : InvertedResidual2d
        self.inverted_res_6 = ConvInvertedResidualBlock2d(64,
                                                  expansion_ratio=2,
                                                  kernel_size=3,
                                                  stride=1,
                                                  activation_layer=nn.SELU)

        # Layer 7 : BottleNeckResidual2d
        self.bottle_neck_res_7 = ConvBottleNeckResidualBlock2d(64,
                                                                4,
                                                                kernel_size=3,
                                                                stride=1,
                                                                activation_layer=nn.SELU)

        # Layer 8 : InvertedResidual2d
        self.inverted_res_8 = ConvInvertedResidualBlock2d(64,
                                                  expansion_ratio=2,
                                                  kernel_size=3,
                                                  stride=1,
                                                  norm_layer=nn.BatchNorm2d, # Inserting Batchnorm
                                                  activation_layer=nn.SELU)

        # Layer 9 : Residual2d
        self.res_9 = ConvResidualBlock2d(64,
                                         128,
                                         kernel_size=3,
                                         stride=2,
                                         activation_layer=nn.SELU)

        # Layer 10 : BottleNeckResidual2d
        self.bottle_neck_res_10 = ConvBottleNeckResidualBlock2d(128,
                                                                8,
                                                                kernel_size=3,
                                                                stride=1,
                                                                activation_layer=nn.SELU)

        # Layer 11 : InvertedResidual2d
        self.inverted_res_11 = ConvInvertedResidualBlock2d(128,
                                                  expansion_ratio=2,
                                                  kernel_size=3,
                                                  stride=1,
                                                  activation_layer=nn.SELU)

        # Layer 12 : BottleNeckResidual2d
        self.bottle_neck_res_12 = ConvBottleNeckResidualBlock2d(128,
                                                                8,
                                                                kernel_size=3,
                                                                stride=1,
                                                                activation_layer=nn.SELU)

        # Layer 13 : InvertedResidual2d
        self.inverted_res_13 = ConvInvertedResidualBlock2d(128,
                                                  expansion_ratio=2,
                                                  kernel_size=3,
                                                  stride=1,
                                                  activation_layer=nn.SELU)

        # Layer 14 : MultiHeadSelfAttention2d
        self.multi_head_self_attention_2d = MultiHeadSelfAttention2d(128,
                                                                     n_heads=4,
                                                                     kernel_size=3)

        # Layer 15 : InvertedResidual2d
        self.inverted_res_15 = ConvInvertedResidualBlock2d(128*4,
                                                           expansion_ratio=2,
                                                           kernel_size=3,
                                                           stride=2,
                                                           activation_layer=nn.SELU)

        # Layer 16 : InvertedResidual2d
        self.inverted_res_16 = ConvInvertedResidualBlock2d(512,
                                                           expansion_ratio=2,
                                                           kernel_size=3,
                                                           stride=1,
                                                           activation_layer=nn.SELU)

        # Layer 17 : MaxPool2d
        self.max_pool_2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.bottle_neck_res_2(x)
        x = self.inverted_res_3(x)

        x = self.res_4(x)

        x = self.bottle_neck_res_5(x)
        x = self.inverted_res_6(x)

        x = self.bottle_neck_res_7(x)
        x = self.inverted_res_8(x)

        x = self.res_9(x)

        x = self.bottle_neck_res_10(x)
        x = self.inverted_res_11(x)

        x = self.bottle_neck_res_12(x)
        x = self.inverted_res_13(x)

        x = self.multi_head_self_attention_2d(x)

        x = self.inverted_res_15(x)
        x = self.inverted_res_16(x)

        x = self.max_pool_2d(x)

        x = x.flatten(start_dim=1)

        return x
