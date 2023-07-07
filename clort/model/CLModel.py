from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoders import (
    CrossObjectEncoder,
    MultiModalEncoder,
    MultiViewEncoder,
    PointCloudEncoder,
)


class CLModel(nn.Module):

    def __init__(self, mv_features: int | None = None, mv_xo: bool = False,
                 pc_features: int | None = None, bbox_aug: bool = True, pc_xo: bool = False,
                 mm_features: int | None = None, mm_xo: bool = False,
                 mmc_features: int | None = None) -> None:
        super().__init__()

        self.out_dim: int | None = None
        if mmc_features is not None:
            self.out_dim = mmc_features
        elif mm_features is not None:
            self.out_dim = mm_features
        elif pc_features is not None:
            self.out_dim = pc_features
        elif mv_features is not None:
            self.out_dim = mv_features
        else:
            raise NotImplementedError("Encoder resolution failed.")

        print(f'Model Config: {mv_features = } \t {mv_xo = } \t {pc_features = } \t {bbox_aug = } \n'
              f'{pc_xo = } \t {mm_features = } \t {mm_xo = } \t {mmc_features = }')
        print(f'Model Out Dims: {self.out_dim = }')

        self.mv_enc = MultiViewEncoder(out_dim=mv_features,
                                       norm_2d=nn.InstanceNorm2d,
                                       norm_1d=nn.LayerNorm,
                                       enable_xo=mv_xo) if mv_features is not None else None

        self.pc_enc = PointCloudEncoder(out_dims=pc_features, bbox_aug=bbox_aug,
                                        norm_layer=nn.LayerNorm, activation_layer=nn.SELU,
                                        offloading=False, enable_xo=pc_xo) if pc_features is not None else None

        self.mm_enc = MultiModalEncoder(mv_features, pc_features, mm_features, norm_layer=nn.LayerNorm,
                                        activation_layer=nn.SELU, enable_xo=mm_xo) if (mv_features is not None and pc_features is not None and mm_features is not None) else None

        self.mmc_enc = CrossObjectEncoder(mm_features, mmc_features, norm_layer=nn.LayerNorm,
                                          activation_layer=nn.SELU) if (mm_features is not None and mmc_features is not None) else None

    def forward(self, pcls: torch.Tensor | List[Any], pcls_sz: np.ndarray | List[Any],
                imgs: torch.Tensor | List[Any], imgs_sz: torch.Tensor | List[Any],
                bboxs: torch.Tensor | List[Any], frame_sz: np.ndarray) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        mv_e = self.mv_enc(imgs, imgs_sz, frame_sz) if self.mv_enc is not None else None
        pc_e = self.pc_enc(pcls, pcls_sz, frame_sz, bboxs) if self.pc_enc is not None else None

        mm_e = self.mm_enc(mv_e, pc_e) if self.mm_enc is not None else None

        mmc_e = self.mmc_enc(mm_e, frame_sz) if (self.mmc_enc is not None and mm_e is not None) else None

        return mv_e, pc_e, mm_e, mmc_e