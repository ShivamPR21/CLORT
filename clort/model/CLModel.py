from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoders import (
    CrossObjectEncoder,
    DLA34EncoderMV,
    MultiModalEncoder,
    MultiViewEncoder,
    PointCloudEncoder,
)


class CLModel(nn.Module):

    def __init__(self, mv_backbone: str = 'small', mv_features: int | None = None, mv_xo: bool = False,
                 pc_features: int | None = None, bbox_aug: bool = True, pc_xo: bool = False,
                 mm_features: int | None = None, mm_xo: bool = False,
                 mmc_features: int | None = None, mode: str = 'train' # 'infer'
                 ) -> None:
        super().__init__()


        print(f'Model Config: {mv_features = } \t {mv_xo = } \t {pc_features = } \t {bbox_aug = } \n'
              f'{pc_xo = } \t {mm_features = } \t {mm_xo = } \t {mmc_features = }')

        norm_2d, norm_1d, act = nn.InstanceNorm2d, nn.LayerNorm, nn.SELU

        self.mv_enc = None
        if mv_backbone == 'dla':
            norm_2d, norm_1d, act = nn.BatchNorm2d, nn.BatchNorm1d, nn.ReLU
            self.mv_enc = DLA34EncoderMV(out_dim=mv_features, enable_mv=True, enable_xo=mv_xo,
                                         features_only=mm_features is not None or mmc_features is not None or (mode == 'infer')) if mv_features is not None else None
        else:
            assert(mv_backbone in ['small', 'medium', 'large'])
            self.mv_enc = MultiViewEncoder(out_dim=mv_features,
                                            norm_2d=norm_2d,
                                            norm_1d=norm_1d,
                                            activation_layer=act,
                                            enable_xo=mv_xo,
                                            features_only=mm_features is not None or mmc_features is not None or (mode == 'infer'),
                                            size=mv_backbone) if mv_features is not None else None

        self.pc_enc = PointCloudEncoder(out_dims=pc_features, bbox_aug=bbox_aug,
                                        norm_layer=norm_1d, activation_layer=act,
                                        offloading=False, enable_xo=pc_xo,
                                        features_only=mm_features is not None or mmc_features is not None or (mode == 'infer')) if pc_features is not None else None

        mv_features = self.mv_enc.out_dim if self.mv_enc is not None else mv_features
        pc_features = self.pc_enc.out_dim if self.pc_enc is not None else pc_features

        self.mm_enc = MultiModalEncoder(mv_features, pc_features, mm_features, mm_features, norm_layer=norm_1d,
                                        activation_layer=act, enable_xo=mm_xo,
                                        features_only=mmc_features is not None or (mode == 'infer')) if (mv_features is not None and pc_features is not None and mm_features is not None) else None

        mm_features = self.mm_enc.out_dim if self.mm_enc is not None else mm_features

        self.mmc_enc = CrossObjectEncoder(mm_features, mmc_features, norm_layer=norm_1d,
                                          activation_layer=act, features_only=(mode == 'infer')) if (mm_features is not None and mmc_features is not None) else None

        mmc_features = self.mmc_enc.out_dim if self.mmc_enc is not None else None

        print(f'Final model Config: {mv_features = } \t {mv_xo = } \t {pc_features = } \t {bbox_aug = } \n'
              f'{pc_xo = } \t {mm_features = } \t {mm_xo = } \t {mmc_features = }')

        self.out_dim: int | None = None
        if mmc_features is not None:
            self.out_dim = mmc_features
        elif mm_features is not None:
            self.out_dim = mm_features
        elif mv_features is not None:
            self.out_dim = mv_features
        else:
            raise NotImplementedError("Encoder resolution failed.")
        print(f'Model Out Dims: {self.out_dim = }')

        self.eps = 1e-9

    def forward(self, pcls: torch.Tensor | List[Any], pcls_sz: np.ndarray | List[Any],
                imgs: torch.Tensor | List[Any], imgs_sz: torch.Tensor | List[Any],
                bboxs: torch.Tensor | List[Any], frame_sz: np.ndarray) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        mv_e = self.mv_enc(imgs, imgs_sz, frame_sz) if self.mv_enc is not None else None
        pc_e = self.pc_enc(pcls, pcls_sz, frame_sz, bboxs) if self.pc_enc is not None else None

        mm_e = self.mm_enc(mv_e, pc_e, frame_sz) if self.mm_enc is not None else None

        mmc_e = self.mmc_enc(mm_e, frame_sz) if (self.mmc_enc is not None and mm_e is not None) else None

        mv_e = mv_e/(mv_e.norm(dim=1, keepdim=True) + self.eps) if mv_e is not None else None
        pc_e = pc_e/(pc_e.norm(dim=1, keepdim=True) + self.eps) if pc_e is not None else None
        mm_e = mm_e/(mm_e.norm(dim=1, keepdim=True) + self.eps) if mm_e is not None else None
        mmc_e = mmc_e/(mmc_e.norm(dim=1, keepdim=True) + self.eps) if mmc_e is not None else None

        return mv_e, pc_e, mm_e, mmc_e
