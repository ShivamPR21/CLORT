from typing import Type

import torch
import torch.nn as nn
from moduleZoo.attention import SelfAttentionLinear


class MultiViewEncoder(nn.Module):

    def __init__(self, sv_enc: Type[nn.Module]) -> None:
        super().__init__()

        self.sv_enc = sv_enc

        # self.gat =
