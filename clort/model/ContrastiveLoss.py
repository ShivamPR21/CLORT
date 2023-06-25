from typing import List

import numpy as np
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self,
                 temp: float = 0.3,
                 eps: float = 1e-9,
                 local_contrast: bool = True,
                 separate_tracks: bool = False,
                 static_contrast: bool = False,
                 use_hard_condition: bool = False,
                 hard_condition_proportion: float = 0.5,
                 shift_to_positive: bool = False) -> None:
        super().__init__()

        self.temp = temp
        self.eps = eps
        self.local_contrast = local_contrast
        self.separate_tracks = separate_tracks

        self.stc = static_contrast

        self.hc, self.hcp = use_hard_condition, hard_condition_proportion

        self.stp = shift_to_positive

        self.eps2 = 1e-10

        self.min, self.max = np.exp(-1./self.temp), np.exp(1./self.temp)

    def loss(self, num:torch.Tensor, den:torch.Tensor, n_pos:torch.Tensor, n_neg:torch.Tensor) -> torch.Tensor:
        loss : torch.Tensor | None = None

        if not self.hc:
            loss = -(num/(den+num+self.eps)+self.eps2).log()
            if self.stp:
                loss = loss + ((n_pos*self.max/(n_neg*self.min + n_pos*self.max + self.eps))+self.eps2).log()
        else:
            loss = (self.hcp*num + (den+num+self.eps).log())/(self.hcp+1.)
            if self.stp:
                loss = loss - (n_neg*self.min + n_pos*self.max + self.eps2).log()/(self.hcp+1.)

        return loss

    def forward(self, x: torch.Tensor, track_idxs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x -> [n_obj', N]
        # y -> [n_obj, Q, N]
        # print(f'{x.shape = } \t {y.shape = }')

        if (y.device != x.device):
            y = y.to(x.device)

        ut_ids = track_idxs.unique()

        y_idxs : torch.Tensor | None = None

        if self.local_contrast:
            y = y[ut_ids, :, :]

            _, Q, _ = y.size()
            y = y.flatten(0, 1)

            y_idxs = ut_ids.repeat(Q)
        else:
            n, Q, _ = y.size()
            y = y.flatten(0, 1)

            y_idxs = torch.arange(n, dtype=torch.int32).repeat(Q)

        num, den = torch.zeros(1, dtype=torch.float32, device=x.device), torch.zeros(1, dtype=torch.float32, device=x.device)
        n_pos, n_neg = \
            torch.tensor([0], dtype=torch.float32, device=x.device, requires_grad=False),\
                  torch.tensor([0], dtype=torch.float32, device=x.device, requires_grad=False)

        loss: torch.Tensor | List[List[torch.Tensor]] | None = [] if self.separate_tracks else None

        for uid in ut_ids:
            x_map = track_idxs == uid
            y_map = y_idxs == uid

            x_pos = x[x_map, :]
            y_pos = y[y_map, :]

            x_pos, y_pos = x_pos.unsqueeze(dim=1), y_pos.unsqueeze(dim=0)

            if not self.hc:
                num = num + ((x_pos * y_pos).sum(dim=-1)/self.temp).exp().sum()
            else:
                num = num + ((x_pos - y_pos).norm(dim=-1)).sum()

            n_pos += x_pos.shape[0] * y_pos.shape[1]

            if self.stc:
                tmp = 0
                if not self.hc:
                    tmp = ((x_pos * x_pos.transpose(0, 1)).sum(dim=-1)/self.temp).exp()
                else:
                    tmp = ((x_pos - x_pos.transpose(0, 1)).norm(dim=-1))

                tmp = tmp * (1.0 - torch.eye(tmp.shape[0], dtype=torch.float32, device=x.device))
                num = num + tmp.sum()/2.
                n_pos += (x_pos.shape[0]*(x_pos.shape[0] - 1)/2.)

            x_neg = x[~x_map, :]
            y_neg = y[~y_map, :]

            x_neg, y_neg = x_neg.unsqueeze(dim=0), y_neg.unsqueeze(dim=0)

            den = den + ((x_pos * y_neg).sum(dim=-1)/self.temp).exp().sum()
            n_neg += x_neg.shape[0] * y_neg.shape[1]

            if self.stc:
                den = den + ((x_pos * x_neg).sum(dim=-1)/self.temp).exp().sum()
                n_neg += x_pos.shape[0] * x_neg.shape[0]

            if self.separate_tracks:
                loss.append([num, den, n_pos, n_neg])
                n_pos, n_neg, num, den = 0, 0, 0, 0

        if self.separate_tracks:
            loss_ = torch.tensor([0], dtype=torch.float32, device=x.device)
            for num, den, n_pos, n_neg in loss:
                loss_ = loss_ + self.loss(num, den, n_pos, n_neg)
            loss = loss_/len(ut_ids)
        else:
            loss = self.loss(num, den, n_pos, n_neg)

        return loss
