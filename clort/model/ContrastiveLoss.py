from typing import Callable, List

import torch
import torch.nn as nn


def dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sim = x @ y.T
    return sim

class ContrastiveLoss(nn.Module):

    def __init__(self,
                 temp: float = 0.3,
                 local_contrast: bool = True,
                 separate_tracks: bool = False,
                 static_contrast: bool = False,
                 use_hard_condition: bool = False,
                 hard_condition_proportion: float = 0.5,
                 sim_type: str = "dot", #"dot/diff" #TODO@ShivamPR21 Ad-hoc short term inplace remedy until sim-fxn is implemented
                 sim_fxn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = dot_similarity) -> None:
        super().__init__()

        self.temp = temp
        self.local_contrast = local_contrast
        self.separate_tracks = separate_tracks

        self.stc = static_contrast

        self.hc, self.hcp = use_hard_condition, hard_condition_proportion

        self.sim_type = sim_type #TODO@ShivamPR21 Ad-hoc short term inplace remedy until sim-fxn is implemented

        self.sim_fxn = sim_fxn #TODO@ShivamPR21 : Upgrade to Similarity function based disparity measure

    def loss(self, num:torch.Tensor, den:torch.Tensor) -> torch.Tensor:
        loss : torch.Tensor | None = None

        if not self.hc:
            loss = -(num/(den+num)).log()
        else:
            loss = (self.hcp*num + (den+num).log())/(self.hcp+1.)

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
                if self.sim_type == "dot":
                    num = num + ((x_pos * y_pos).sum(dim=-1)/self.temp).exp().sum()
                elif self.sim_type == "diff":
                    num = num + (-((x_pos - y_pos).norm(dim=-1))/self.temp).exp().sum()
                else:
                    raise NotImplementedError(f'Similarity Type: {self.sim_type} not implemented')
            else:
                num = num + ((x_pos - y_pos).norm(dim=-1)).mean() # Hard constraint numerator is added to
                                                                # loss thus needs to mean instead of sum

            n_pos += x_pos.shape[0] * y_pos.shape[1]

            if self.stc:
                tmp = 0
                if not self.hc:
                    if self.sim_type == "dot":
                        tmp = ((x_pos * x_pos.transpose(0, 1)).sum(dim=-1)/self.temp).exp()
                    elif self.sim_type == "diff":
                        tmp = (-((x_pos - x_pos.transpose(0, 1)).norm(dim=-1))/self.temp).exp()
                    else:
                        raise NotImplementedError(f'Similarity Type: {self.sim_type} not implemented')
                else:
                    tmp = ((x_pos - x_pos.transpose(0, 1)).norm(dim=-1))

                tmp = tmp * (1.0 - torch.eye(tmp.shape[0], dtype=torch.float32, device=x.device))
                num = num + (tmp.sum()/2. if not self.hc else tmp.mean()) # Hard constraint numerator is added to
                                                                # loss thus needs to mean instead of sum
                n_pos += (x_pos.shape[0]*(x_pos.shape[0] - 1)/2.)

            x_neg = x[~x_map, :]
            y_neg = y[~y_map, :]

            x_neg, y_neg = x_neg.unsqueeze(dim=0), y_neg.unsqueeze(dim=0)

            if self.sim_type == "dot":
                den = den + ((x_pos * y_neg).sum(dim=-1)/self.temp).exp().sum()
            elif self.sim_type == "diff":
                den = den + (- ((x_pos - y_neg).norm(dim=-1))/self.temp).exp().sum()
            else:
                raise NotImplementedError(f'Similarity Type: {self.sim_type} not implemented')
            n_neg += x_neg.shape[0] * y_neg.shape[1]

            if self.stc:
                if self.sim_type == "dot":
                    den = den + ((x_pos * x_neg).sum(dim=-1)/self.temp).exp().sum()
                elif self.sim_type == "diff":
                    den = den + (-((x_pos - x_neg).norm(dim=-1))/self.temp).exp().sum()
                else:
                    raise NotImplementedError(f'Similarity Type: {self.sim_type} not implemented')
                n_neg += x_pos.shape[0] * x_neg.shape[0]

            if self.separate_tracks:
                loss.append([num, den, n_pos, n_neg])
                n_pos, n_neg, num, den = 0, 0, 0, 0

        if self.separate_tracks:
            loss_ = torch.tensor([0], dtype=torch.float32, device=x.device)
            for num, den, _, _ in loss:
                loss_ = loss_ + self.loss(num, den)
            loss = loss_/len(ut_ids)
        else:
            loss = self.loss(num, den)

        return loss
