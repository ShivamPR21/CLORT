import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self,
                 temp: float = 0.3,
                 eps: float = 1e-9,
                 static_contrast: bool = False) -> None:
        super().__init__()

        self.temp = temp
        self.eps = eps
        self.stc = static_contrast
        self.eps2 = 1e-10

    def forward(self, x: torch.Tensor, track_idxs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x -> [n_obj', N]
        # y -> [n_obj, Q, N]
        # print(f'{x.shape = } \t {y.shape = }')

        if (y.device != x.device):
            y = y.to(x.device)

        ut_ids = track_idxs.unique()

        y = y[ut_ids, :, :]

        # print(f'{y.shape = }')

        _, Q, _ = y.size()
        y = y.flatten(0, 1)

        y_idxs = ut_ids.repeat(Q)

        num, den = torch.zeros(1, device=x.device), torch.zeros(1, device=x.device)

        # print(f'{y_idxs.shape = }')
        for uid in ut_ids:
            x_map = track_idxs == uid
            y_map = y_idxs == uid

            x_pos = x[x_map, :]
            y_pos = y[y_map, :]

            x_pos, y_pos = x_pos.unsqueeze(dim=1), y_pos.unsqueeze(dim=0)

            num = num + ((x_pos * y_pos).sum(dim=-1)/self.temp).exp().sum()

            if self.stc:
                tmp = ((x_pos * x_pos.transpose(0, 1)).sum(dim=-1)/self.temp).exp()
                tmp = tmp * (1.0 - torch.eye(tmp.shape[0], dtype=torch.float32, device=x.device))
                num = num + tmp.sum()/2.

            x_neg = x[~x_map, :]
            y_neg = y[~y_map, :]

            x_neg, y_neg = x_neg.unsqueeze(dim=0), y_neg.unsqueeze(dim=0)

            den = den + ((x_pos * y_neg).sum(dim=-1)/self.temp).exp().sum()

            if self.stc:
                den = den + ((x_pos * x_neg).sum(dim=-1)/self.temp).exp().sum()

        loss = -(num/(den+self.eps)+self.eps2).log()

        return loss
