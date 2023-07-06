from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn


def gradual_increase_temp(x: torch.Tensor) -> float:
    _mean = x.mean().item()
    _std = x.std().item()
    _max = x.max().item()
    _min = x.std().item()

    if _mean > 0.7:
        return 0.07
    elif 0.5 < _mean <= 0.7:
        return 0.15
    elif 0.3 < _mean <= 0.5:
        return 0.20
    elif 0.1 < _mean <= 0.3:
        return 0.30
    else:
        pass

    return 1.

def dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sim = x @ y.T
    return sim

def diff_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(dim=1) if x.ndim == 2 else x
    y = y.unsqueeze(dim=0) if y.ndim == 2 else y

    assert(x.ndim == 3 and y.ndim == 3)

    sim = -(x - y).norm(dim=-1)
    return sim

class ContrastiveLoss(nn.Module):

    def __init__(self,
                 temp: float = 0.3,
                 global_contrast: bool = True,
                 separate_tracks: bool = True,
                 static_contrast: Tuple[bool, bool] | bool = True,
                 soft_condition: bool = True,
                 global_horizon: bool = True,
                 hard_condition_proportion: float = 0.5,
                 sim_type: str | None = "dot", #"dot/diff" #TODO@ShivamPR21 Ad-hoc short term inplace remedy until sim-fxn is implemented
                 sim_fxn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 temperature_adaptation_policy: Callable[[torch.Tensor], torch.Tensor] | str | None = None # gradual_increase
                 ) -> None:
        super().__init__()

        self.temp = float(temp)
        self.temp_adapt_policy = temperature_adaptation_policy

        if isinstance(self.temp_adapt_policy, str):
            if self.temp_adapt_policy == 'gradual_increase':
                self.temp_adapt_policy = gradual_increase_temp
            else:
                raise NotImplementedError(f'Temperature adaptation policy < {self.temp_adapt_policy} > is not implemented.')

        self.global_contrast = global_contrast
        self.separate_tracks = separate_tracks

        self.p_stc, self.n_stc = (static_contrast, static_contrast) if isinstance(static_contrast, bool) else static_contrast

        self.sc, self.hcp = soft_condition, hard_condition_proportion
        self.glh = global_horizon # Localize to temporal horizon

        assert((sim_type is None and sim_fxn is not None) or (sim_type is not None and sim_type in ['dot', 'diff'] and sim_fxn is None)) # Mutually exclusive parameters

        if sim_fxn is not None:
            if sim_type == 'dot':
                self.neg_sim_fxn = dot_similarity
            elif sim_type == 'diff':
                self.neg_sim_fxn = diff_similarity
            else:
                raise NotImplementedError(f'Similarity Type argument is given as < {sim_type} > which is not implemented.')

        if sim_fxn is not None:
            self.neg_sim_fxn = sim_fxn

        self.pos_sim_fxn = diff_similarity if not self.sc else self.neg_sim_fxn

    def _get_temp(self, sim: torch.Tensor) -> float:
        temp = self.temp
        if self.temp_adapt_policy is not None:
            temp = self.temp_adapt_policy(sim) # type: ignore

        return temp

    def loss(self, num:torch.Tensor, den:torch.Tensor) -> torch.Tensor:
        loss : torch.Tensor | None = None

        if self.sc:
            loss = -(num/(den+num)).log()
        else:
            loss = (self.hcp*num + (den+num).log())/(self.hcp+1.)

        return loss

    def forward(self, x: torch.Tensor, track_idxs: torch.Tensor, y: torch.Tensor,
                n_tracks: np.ndarray | None = None) -> torch.Tensor:
        if n_tracks is not None and not self.glh:
            loss = torch.tensor([0], dtype=torch.float32, device=x.device)

            x = x.split(n_tracks.tolist(), dim=0)
            track_idxs = track_idxs.split(n_tracks.tolist(), dim=0)

            for i in range(len(n_tracks)):
                loss = loss + self.forward_(x[i], track_idxs[i], y)/len(n_tracks)

            return loss

        return self.forward_(x, track_idxs, y)

    def forward_(self, x: torch.Tensor, track_idxs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x -> [n_obj', N]
        # y -> [n_obj, Q, N]

        if (y.device != x.device):
            y = y.to(x.device)

        ut_ids = track_idxs.unique()

        y_idxs : torch.Tensor | None = None
        Q : int | None = None

        if self.global_contrast:
            n, Q, _ = y.size()
            y = y.flatten(0, 1)

            y_idxs = torch.arange(n, dtype=torch.int32).repeat(Q)
        else:
            y = y[ut_ids, :, :]

            _, Q, _ = y.size()
            y = y.flatten(0, 1)

            y_idxs = ut_ids.repeat(Q)

        assert(Q is not None)

        num, den = torch.zeros(len(ut_ids), dtype=torch.float32, device=x.device), \
            torch.zeros(len(ut_ids), dtype=torch.float32, device=x.device)
        n_pos, n_neg = 0, 0

        loss: torch.Tensor | None = None

        for i, uid in enumerate(ut_ids):
            x_map = track_idxs == uid
            y_map = y_idxs == uid

            x_pos = x[x_map, :]
            y_pos = y[y_map, :]

            ## Positive similarities
            sim = self.pos_sim_fxn(x_pos, y_pos)

            temp = self._get_temp(sim)
            num[i] = num[i] + (
                (sim/temp).exp().sum() if self.sc else (-sim).mean()
                )
            # Hard constraint numerator is added to loss thus needs to mean instead of sum
            n_pos += x_pos.shape[0] * y_pos.shape[0]

            if self.p_stc:
                sim = self.pos_sim_fxn(x_pos, x_pos)

                temp = self._get_temp(sim)
                tmp = (sim/temp).exp() if self.sc else -sim
                tmp = tmp * (1.0 - torch.eye(tmp.shape[0], dtype=torch.float32, device=x.device))
                num[i] = num[i] + (tmp.sum()/2. if self.sc else tmp.mean())
                # Hard constraint numerator is added to loss thus needs to mean instead of sum
                n_pos += (x_pos.shape[0]*(x_pos.shape[0] - 1)/2.)

            # Negative similarities
            x_neg = x[~x_map, :]
            y_neg = y[~y_map, :]

            sim = self.neg_sim_fxn(x_pos , y_neg)

            temp = self._get_temp(sim)
            den[i] = den[i] + (sim/temp).exp().sum()
            n_neg += x_neg.shape[0] * y_neg.shape[1]

            if self.n_stc:
                sim = self.neg_sim_fxn(x_pos, x_neg)

                temp = self._get_temp(sim)
                den[i] = den[i] + (sim/temp).exp().sum()
                n_neg += x_pos.shape[0] * x_neg.shape[0]

        if self.separate_tracks:
            loss = self.loss(num, den).mean()/float(Q)
        else:
            loss = self.loss(num.sum(), den.sum())/float(len(ut_ids)*Q)

        return loss
