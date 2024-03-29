from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def _increase_util(_mean: float, default: float = 0.07) -> float:
    _temp = default

    if _mean > 0.8:
        _temp = 0.7
    elif 0.7 < _mean <= 0.8:
        _temp = 0.5
    elif 0.5 < _mean <= 0.7:
        _temp = 0.3
    elif 0.3 < _mean <= 0.5:
        _temp = 0.2
    elif -0.1 < _mean <= 0.3:
        _temp = 0.1
    elif -0.2 < _mean <= -0.1:
        _temp = 0.085
    else:
        pass

    return _temp

def gradual_increase_temp(pos: torch.Tensor, neg: torch.Tensor) -> Tuple[float | torch.Tensor, float | torch.Tensor]:
    neg = -neg
    _mean_p, _mean_n = pos.mean().item(), neg.mean().item()
    _std_p, _std_n = pos.std().item(), neg.std().item()
    # _max_p, _max_n = pos.max().item(), neg.max().item()
    # _min_p, _min_n = pos.min().item(), neg.min().item()

    _temp_p, _temp_n = 0.07, 0.07
    _temp_p, _temp_n = _increase_util(_mean_p+_std_p, default=_temp_p), _increase_util(_mean_n+_std_n, default=_temp_n)

    return _temp_p, _temp_n

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
                 temp: float = 0.05,
                 max_t: float = 0.15,
                 global_contrast: bool = True,
                 separation: str = 'elements', # elements, tracks, dense
                #  pos_norm: str = 'mean', # 'max', 'min', 'max.std' #TODO@ShivamPR21: Implement different positive normalization method
                 static_contrast: Tuple[bool, bool] | bool = True,
                 soft_condition: bool = True,
                 global_horizon: bool = True,
                 hard_condition_proportion: float = 0.5,
                 sim_type: str | None = "dot", #"dot/diff" #TODO@ShivamPR21 Ad-hoc short term inplace remedy until sim-fxn is implemented
                 sim_fxn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 temperature_adaptation_policy: Callable[[torch.Tensor, torch.Tensor],
                                                         Tuple[float, float]] | str | None = None, # gradual_increase
                 temperature_increase_coeff: float = 0.01,
                 pivot: float = 0.) -> None:
        super().__init__()

        self.temp = float(temp)

        self.temp_adapt_policy = temperature_adaptation_policy
        self.t_coeff = temperature_increase_coeff
        self.max_t = max_t

        if isinstance(self.temp_adapt_policy, str):
            if self.temp_adapt_policy == 'gradual_increase':
                self.temp_adapt_policy = gradual_increase_temp
            else:
                raise NotImplementedError(f'Temperature adaptation policy < {self.temp_adapt_policy} > is not implemented.')

        self.global_contrast = global_contrast
        if separation not in ['elements', 'tracks', 'dense']:
            raise NotImplementedError(f'Separation method < {separation} > is not implemented. choose one of the following < ["elements", "tracks", "dense"] >')
        self.separation = separation

        self.p_stc, self.n_stc = static_contrast if isinstance(static_contrast, tuple) else (static_contrast, static_contrast)

        self.sc, self.hcp = soft_condition, hard_condition_proportion
        self.glh = global_horizon # Localize to temporal horizon

        assert((sim_type is None and sim_fxn is not None) or (sim_type is not None and sim_type in ['dot', 'diff'] and sim_fxn is None)) # Mutually exclusive parameters

        if sim_type is not None:
            if sim_type == 'dot':
                self.neg_sim_fxn = dot_similarity
            elif sim_type == 'diff':
                self.neg_sim_fxn = diff_similarity
            else:
                raise NotImplementedError(f'Similarity Type argument is given as < {sim_type} > which is not implemented.')

        if sim_fxn is not None:
            self.neg_sim_fxn = sim_fxn

        self.pos_sim_fxn = diff_similarity if not self.sc else self.neg_sim_fxn
        self.pivot = pivot

    def _temp_step(self):
        self.temp = min(self.max_t, self.temp+self.t_coeff)

    def _get_temp(self) -> Tuple[float, float]:
        temp_p, temp_n = self.temp, self.temp
        # if self.temp_adapt_policy is not None:
        #     temp_p, temp_n = self.temp_adapt_policy(pos, neg) # type: ignore

        return temp_p, temp_n # type: ignore

    def loss(self, num:torch.Tensor, den:torch.Tensor, pivot: torch.Tensor | None = None) -> torch.Tensor:
        loss : torch.Tensor | None = None

        if self.sc:
            loss = -(num/(den+num)).log()
        else:
            loss = (self.hcp*num + (den+num).log())/(self.hcp+1.)

        if pivot is not None and self.pivot > 0.:
            loss = loss + self.pivot * pivot

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

        # y_idxs : torch.Tensor | None = None
        # Q : int | None = None

        n, Q, _ = y.size()
        y_idxs = torch.arange(n, dtype=torch.int32)
        local_contrast_map : torch.Tensor = torch.ones(n, dtype=torch.bool) # All true
        if not self.global_contrast: # Local Contrast
            local_contrast_map = torch.zeros(n, dtype=torch.bool) # All False
            local_contrast_map[ut_ids] = True # Except current track_ids

        num, den, = [], []
        pivot_loss: List[torch.Tensor] | torch.Tensor | None = []
        _p_cnt, n_cnt = [], []

        for _i, uid in enumerate(ut_ids):
            x_map = track_idxs == uid
            y_map = y_idxs == uid

            ## Positive similarities
            x_pos = x[x_map, :]
            y_pos = y[y_map, :, :].flatten(0, 1)

            sim_p = self.pos_sim_fxn(x_pos, y_pos).min(dim=1, keepdim=True).values

            ## Negative similarities
            x_neg = x[~x_map, :]
            y_neg = y[(~y_map) * local_contrast_map, :].flatten(0, 1)

            sim_n = self.neg_sim_fxn(x_pos , y_neg)
            if self.n_stc:
                # idxs = tuple(torch.triu_indices(x_pos.shape[0], x_neg.shape[0], 1))
                sim_n = torch.cat([sim_n,
                                  self.neg_sim_fxn(x_pos, x_neg).detach()],
                                  dim=1)
            n_cnt.append(np.array([sim_n.shape[1]]*sim_n.shape[0]))

            if self.temp_adapt_policy is not None:
                raise NotImplementedError("Temperature adaptation policy is currently defunct, will be re-implemented soon.")
            temp_p, temp_n = self._get_temp() #TODO@ShivamPR21: Temperature adaptation defunct

            # Positive similarities
            num.append((sim_p/temp_p).exp() if self.sc else -sim_p)
            # Hard constraint numerator is added to loss thus needs to mean instead of sum

            # Negative similarities
            den.append((sim_n/temp_n).exp().sum(dim=-1))

            # Pivot loss
            if self.pivot > 0.:
                pivot_loss.append(((1 - sim_p).mean(dim=-1).square() + (1 + sim_n).mean(dim=-1).square()).sqrt())

        if self.separation == 'tracks':
            # Separate tracks, but joint loss
            num = torch.cat([num_.mean() for num_ in num])
            den = torch.cat([den_.sum() for den_ in den])
            pivot_loss = torch.cat([pivot_loss_.sum() for pivot_loss_ in pivot_loss]) if len(pivot_loss) > 0. else None
            # Negative Counts for loss normalization
            n_cnt = np.concatenate([n_cnt_.sum() for n_cnt_ in n_cnt]).reshape(-1, )
        elif self.separation == 'elements':
            # Complete separation
            num, den, pivot_loss = \
                torch.cat(num), torch.cat(den), \
                    (torch.cat(pivot_loss) if len(pivot_loss) > 0. else None)
            # Negative Counts for loss normalization
            n_cnt = np.concatenate(n_cnt).reshape(-1, )
        elif self.separation == 'dense':
            # Complete Loss
            num, den, pivot_loss = \
                torch.cat(num).mean(), torch.cat(den).sum(), \
                    (torch.cat(pivot_loss).sum() if len(pivot_loss) > 0. else None)
            # Negative Counts for loss normalization
            n_cnt = np.concatenate(n_cnt).sum().reshape(-1, )
        else:
            raise NotImplementedError(f'Separation method < {self.separation} > is not implemented. choose one of the following < ["elements", "tracks", "dense"] >')

        loss = self.loss(num, den, pivot_loss).mean()

        return loss
