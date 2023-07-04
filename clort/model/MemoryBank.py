from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class MemoryBank(nn.Module):

    def __init__(self, n_tracks: int, N: int, Q: int,
                 alpha: Union[np.ndarray, torch.Tensor],
                 eps: float = 1e-9, device: torch.device | str = 'cpu',
                 init: str = 'zeros',
                 init_dilation: int = 1,
                 init_density: int = 1) -> None:
        assert(init in ['zeros', 'uniform', 'orthogonal.uniform', 'orthogonal.distributed'])
        assert((init_density >= 1) and (init_dilation >= 1) and (init in ['orthogonal.uniform', 'orthogonal.distributed'] or (init_dilation == 1 and init_density == 1)))

        super().__init__()

        self.eps = eps
        self.device = device
        self.N, self.Q, self.n_tracks = N, Q, n_tracks
        self.init, self.init_dilation, self.init_density = init, init_dilation,init_density

        if isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha.tolist(), dtype=torch.float32, device=self.device)

        if alpha.device != self.device:
            alpha = alpha.to(self.device)

        self.alpha = nn.Parameter(alpha.reshape((Q, 1)), requires_grad = False)

        self.memory = nn.Parameter(self.init_memory(), requires_grad=False)
        self.update_cnt = nn.Parameter(torch.zeros((self.n_tracks,), dtype=torch.bool), requires_grad=False)

    def init_memory(self) -> torch.Tensor:
        memory = None
        if self.init == 'zeros':
            memory = torch.zeros((self.n_tracks, self.Q, self.N), dtype=torch.float32, device=self.device) # zeros initialization
        if self.init == 'uniform':
            memory = torch.ones((self.n_tracks, self.Q, self.N), dtype=torch.float32, device=self.device) / np.sqrt(self.N) # uniform initialization
        elif self.init == 'orthogonal.uniform':
            memory = torch.zeros((self.n_tracks, self.Q, self.N), dtype=torch.float32, device=self.device) # uniform initialization
            for i in range(self.n_tracks):
                truth_idxs = (torch.arange(i, i+self.init_density, dtype=torch.int32)*self.init_dilation)%self.N
                memory[i, :, truth_idxs] = 1. # Orthogonal uniform initialization
            memory /= np.sqrt(self.init_density)
        elif self.init == 'orthogonal.distributed':
            memory = torch.zeros((self.n_tracks, self.Q, self.N), dtype=torch.float32, device=self.device) # uniform initialization
            for i in range(self.n_tracks):
                for q in range(self.Q):
                    truth_idxs = (torch.arange(i, i+self.init_density, dtype=torch.int32)*self.init_dilation+\
                                  q*self.init_dilation)%self.N
                    memory[i, q, truth_idxs] = 1. # Orthogonal distributed initialization
            memory /= np.sqrt(self.init_density)
        else:
            raise NotImplementedError(f'The initialization method {self.init} isn\'t implemented')

        return memory

    def reset(self) -> None:
        self.memory[:, :, :] = self.init_memory()
        self.update_cnt[:] = False

    def update(self, reprs: torch.Tensor, track_idxs: torch.Tensor) -> None:
        assert(self.memory is not None)
        # Warning: Use normalized representations
        # track_idxs -> [n, ]
        # reprs_dims -> [n, N]
        if reprs.device != self.device:
            reprs = reprs.to(self.device)

        u_tids = torch.unique(track_idxs) # unique track ids # [n',]

        for uid in u_tids:
            map = (uid == track_idxs) # [n, ]
            track_reprs = reprs[map, :].unsqueeze(dim=0) # [1, k<<n, N]
            mem_reprs = self.memory[uid, :, :].unsqueeze(dim=1) # [Q, 1, N]

            if self.init == 'zeros' and not self.update_cnt[uid]:
                sim_idxs = torch.randint(track_reprs.shape[1], size=(self.Q,), device=self.device)
                self.update_cnt[uid] = True
            else:
                sim_mat = (mem_reprs * track_reprs).sum(dim=-1) # Similarity matrix # [Q, k]
                sim_idxs = sim_mat.argmin(dim=1) # Least similar index over k encodings # [Q,]

            track_reprs, mem_reprs = track_reprs.squeeze(dim=0), mem_reprs.squeeze(dim=1)

            mem_reprs = mem_reprs*self.alpha + track_reprs[sim_idxs, :]*(1-self.alpha) # [Q, N]
            self.memory[uid, :, :] = mem_reprs / (mem_reprs.norm(dim=-1, keepdim=True) + self.eps) # Representation Normalization

    def get_reprs(self, track_idxs: torch.Tensor) -> torch.Tensor:
        assert(self.memory is not None)
        # track_idxs -> [n, ]
        return self.memory[track_idxs, :, :] # [n, Q, N]

    def get_memory(self) -> torch.Tensor:
        assert(self.memory is not None)
        return self.memory

class MemoryBankInfer(nn.Module):

    def __init__(self, n_tracks: int, N: int, Q: int, t: int = 3,
                 alpha_threshold: Tuple[float, float] = (0.1, 0.9), beta_threshold: Tuple[float, float] = (0.1, 0.9), device: torch.device | str = 'cpu') -> None:
        super().__init__()
        self.n_tracks, self.N, self.Q, self.t, self.alpha_t, self.beta_t = \
            n_tracks, N, Q, t, alpha_threshold, beta_threshold

        self.eps = 1e-9
        self.device = device

        self.beta = nn.Parameter(torch.zeros((n_tracks, Q), dtype=torch.float32, device=self.device),
                                 requires_grad = False)
        self.count = nn.Parameter(torch.zeros((n_tracks, 1), dtype=torch.float32, device=self.device),
                                  requires_grad = False)

        self.memory = nn.Parameter(torch.zeros((n_tracks, Q, N), dtype=torch.float32, device=self.device),
                                   requires_grad = False)

    def reset(self):
        self.beta[:, :] = 0
        self.count[:, :] = 0.
        self.memory[:, :, :] = 0.

    def update(self, reprs: torch.Tensor, track_idxs: torch.Tensor) -> None:
        # reprs -> [n_tracks, N]
        if reprs.device != self.device:
            reprs = reprs.to(self.device)

        if track_idxs.device != self.device:
            track_idxs = track_idxs.to(self.device)

        nrm = reprs.norm(dim=1, keepdim=True)
        if not torch.all(torch.isclose(nrm, torch.tensor([1.], dtype=torch.float32, device=reprs.device))):
            reprs = reprs/nrm

        count = self.count[track_idxs, :]

        for q in range(self.Q):
            beta, memory = self.beta[track_idxs, q:q+1], self.memory[track_idxs, q, :] # [n_tracks, 1], [n_tracks, N]

            beta_ = (reprs * memory).sum(dim=1, keepdim=True) # [n_tracks, 1]
            beta_[(count == 0).flatten(), :] = 0.1 # Check for initialization

            beta = (beta*count + beta_)/(count+1.)
            beta = torch.clip(beta, self.beta_t[0], self.beta_t[1]) # Clip to threshold

            self.beta[track_idxs, q:q+1] = beta

            prev_mem = torch.zeros(memory.size(), dtype=memory.dtype, device=self.device)

            if q !=0 :
                prev_mem = self.memory[track_idxs, q-1, :]

            alpha = (memory*prev_mem).sum(dim=1, keepdim=True) # [n_tracks, 1]
            alpha = torch.clip(alpha, self.alpha_t[0], self.alpha_t[1]) # Clip to threshold

            sf = max(max(q+1, self.t), 2)

            memory = (1. - (alpha+beta)/sf) * memory + (alpha/sf)*prev_mem + (beta/sf)*reprs

            memory = memory/(memory.norm(dim = 1, keepdim = True) + self.eps) # Normalization

            self.memory[track_idxs, q, :] = memory

        self.count[track_idxs, :] += 1.

    def get_reprs(self, track_idxs: torch.Tensor) -> torch.Tensor:
        # track_idxs -> [n, ]
        return self.memory[track_idxs, :, :] # [n, Q, N]

    def get_memory(self) -> torch.Tensor:
        return self.memory
