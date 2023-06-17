from typing import Union

import numpy as np
import torch


class MemoryBank:

    def __init__(self, n_tracks: int, N: int, Q: int,
                 alpha: Union[np.ndarray, torch.Tensor],
                 eps: float = 1e-9, device: torch.device | str = 'cpu') -> None:
        self.eps = eps
        self.device = device
        self.N, self.Q, self.n_tracks = N, Q, n_tracks

        if isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha.tolist(), dtype=torch.float32, device=self.device)

        if alpha.device != self.device:
            alpha = alpha.to(self.device)

        self.alpha = alpha.reshape((Q, 1))

        self.memory = torch.zeros((n_tracks, Q, N), dtype=torch.float32, device=self.device)

    def update(self, reprs: torch.Tensor, track_idxs: torch.Tensor) -> None:
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

            if torch.all(mem_reprs == 0.):
                sim_idxs = torch.randint(track_reprs.shape[1], size=(self.Q,), device=self.device)
            else:
                sim_mat = (mem_reprs * track_reprs).sum(dim=-1) # Similarity matrix # [Q, k]
                sim_idxs = sim_mat.argmin(dim=1) # Least similar index over k encodings # [Q,]

            track_reprs, mem_reprs = track_reprs.squeeze(dim=0), mem_reprs.squeeze(dim=1)

            mem_reprs = mem_reprs*self.alpha + track_reprs[sim_idxs, :]*(1-self.alpha) # [Q, N]
            self.memory[uid, :, :] = mem_reprs / (mem_reprs.norm(dim=-1, keepdim=True) + self.eps) # Representation Normalization

    def get_reprs(self, track_idxs: torch.Tensor) -> torch.Tensor:
        # track_idxs -> [n, ]
        return self.memory[track_idxs, :, :] # [n, Q, N]

    def get_memory(self) -> torch.Tensor:
        return self.memory

class MemoryBankInfer:

    def __init__(self, n_tracks: int, N: int, Q: int, t: int = 3,
                 alpha_threshold: float = 0.3, beta_threshold: float = 0.2, device: torch.device | str = 'cpu') -> None:
        self.n_tracks, self.N, self.Q, self.t, self.alpha_t, self.beta_t = n_tracks, N, Q, t, alpha_threshold, beta_threshold

        self.eps = 1e-9
        self.device = device

        self.beta = torch.zeros((n_tracks, Q), dtype=torch.float32, device=self.device)
        self.count = torch.zeros((n_tracks, 1), dtype=torch.float32, device=self.device)

        self.memory = torch.zeros((n_tracks, Q, N), dtype=torch.float32, device=self.device)

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
            beta_[(count == 0).flatten(), :] = 1. # Check for initialization

            beta = (beta*count + beta_)/(count+1.)
            beta = torch.clip(beta, self.beta_t, 1.) # Clip to threshold

            self.beta[track_idxs, q:q+1] = beta

            prev_mem = torch.zeros(memory.size(), dtype=memory.dtype, device=self.device)

            if q !=0 :
                prev_mem = self.memory[track_idxs, q-1, :]

            alpha = (memory*prev_mem).sum(dim=1, keepdim=True) # [n_tracks, 1]
            alpha = torch.clip(alpha, self.alpha_t, 1.) # Clip to threshold

            sf = max(q+1, self.t)

            memory = (1. - (alpha+beta)/sf) * memory + (alpha/sf)*prev_mem + (beta/sf)*reprs

            memory = memory/(memory.norm(dim = 1, keepdim = True) + self.eps) # Normalization

            self.memory[track_idxs, q, :] = memory

        self.count[track_idxs, :] += 1.

    def get_reprs(self, track_idxs: torch.Tensor) -> torch.Tensor:
        # track_idxs -> [n, ]
        return self.memory[track_idxs, :, :] # [n, Q, N]

    def get_memory(self) -> torch.Tensor:
        return self.memory
