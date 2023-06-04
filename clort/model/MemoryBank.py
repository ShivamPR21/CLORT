from typing import Union

import numpy as np
import torch


class MemoryBank:

    def __init__(self, n_tracks: int, N: int, Q: int,
                 alpha: Union[np.ndarray, torch.Tensor], eps: float = 1e-8) -> None:
        self.eps = eps
        self.alpha = alpha.reshape((Q, 1))
        self.memory = torch.zeros((n_tracks, N, Q), dtype=torch.float32)

    def update(self, reprs: torch.Tensor, track_idxs: torch.Tensor) -> None:
        # Warning: Use normalized representations
        # track_idxs -> [n, ]
        # reprs_dims -> [n, N]
        u_tids = torch.unique(track_idxs) # unique track ids # [n',]

        for uid in u_tids:
            map = (uid == track_idxs) # [n, ]
            track_reprs = reprs[map, :].unsqueeze(dim=0) # [1, k<<n, N]
            mem_reprs = self.memory[uid, :, :].unsqueeze(dim=1) # [Q, 1, N]

            sim_mat = (mem_reprs * track_reprs).sum(dim=-1) # Similarity matrix # [Q, k]
            sim_idxs = sim_mat.argmin(dim=1) # Least similar index over k encodings # [Q,]

            mem_reprs = mem_reprs*self.alpha + track_reprs[sim_idxs, :]*(1-self.alpha) # [Q, N]
            self.memory[uid, :, :] = mem_reprs / (mem_reprs.norm(dim=1, keepdim=True) + self.eps) # Representation Normalization

    def get_reprs(self, track_idxs: torch.Tensor) -> torch.Tensor:
        # track_idxs -> [n, ]
        return self.memory[track_idxs, :, :] # [n, Q, N]
