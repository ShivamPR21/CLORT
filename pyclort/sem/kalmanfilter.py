'''
Copyright (C) 2021  Shiavm Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import Any, Callable

import numpy as np
from filterpy.kalman import KalmanFilter as KF

from .defaults import KFSampleParams
from .evolution import StateEvolution


class KalmanFilter(StateEvolution):

    def __init__(self,
                 state_dim: int = 12,
                 obs_dim: int = 6,
                 params: Callable = KFSampleParams()) -> None:
        super().__init__(state_dim=state_dim, obs_dim=obs_dim)

        assert(isinstance(params, KFSampleParams))

        self.filter = KF(dim_x = self.state_dim, dim_z = self.obs_dim)

        self.filter.F = params.state_transition_matrix()
        self.filter.H = params.measurement_function()
        self.filter.Q = params.process_noise()
        self.filter.P = params.state_covariance_matrix()
        self.filter.R = params.measurement_noise()

    def predict(self) -> Any:
        x, P = self.filter.predict()
        return x, P

    def update(self,
               observation = None,
               measurement_noise = None) -> Any:
        assert((not observation is None) and (np.shape(observation) == self.obs_dim))

        if (not measurement_noise is None):
            assert(np.shape(measurement_noise) == np.shape(self.filter.R))
            self.filter.R = measurement_noise

        x, P = self.filter.update(observation)
        return x, P

    def state_covar(self) -> Any:
        return self.filter.P

    def obs_covar(self) -> Any:
        return self.filter.R

    def process_noise(self) -> Any:
        return self.filter.Q
