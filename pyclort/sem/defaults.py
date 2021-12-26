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

import numpy as np
from filterpy.common import Q_discrete_white_noise


class KFSampleParams:

    def __init__(self,
                 dt: float=0.1) -> None:
        self.dt = dt

    def state_transition_matrix(self):

        F = np.eye(12)

        for i in range(6):
            F[i, 6+i] = self.dt

        return F


    def measurement_function(self):

        H = np.zeros(shape=(6, 12), dtype=np.float32)

        for i in range(6):
            H[i, i] = 1.

        return H

    def state_covariance_matrix(self):

        return np.eye(12)

    def measurement_noise(self):

        return np.eye(6)

    def process_noise(self):

        return Q_discrete_white_noise(dim=12, dt=self.dt)
