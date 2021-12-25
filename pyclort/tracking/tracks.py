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

from typing import Callable, float, int

import numpy as np
from sem.evolution import StateEvolution
from sem.kalmanfilter import KalmanFilter

from ..helpers import ModifiedQueue
from .shapes import BoundingBox
from .tracklet import Tracklet3D


class Track3D:

    def __init__(self,
                id : int = None,
                dt : float = 0.5,
                observation : BoundingBox =
                    {'center' : {'x' : 0.0, 'y' : 0.0, 'z' : 0.0},
                    'shape' : {'l' : 0.0, 'b' : 0.0, 'h' : 0.0}},
                max_tracklets : int = 20,
                stale_lim : int = 5,
                sem: Callable = KalmanFilter()) -> None:

        assert(id)
        self.track_id = id
        self.dt = dt
        self.tracklets = ModifiedQueue(maxsize=max_tracklets)
        assert(isinstance(sem, StateEvolution))
        self.sem = sem

        # Create initial tracklet
        tracklet = Tracklet3D(tracklet = observation,
                              observation = observation,
                              x_covar = sem.state_covar(),
                              z_covar = sem.obs_covar())

        self.tracklets.put(tracklet)
        self.setup_kf(tracklet)
        self.tracks_count = 0
        self.missed_frames = 0
        self.stale_lim = stale_lim
        self.stale = False

    def predict(self) :
        x , P = self.sem.predict()
        self.tracklets.put(Tracklet3D(x_covar = P))
        self.tracklets[-1].update_state_from_vec(np.squeeze(x))
        return self.tracklets[-1]

    def update(self, observation : BoundingBox = None, z_covar = None) -> None:
        if (observation is None) :
            self.missed_frames += 1
            if (self.missed_frames >= self.stale_lim):
                self.stale = True
            return

        self.missed_frames = 0
        self.tracks_count += 1

        # Create a new tracklet
        self.tracklets[-1].update(observation=observation, z_covar=z_covar)

        x, P = self.sem.update(self.tracklets[-1].observation, self.tracklets[-1].z_covar)
        self.tracklets[-1].update(x_covar = P)
        self.tracklets[-1].update_state_from_vec(np.squeeze(x))

        return self.predict()
