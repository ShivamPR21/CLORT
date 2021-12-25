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

from .shapes import (
    BoundingBox,
    BoundingBoxVelocity,
    BoundingRectangle,
    BoundingRectangleVelocity,
    SingleDimCenter,
    SingleDimVelocity,
)


class Tracklet1D:

    def __init__(self,
                 tracklet : SingleDimCenter =
                    {'center' : {'x' : 0.0},
                    'shape' : {'l' : 0.0}},
                 velocity : SingleDimVelocity =
                    {'center_dot' : {'x_dot' : 0.0},
                    'shape_dot' : {'l_dot' : 0.0}},
                 observation : SingleDimCenter =
                    {'center' : {'x' : 0.0},
                    'shape' : {'l' : 0.0}},
                 covar = np.eye(4)) -> None:
        """Tracklet1D stores the data about object's location and movement at specific instance in time.

        Args:
            tracklet (BoundingBox, optional): [tracklet position, and shape]. Defaults to {'center' : {'x' : 0.0}, 'shape' : {'l' : 0.0}}.
            velocity (BoundingBox, optional): [tracklet state change rate]. Defaults to {'center_dot' : {'x_dot' : 0.0}, 'shape_dot' : {'l_dot' : 0.0}}.
            observation (BoundingBox, optional): [most recent associated bounding box]. Defaults to {'center' : {'x' : 0.0}, 'shape' : {'l' : 0.0}}.
            covar (numpy matrix, optional): [covariance at the tracklet]. Defaults to np.eye(4).
        """
        self.tracklet = tracklet
        self.velocity = velocity

        self.observation = observation

        assert(covar.shape == (4, 4))
        self.covar = covar

    def update(self,
                tracklet : SingleDimCenter = None,
                velocity : SingleDimVelocity = None,
                observation : SingleDimCenter = None,
                covar = None) -> None:
        """Updates the requested state realted to tracklet.

        Args:
            tracklet (SingleDimCenter, optional): [tracklet position, and shape]. Defaults to None.
            velocity (SingleDimVelocity, optional): [tracklet state change rate]. Defaults to None.
            observation (SingleDimCenter, optional): [most recent associated bounding box]. Defaults to None.
            covar ([type], optional): [covariance at the tracklet]. Defaults to None.
        """
        if (tracklet) : self.tracklet = tracklet
        if (velocity) : self.velocity = velocity
        if (observation) : self.observation = observation
        if (covar) :
            assert(covar.shape == (4, 4))
            self.covar = covar

    def state(self):
        return np.array([self.tracklet['x'],
                        self.tracklet['l'],
                        self.velocity['x_dot'],
                        self.velocity['l_dot']], dtype=np.float32)


class Tracklet2D:

    def __init__(self,
                 tracklet : BoundingRectangle =
                    {'center' : {'x' : 0.0, 'y' : 0.0},
                    'shape' : {'l' : 0.0, 'b' : 0.0}},
                 velocity : BoundingRectangleVelocity =
                    {'center_dot' : {'x_dot' : 0.0, 'y_dot' : 0.0},
                    'shape_dot' : {'l_dot' : 0.0, 'b_dot' : 0.0}},
                 observation : BoundingRectangle =
                    {'center' : {'x' : 0.0, 'y' : 0.0},
                    'shape' : {'l' : 0.0, 'b' : 0.0}},
                 covar = np.eye(8)) -> None:
        """Tracklet2D stores the data about object's location and movement at specific instance in time.

        Args:
            tracklet (BoundingRectangle, optional): [tracklet position, and shape]. Defaults to {'center' : {'x' : 0.0, 'y' : 0.0}, 'shape' : {'l' : 0.0, 'b' : 0.0}}.
            velocity (BoundingRectangleVelocity, optional): [tracklet state change rate]. Defaults to {'center_dot' : {'x_dot' : 0.0, 'y_dot' : 0.0}, 'shape_dot' : {'l_dot' : 0.0, 'b_dot' : 0.0}}.
            observation (BoundingRectangle, optional): [most recent associated bounding box]. Defaults to {'center' : {'x' : 0.0, 'y' : 0.0}, 'shape' : {'l' : 0.0, 'b' : 0.0}}.
            covar (numpy matrix, optional): [covariance at the tracklet]. Defaults to np.eye(8).
        """
        self.tracklet = tracklet
        self.velocity = velocity

        self.observation = observation

        assert(covar.shape == (8, 8))
        self.covar = covar

    def update(self,
                tracklet : BoundingRectangle = None,
                velocity : BoundingRectangleVelocity = None,
                observation : BoundingRectangle = None,
                covar = None) -> None:
        """Updates the requested state realted to tracklet.

        Args:
            tracklet (BoundingRectangle, optional): [tracklet position, and shape]. Defaults to None.
            velocity (BoundingRectangleVelocity, optional): [tracklet state change rate]. Defaults to None.
            observation (BoundingRectangle, optional): [most recent associated bounding box]. Defaults to None.
            covar ([type], optional): [covariance at the tracklet]. Defaults to None.
        """
        if (tracklet) : self.tracklet = tracklet
        if (velocity) : self.velocity = velocity
        if (observation) : self.observation = observation
        if (covar) :
            assert(covar.shape == (8, 8))
            self.covar = covar

    def state(self):
        return np.array([self.tracklet['x'],
                        self.tracklet['y'],
                        self.tracklet['l'],
                        self.tracklet['b'],
                        self.velocity['x_dot'],
                        self.velocity['y_dot'],
                        self.velocity['l_dot'],
                        self.velocity['b_dot']], dtype=np.float32)

class Tracklet3D:

    def __init__(self,
                 tracklet : BoundingBox =
                    {'center' : {'x' : 0.0, 'y' : 0.0, 'z' : 0.0},
                    'shape' : {'l' : 0.0, 'b' : 0.0, 'h' : 0.0}},
                 velocity : BoundingBoxVelocity =
                    {'center_dot' : {'x_dot' : 0.0, 'y_dot' : 0.0, 'z_dot' : 0.0},
                    'shape_dot' : {'l_dot' : 0.0, 'b_dot' : 0.0, 'h_dot' : 0.0}},
                 observation : BoundingBox =
                    {'center' : {'x' : 0.0, 'y' : 0.0, 'z' : 0.0},
                    'shape' : {'l' : 0.0, 'b' : 0.0, 'h' : 0.0}},
                 x_covar = np.eye(12),
                 z_covar = np.eye(6)) -> None:
        """Tracklet3D stores the data about object's location and movement at specific instance in time.

        Args:
            tracklet (BoundingBox, optional): [tracklet position, and shape]. Defaults to {'center' : {'x' : 0.0, 'y' : 0.0, 'z' : 0.0}, 'shape' : {'l' : 0.0, 'b' : 0.0, 'h' : 0.0}}.
            velocity (BoundingBox, optional): [tracklet state change rate]. Defaults to {'center_dot' : {'x_dot' : 0.0, 'y_dot' : 0.0, 'z_dot' : 0.0}, 'shape_dot' : {'l_dot' : 0.0, 'b_dot' : 0.0, 'h_dot' : 0.0}}.
            observation (BoundingBox, optional): [most recent associated bounding box]. Defaults to {'center' : {'x' : 0.0, 'y' : 0.0, 'z' : 0.0}, 'shape' : {'l' : 0.0, 'b' : 0.0, 'h' : 0.0}}.
            covar (numpy matrix, optional): [covariance at the tracklet]. Defaults to np.eye(12).
        """
        self.tracklet = tracklet
        self.velocity = velocity

        self.observation = observation

        assert(x_covar.shape == (12, 12))
        self.x_covar = x_covar

        assert(z_covar.shape == (6, 6))
        self.z_covar = z_covar

    def update(self,
                tracklet : BoundingBox = None,
                velocity : BoundingBox = None,
                observation : BoundingBox = None,
                x_covar = None,
                z_covar = None) -> None:
        """Updates the requested state realted to tracklet.

        Args:
            tracklet (BoundingBox, optional): [tracklet position, and shape]. Defaults to None.
            velocity (BoundingBox, optional): [tracklet state change rate]. Defaults to None.
            observation (BoundingBox, optional): [most recent associated bounding box]. Defaults to None.
            covar ([type], optional): [covariance at the tracklet]. Defaults to None.
        """
        if (tracklet) : self.tracklet = tracklet
        if (velocity) : self.velocity = velocity
        if (observation) : self.observation = observation
        if (x_covar) :
            assert(x_covar.shape == (12, 12))
            self.x_covar = x_covar
        if (z_covar) :
            assert(z_covar.shape == (6, 6))
            self.z_covar = z_covar

    def state(self):
        return np.array([self.tracklet['x'],
                        self.tracklet['y'],
                        self.tracklet['z'],
                        self.tracklet['l'],
                        self.tracklet['b'],
                        self.tracklet['h'],
                        self.velocity['x_dot'],
                        self.velocity['y_dot'],
                        self.velocity['z_dot'],
                        self.velocity['l_dot'],
                        self.velocity['b_dot'],
                        self.velocity['h_dot']], dtype=np.float32)

    def update_state_from_vec(self, vec) -> None:
        assert(len(vec) == 12)

        self.tracklet = {'center' : {'x' : vec[0], 'y' : vec[1], 'z' : vec[2]},
                        'shape' : {'l' : vec[3], 'b' : vec[4], 'h' : vec[5]}}

        self.velocity = {'center_dot' : {'x_dot' : vec[6], 'y_dot' : vec[7], 'z_dot' : vec[8]},
                        'shape_dot' : {'l_dot' : vec[9], 'b_dot' : vec[10], 'h_dot' : vec[11]}}
