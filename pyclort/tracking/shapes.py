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

from typing import TypedDict, float, int

import numpy as np

#        /        /       |
#      /        /         |
#      . ______.         |
#      |       |        |
# <b>  |       |      /
#      |       |    / <h>
#      ._______.  /
#        <l>


# Single dimension
class SingleDimPos(TypedDict):
    x : float

class SingleDimShape(TypedDict):
    l : float

class SingleDimPosVel(TypedDict):
    x_dot : float

class SingleDimShapeVel(TypedDict):
    l_dot : float

class SingleDimCenter(TypedDict):
    center : SingleDimPos
    shape : SingleDimShape

class SingleDimVelocity(TypedDict):
    center_dot : SingleDimPosVel
    shape_dot : SingleDimShapeVel

# 2D types
class BoundingRectangleCenter(SingleDimPos):
    y : float

class BoundingRectangleShape(SingleDimShape):
    b : float

class BoundingRectangleCenterVel(SingleDimPosVel):
    y_dot : float

class BoundingRectangleShapeVel(SingleDimShapeVel):
    b_dot : float

class BoundingRectangle(TypedDict):
    center : BoundingRectangleCenter
    shape : BoundingRectangleShape

class BoundingRectangleVelocity(TypedDict):
    center_dot : BoundingRectangleCenterVel
    shape_dot : BoundingRectangleShapeVel

# 3D types
class BoundingBoxCenter(BoundingRectangleCenter):
    z : float

class BoundingBoxShape(BoundingRectangleShape):
    h : float

class BoundingBoxCenterVel(BoundingRectangleCenterVel):
    z_dot : float

class BoundingBoxShapeVel(BoundingRectangleShapeVel):
    h_dot : float

class BoundingBox(TypedDict):
    center : BoundingBoxCenter
    shape : BoundingBoxShape

class BoundingBoxVelocity(TypedDict):
    center_dot : BoundingBoxCenterVel
    shape_dot : BoundingBoxShapeVel
