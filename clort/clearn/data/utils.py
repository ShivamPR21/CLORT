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

from typing import List, Tuple

import numpy as np
from argoverse.utils.calibration import CameraConfig, proj_cam_to_uv
from argoverse.utils.frustum_clipping import clip_segment_v3_plane_n


def clip_line_segment(
    vert_a: np.ndarray,
    vert_b: np.ndarray,
    camera_config: CameraConfig,
    planes: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> List[np.ndarray]:
    """Plot the portion of a line segment that lives within a parameterized 3D camera frustum.

    Args:
        vert_a: first point, in the camera coordinate frame.
        vert_b: second point, in the camera coordinate frame.
        camera_config: CameraConfig object
        planes: frustum clipping plane parameters
    """
    clip_vert_a, clip_vert_b = clip_segment_v3_plane_n(vert_a.copy(), vert_b.copy(), planes.copy())
    if clip_vert_a is None or clip_vert_b is None:
        return None

    uv_a, _, _, _ = proj_cam_to_uv(clip_vert_a.reshape(1, 3), camera_config)
    uv_b, _, _, _ = proj_cam_to_uv(clip_vert_b.reshape(1, 3), camera_config)

    uv_a = uv_a.squeeze()
    uv_b = uv_b.squeeze()
    # cv2 line [uv_a[0], uv_a[1]], [uv_b[0], uv_b[1]]

    return [uv_a, uv_b]

def get_object_patch_from_image(
        img: np.ndarray,
        corners: np.ndarray,
        planes: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        camera_config: CameraConfig,
        central_patch: bool = True,
        ww: Tuple[float, float] = (0.6, 0.6)
    ) -> np.ndarray:
        r"""We bring the 3D points into each camera, and do the clipping there.

        Renders box using OpenCV2. Edge coloring and vertex ordering is roughly based on
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes_utils/data_classes.py

        ::

                5------4
                |\\    |\\
                | \\   | \\
                6--\\--7  \\
                \\  \\  \\ \\
            l    \\  1-------0    h
             e    \\ ||   \\ ||   e
              n    \\||    \\||   i
               g    \\2------3    g
                t      width.     h
                 h.               t.

        Args:
            img: Numpy array of shape (M,N,3)
            corners: Numpy array of shape (8,3) in camera coordinate frame.
            planes: Iterable of 5 clipping planes. Each plane is defined by 4 points.
            camera_config: CameraConfig object
            colors: tuple of RGB 3-tuples, Colors for front, side & rear.
                defaults are    0. blue (0,0,255) in RGB and (255,0,0) in OpenCV's BGR
                                1. red (255,0,0) in RGB and (0,0,255) in OpenCV's BGR
                                2. green (0,255,0) in RGB and BGR alike.
            linewidth: integer, linewidth for plot

        Returns:
            img: Numpy array of shape (M,N,3), representing updated image
        """

        def rectangle_corners(
            selected_corners: np.ndarray) -> List[List[np.ndarray]]:
            new_corners = []
            prev = selected_corners[-1]
            for corner in selected_corners:
                clipped_corners = clip_line_segment(
                    prev.copy(),
                    corner.copy(),
                    camera_config,
                    planes,
                )
                prev = corner
                new_corners.extend([clipped_corners])
            return new_corners

        patch_corners = []
        ww_f, ww_b = ww

        if central_patch:
            central_corners = ww_f*corners[:4] + (1. - ww_f)*corners[4:]
            patch_corners += rectangle_corners(central_corners)
            central_corners = (1. - ww_b)*corners[:4] + ww_b*corners[4:]
            patch_corners += rectangle_corners(central_corners)
        else:
            # Draw the sides in green
            for i in range(4):
                # between front and back corners
                clipped_corners = clip_line_segment(
                    corners[i],
                    corners[i + 4],
                    camera_config,
                    planes
                )
                patch_corners.extend([clipped_corners])

            # Draw front (first 4 corners) in blue
            patch_corners += rectangle_corners(corners[:4])
            # Draw rear (last 4 corners) in red
            patch_corners += rectangle_corners(corners[4:])

        # Remove corners non-existent inside the frustrum view
        _patch_corners = []
        for corner in patch_corners:
            if not corner is None:
                _patch_corners.extend([corner])

        if not _patch_corners.__len__():
            return None

        patch_corners = np.array(_patch_corners, dtype=np.float32)
        patch_corners = np.concatenate((patch_corners[:, 0], patch_corners[:, 1]), axis=0)

        h, w, _ = img.shape

        patch_boundary = {
            'tr' : [np.max(patch_corners[:, 0]), np.min(patch_corners[:, 1])],
            'bl' : [np.min(patch_corners[:, 0]), np.max(patch_corners[:, 1])]
            }

        patch_boundary['tr'][0] = int(min(patch_boundary['tr'][0], w))
        patch_boundary['tr'][0] = max(patch_boundary['tr'][0], 0)
        patch_boundary['tr'][1] = int(max(patch_boundary['tr'][1], 0))
        patch_boundary['tr'][1] = min(patch_boundary['tr'][1], h)


        patch_boundary['bl'][0] = int(max(patch_boundary['bl'][0], 0))
        patch_boundary['bl'][0] = min(patch_boundary['bl'][0], w)
        patch_boundary['bl'][1] = int(min(patch_boundary['bl'][1], h))
        patch_boundary['bl'][1] = max(patch_boundary['bl'][1], 0)

        # print(patch_boundary, img.shape)

        assert(uv_coord_is_valid(patch_boundary['tr'], img) and uv_coord_is_valid(patch_boundary['bl'], img))

        img_patch = img[patch_boundary['tr'][1]:patch_boundary['bl'][1], patch_boundary['bl'][0]:patch_boundary['tr'][0], :]

        return img_patch

def uv_coord_is_valid(uv: np.ndarray, img: np.ndarray) -> bool:
    """Check if 2d-point lies within 3-channel color image boundaries"""
    h, w, _ = img.shape
    return bool(uv[0] >= 0 and uv[1] >= 0 and uv[0] <= w and uv[1] <= h)
