from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BoundingBox3D:
    """3D bounding box, same as in BPS and PhoLocalizationGui"""
    corner: np.ndarray
    size: np.ndarray
    rotation_vec: np.ndarray

    @classmethod
    def create_empty(cls):
        return cls(corner=np.empty(3, dtype=np.int32),
                   size=np.empty(3, dtype=np.int32),
                   rotation_vec=np.empty(3, dtype=np.float64))

    def get_corners(self):
        tmp = [0, 1]
        bbox_3d_points = np.array([[x, y, z] for x in tmp for y in tmp for z in tmp], dtype=np.float32)
        bbox_3d_points *= self.size

        rotation = Rotation.from_rotvec(self.rotation_vec)
        bbox_3d_points = rotation.apply(bbox_3d_points)

        bbox_3d_points += self.corner
        return bbox_3d_points

    def check(self):
        assert self.corner.dtype == np.int32
        assert self.corner.shape == (3,)
        assert self.size.dtype == np.int32
        assert self.size.shape == (3,)
        assert self.rotation_vec.dtype == np.float64
        assert self.rotation_vec.shape == (3,)


@dataclass
class BoundingBox2D:
    """

    2D bounding box in image space coordinates.
    Used as a view to first 2 dimensions of numpy arrays.
    OpenCV uses conventions (height, width), (height, width, channels)
    for numpy arrays.

    Attributes:
        top_left: Start of first and second dimension (inclusive).
        bottom_right: End of first and second dimension (exclusive).

    """
    top_left: np.ndarray
    bottom_right: np.ndarray

    @classmethod
    def create_empty(cls):
        return cls(top_left=np.empty(2, dtype=np.int32),
                   bottom_right=np.empty(2, dtype=np.int32))

    def as_crop(self):
        # crop is [x1, y1, x2, y2]
        return np.concatenate((self.top_left[::-1], self.bottom_right[::-1]), axis=None)
