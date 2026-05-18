"""
Projection: map a 3D camera-frame point to image pixel coordinates.

Standard pinhole camera model:

  u = fx * (X / Z) + cx
  v = fy * (Y / Z) + cy

where (X, Y, Z) are coordinates in the camera frame (Z > 0 is forward).
"""

import numpy as np


def project_to_pixel(
    point_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Project a 3D camera-frame point to 2D pixel coordinates.

    Parameters
    ----------
    point_cam : np.ndarray, shape (3,)
        Point in camera frame: [X, Y, Z] with Z > 0 (in front of camera).
    fx : float
        Focal length in pixels (x-direction).
    fy : float
        Focal length in pixels (y-direction).
    cx : float
        Principal point x-coordinate (pixels).
    cy : float
        Principal point y-coordinate (pixels).

    Returns
    -------
    np.ndarray, shape (2,)
        Pixel coordinates [u, v].

    Notes
    -----
    Assumes the point is in front of the camera (Z > 0).
    """
    point_cam = np.asarray(point_cam, dtype=np.float64)
    x_cam, y_cam, z_cam = point_cam[0], point_cam[1], point_cam[2]

    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    return np.array([u, v], dtype=np.float64)
