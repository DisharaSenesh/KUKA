"""
Reprojection error: measure the difference between observed pixel and
the projection of an estimated 3D point.

Given an estimated 3D world point X_world and a camera pose,
the reprojection error is:

    error = pixel_observed - project_to_pixel(pose.world_to_camera(X_world), K)

This is used for validation, not for the core trajectory optimization.
"""

import numpy as np

from core.types.pose import Pose
from .projection import project_to_pixel


def reprojection_error(
    point_world: np.ndarray,
    pixel_observed: np.ndarray,
    pose: Pose,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Compute the 2D reprojection error for a single point.

    Parameters
    ----------
    point_world : np.ndarray, shape (3,)
        Estimated 3D point in world coordinates.
    pixel_observed : np.ndarray, shape (2,)
        Observed pixel coordinates [u, v].
    pose : Pose
        Camera pose for this observation.
    fx, fy, cx, cy : float
        Camera intrinsic parameters.

    Returns
    -------
    np.ndarray, shape (2,)
        Reprojection error vector [du, dv] in pixels.
    """
    # Transform to camera frame
    point_cam = pose.world_to_camera(point_world)

    # Project to image plane
    pixel_projected = project_to_pixel(point_cam, fx, fy, cx, cy)

    # Compute error
    pixel_observed = np.asarray(pixel_observed, dtype=np.float64)
    return pixel_observed - pixel_projected
