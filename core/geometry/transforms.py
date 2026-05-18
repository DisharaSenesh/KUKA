"""
Coordinate transforms: convert points and rays between camera and world frames.

All transformations are explicit about the source and destination frames.

World <-> Camera point transforms:
  X_cam = R @ (X_world - C)
  X_world = C + R.T @ X_cam

Ray direction transform (vectors, not points):
  d_world = R.T @ d_cam
  d_cam = R @ d_world
"""

import numpy as np

from core.types.pose import Pose


def transform_point_world_to_camera(point_world: np.ndarray, pose: Pose) -> np.ndarray:
    """
    Transform a 3D point from world frame to camera frame.

    X_cam = R @ (X_world - C)

    Parameters
    ----------
    point_world : np.ndarray, shape (3,)
        Point in world coordinates.
    pose : Pose
        Camera pose with rotation R and center C.

    Returns
    -------
    np.ndarray, shape (3,)
        Point in camera coordinates.
    """
    point_world = np.asarray(point_world, dtype=np.float64)
    return pose.R @ (point_world - pose.C)


def transform_point_camera_to_world(point_cam: np.ndarray, pose: Pose) -> np.ndarray:
    """
    Transform a 3D point from camera frame to world frame.

    X_world = C + R.T @ X_cam

    Parameters
    ----------
    point_cam : np.ndarray, shape (3,)
        Point in camera coordinates.
    pose : Pose
        Camera pose with rotation R and center C.

    Returns
    -------
    np.ndarray, shape (3,)
        Point in world coordinates.
    """
    point_cam = np.asarray(point_cam, dtype=np.float64)
    return pose.C + pose.R.T @ point_cam


def transform_ray_camera_to_world(ray_cam: np.ndarray, pose: Pose) -> np.ndarray:
    """
    Transform a direction vector from camera frame to world frame.

    Directions are vectors (not points), so only rotation applies:
    d_world = R.T @ d_cam

    Parameters
    ----------
    ray_cam : np.ndarray, shape (3,)
        Direction vector in camera frame.
    pose : Pose
        Camera pose.

    Returns
    -------
    np.ndarray, shape (3,)
        Direction vector in world frame (not renormalized).
    """
    ray_cam = np.asarray(ray_cam, dtype=np.float64)
    return pose.R.T @ ray_cam
