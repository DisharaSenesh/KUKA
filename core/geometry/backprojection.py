"""
Backprojection: convert pixel coordinates into direction rays.

A pixel (u, v) with known camera intrinsics defines a ray from the
camera center through the image plane.

Camera-frame normalized ray:

    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy

    d_cam = [x_norm, y_norm, 1]    (unnormalized)
    d_cam = d_cam / ‖d_cam‖        (unit direction)

The world-frame ray is obtained by rotating d_cam into world coordinates
and setting the origin to the camera center C.
"""

import numpy as np

from core.types.pose import Pose
from core.types.ray import Ray


def pixel_to_camera_ray(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> Ray:
    """
    Backproject a pixel into a camera-frame direction ray.

    Parameters
    ----------
    u : float
        Horizontal pixel coordinate.
    v : float
        Vertical pixel coordinate.
    fx : float
        Focal length in pixels (x-direction).
    fy : float
        Focal length in pixels (y-direction).
    cx : float
        Principal point x-coordinate in pixels.
    cy : float
        Principal point y-coordinate in pixels.

    Returns
    -------
    Ray
        Ray in camera frame with origin at (0,0,0) and unit direction.
    """
    # Convert pixel into normalized camera coordinates
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy

    # Unnormalized direction vector in camera frame
    d_unnorm = np.array([x_norm, y_norm, 1.0], dtype=np.float64)

    # Normalize to unit direction
    direction = d_unnorm / np.linalg.norm(d_unnorm)

    # Origin is camera center in camera frame (always at origin)
    origin = np.zeros(3, dtype=np.float64)

    return Ray(origin=origin, direction=direction, frame="camera")


def pixel_to_world_ray(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pose: Pose
) -> Ray:
    """
    Backproject a pixel into a world-frame ray using camera pose.

    Combines:
      1. pixel -> camera-frame direction
      2. camera-frame direction -> world-frame direction (rotate only)
      3. origin = camera center C (world coordinates)

    Parameters
    ----------
    u, v : float
        Pixel coordinates.
    fx, fy, cx, cy : float
        Camera intrinsic parameters.
    pose : Pose
        Camera pose (rotation matrix R and center C).

    Returns
    -------
    Ray
        World-frame ray: origin = C, direction = R.T @ d_cam (unit).
    """
    # Step 1: backproject pixel to camera-frame ray
    ray_cam = pixel_to_camera_ray(u, v, fx, fy, cx, cy)

    # Step 2: rotate direction into world frame (no translation for directions)
    d_world = pose.camera_ray_to_world(ray_cam.direction)

    # Step 3: origin is the camera center in world coordinates
    origin = pose.C.copy()

    return Ray(origin=origin, direction=d_world, frame="world")
