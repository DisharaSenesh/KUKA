"""
Pose: camera extrinsics in world frame.

A camera pose is defined by:
  - R: 3x3 rotation matrix (camera-to-world)
  - C: 3x1 camera center in world coordinates
  - t: observation timestamp

World-to-camera transformation:
  X_cam = R @ (X_world - C)

Camera-to-world transformation:
  X_world = C + R.T @ X_cam
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """
    Camera pose in world coordinate frame.

    Attributes
    ----------
    R : np.ndarray, shape (3, 3)
        Rotation matrix: maps vectors from camera-frame to world-frame.
        Satisfies R @ R.T = I and det(R) = 1 (SO(3)).
    C : np.ndarray, shape (3,)
        Camera optical center expressed in world coordinates.
    t : float
        Timestamp associated with this pose (seconds).
    """

    R: np.ndarray
    C: np.ndarray
    t: float

    def __post_init__(self):
        object.__setattr__(self, "R", np.asarray(self.R, dtype=np.float64))
        object.__setattr__(self, "C", np.asarray(self.C, dtype=np.float64))

    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point from world frame to camera frame.

        X_cam = R @ (X_world - C)

        Parameters
        ----------
        point_world : np.ndarray, shape (3,)
            Point in world coordinates.

        Returns
        -------
        np.ndarray, shape (3,)
            Point in camera coordinates.
        """
        point_world = np.asarray(point_world, dtype=np.float64)
        return self.R @ (point_world - self.C)

    def camera_to_world(self, point_cam: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point from camera frame to world frame.

        X_world = C + R.T @ X_cam

        Parameters
        ----------
        point_cam : np.ndarray, shape (3,)
            Point in camera coordinates.

        Returns
        -------
        np.ndarray, shape (3,)
            Point in world coordinates.
        """
        point_cam = np.asarray(point_cam, dtype=np.float64)
        return self.C + self.R.T @ point_cam

    def camera_ray_to_world(self, ray_cam: np.ndarray) -> np.ndarray:
        """
        Transform a direction vector from camera frame to world frame.

        This rotates the ray direction only (no translation of direction).

        d_world = R.T @ d_cam

        Parameters
        ----------
        ray_cam : np.ndarray, shape (3,)
            Direction vector in camera coordinates.

        Returns
        -------
        np.ndarray, shape (3,)
            Direction vector in world coordinates.
        """
        ray_cam = np.asarray(ray_cam, dtype=np.float64)
        return self.R.T @ ray_cam
