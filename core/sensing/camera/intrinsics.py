"""Camera intrinsics dataclass and serialization utilities.

Intrinsics encode the mapping from camera-normalised coordinates
to pixel coordinates via the pinhole projection model:

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

Distortion is modelled using the OpenCV radial-tangential
parameterisation (k1, k2, p1, p2, k3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics with optional distortion.

    Attributes
    ----------
    fx : float
        Focal length in horizontal pixels.
    fy : float
        Focal length in vertical pixels.
    cx : float
        Principal point horizontal coordinate (pixels).
    cy : float
        Principal point vertical coordinate (pixels).
    distortion : np.ndarray or None
        Distortion coefficients [k1, k2, p1, p2, k3] following
        the OpenCV convention. None for an ideal pinhole camera.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.distortion is not None:
            d = np.asarray(self.distortion, dtype=np.float64)
            object.__setattr__(self, "distortion", d)

    @property
    def camera_matrix(self) -> np.ndarray:
        """The 3x3 intrinsic camera matrix K.

        Returns
        -------
        K : np.ndarray
            shape (3, 3), float64

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        K = np.zeros((3, 3), dtype=np.float64)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K

    @property
    def image_size(self) -> tuple[float, float]:
        """Approximate image dimensions implied by the principal point.

        Note
        ----
        This gives (2*cx, 2*cy) which is exact when the principal point
        is at the image centre.
        """
        return (2.0 * self.cx, 2.0 * self.cy)

    def save(self, path: str) -> None:
        """Save intrinsics to a .npz file.

        Parameters
        ----------
        path : str
            File path for the saved intrinsics.
        """
        data = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }
        if self.distortion is not None:
            data["distortion"] = self.distortion
        np.savez(path, **data)

    @staticmethod
    def load(path: str) -> "CameraIntrinsics":
        """Load intrinsics from a .npz file.

        Parameters
        ----------
        path : str
            File path to the saved intrinsics.

        Returns
        -------
        CameraIntrinsics
        """
        data = np.load(path)
        distortion = None
        if "distortion" in data:
            distortion = data["distortion"]
        return CameraIntrinsics(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            distortion=distortion,
        )

    def to_opencv(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (camera_matrix, dist_coeffs) in OpenCV form.

        Returns
        -------
        K : np.ndarray
            shape (3, 3), float64
        dist : np.ndarray or None
            shape (5,) or None
        """
        return self.camera_matrix, self.distortion
