"""
Ray: a geometric ray defined by an origin and unit direction.

A ray is parameterized as:
  X(λ) = origin + λ * direction

where λ >= 0 is the distance along the ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ray:
    """
    Geometric ray in a specified coordinate frame.

    Attributes
    ----------
    origin : np.ndarray, shape (3,)
        Starting point of the ray (e.g., camera center C).
    direction : np.ndarray, shape (3,)
        Unit-norm direction vector.
    frame : str
        Coordinate frame identifier: 'camera' or 'world'.
    """

    origin: np.ndarray
    direction: np.ndarray
    frame: str = "world"

    def __post_init__(self):
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64))
        object.__setattr__(self, "direction", np.asarray(self.direction, dtype=np.float64))

    def point_at(self, lam: float) -> np.ndarray:
        """
        Evaluate the point along the ray at distance λ.

        X = origin + λ * direction

        Parameters
        ----------
        lam : float
            Distance along ray (non-negative for forward ray).

        Returns
        -------
        np.ndarray, shape (3,)
            3D point at the specified distance.
        """
        return self.origin + lam * self.direction
