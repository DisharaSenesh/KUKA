"""
Measurement: a single monocular observation of the target.

Each measurement bundles:
  - a world-frame ray from the camera through the target
  - the timestamp of observation
  - optionally, the raw pixel observation and camera intrinsics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ray import Ray


@dataclass(frozen=True)
class PixelObservation:
    """
    Raw image-plane measurement with associated intrinsics.

    Attributes
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
    """

    u: float
    v: float
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Measurement:
    """
    A single monocular measurement of the target.

    The ray is expressed in world coordinates and passes through
    the target's true 3D position at timestamp t.

    Attributes
    ----------
    ray : Ray
        World-frame ray from camera center through the target.
    t : float
        Timestamp of this observation (seconds).
    pixel : Optional[PixelObservation]
        The raw pixel observation and intrinsics, if available.
    """

    ray: Ray
    t: float
    pixel: Optional[PixelObservation] = None
