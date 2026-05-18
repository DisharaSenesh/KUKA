"""Frame dataclass for camera image acquisition.

A Frame bundles raw image data with acquisition metadata.
It is the output of the camera acquisition layer and the
input to the detection layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """A single image frame acquired from a camera.

    The frame carries the raw pixel data together with the
    moment of acquisition and a monotonic identifier.

    Attributes
    ----------
    image : np.ndarray
        Raw image array in OpenCV format (H, W) for grayscale
        or (H, W, C) for colour. Pixel values are uint8.
    timestamp : float
        Acquisition timestamp in seconds. Interpreted relative
        to the clock of the acquisition system.
    frame_id : int
        Monotonically increasing frame counter. Starts at 0
        when the camera stream is opened.
    """

    image: np.ndarray
    timestamp: float
    frame_id: int

    def __post_init__(self):
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
