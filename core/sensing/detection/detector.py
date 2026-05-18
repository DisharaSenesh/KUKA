"""Lightweight detector interface.

A Detector consumes a Frame and yields a list of Detections.
Implementations are free to use any algorithm (ArUco, blob,
deep-learning, …) as long as they produce image-space results.
"""

from __future__ import annotations

from typing import List

from ..camera.frame import Frame
from .detections import Detection


class BaseDetector:
    """Minimal interface for image-space target detectors.

    Subclasses override ``detect`` to implement a specific
    detection algorithm.

    Notes
    -----
    - The detector MUST NOT produce world coordinates or rays.
    - Returned ``Detection`` objects live entirely in pixel space.
    """

    def detect(self, frame: Frame) -> List[Detection]:
        """Detect targets in a single frame.

        Parameters
        ----------
        frame : Frame
            Input image frame with timestamp.

        Returns
        -------
        list of Detection
            Detections in image space.  May be empty.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement detect()"
        )

    def __call__(self, frame: Frame) -> List[Detection]:
        """Convenience: make the detector callable."""
        return self.detect(frame)
