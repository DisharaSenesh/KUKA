"""Simple blob detector using OpenCV's SimpleBlobDetector.

Useful for tracking a single bright/dark circular target
when fiducial markers are not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from ..camera.frame import Frame
from .detections import Detection
from .detector import BaseDetector


@dataclass
class BlobDetector(BaseDetector):
    """Detect circular blobs in image frames.

    Wraps OpenCV's ``SimpleBlobDetector`` with sensible defaults
    for a bright circular target on a darker background.

    Notes
    -----
    This detector picks the single strongest blob.  For multi-blob
    scenarios, use ``detect_all`` to retrieve all candidates.

    Attributes
    ----------
    min_area : float
        Minimum blob area in pixels (default 100).
    max_area : float
        Maximum blob area in pixels (default 5000).
    min_circularity : float
        Minimum circularity [0, 1] (default 0.7).
    min_convexity : float
        Minimum convexity [0, 1] (default 0.8).
    """

    min_area: float = 100.0
    max_area: float = 5000.0
    min_circularity: float = 0.7
    min_convexity: float = 0.8
    blob_colour: int = 255  # 255 = bright blobs, 0 = dark blobs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_detector(self) -> cv2.SimpleBlobDetector:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.min_area
        params.maxArea = self.max_area
        params.filterByCircularity = True
        params.minCircularity = self.min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.min_convexity
        params.filterByColor = True
        params.blobColor = self.blob_colour
        return cv2.SimpleBlobDetector_create(params)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> List[Detection]:
        """Detect the single strongest blob and return a Detection.

        Parameters
        ----------
        frame : Frame
            Input image frame.

        Returns
        -------
        list of Detection
            At most one Detection for the strongest blob.
        """
        detector = self._create_detector()

        gray = frame.image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        keypoints = detector.detect(gray)

        if not keypoints:
            return []

        # Pick the keypoint with the largest response
        best = max(keypoints, key=lambda kp: kp.response)

        return [
            Detection(
                u=best.pt[0],
                v=best.pt[1],
                confidence=min(best.response / 100.0, 1.0),
                timestamp=frame.timestamp,
                bbox=(
                    best.pt[0] - best.size / 2,
                    best.pt[1] - best.size / 2,
                    best.size,
                    best.size,
                ),
                metadata={"size": best.size, "response": best.response},
            )
        ]

    def detect_all(self, frame: Frame) -> List[Detection]:
        """Return ALL detected blobs as Detections.

        Parameters
        ----------
        frame : Frame
            Input image frame.

        Returns
        -------
        list of Detection
            All detected blobs, sorted by response (strongest first).
        """
        detector = self._create_detector()

        gray = frame.image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        keypoints = detector.detect(gray)

        # Sort by response, strongest first
        keypoints.sort(key=lambda kp: kp.response, reverse=True)

        detections: List[Detection] = []
        for kp in keypoints:
            detections.append(
                Detection(
                    u=kp.pt[0],
                    v=kp.pt[1],
                    confidence=min(kp.response / 100.0, 1.0),
                    timestamp=frame.timestamp,
                    bbox=(
                        kp.pt[0] - kp.size / 2,
                        kp.pt[1] - kp.size / 2,
                        kp.size,
                        kp.size,
                    ),
                    metadata={"size": kp.size, "response": kp.response},
                )
            )
        return detections
