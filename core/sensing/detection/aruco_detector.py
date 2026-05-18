"""ArUco fiducial marker detection.

Produces image-space Detection objects from ArUco markers
visible in the camera frame.  Marker centres are returned
as (u, v) pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import cv2

from ..camera.frame import Frame
from .detections import Detection
from .detector import BaseDetector


@dataclass
class ArUcoDetector(BaseDetector):
    """Detect ArUco markers in image frames.

    Uses OpenCV's ArUco module.  Each detected marker is
    reduced to its image-centre pixel coordinate.

    Attributes
    ----------
    dictionary : int
        ArUco dictionary enum, e.g. ``cv2.aruco.DICT_4X4_50``.
    marker_size_mm : float
        Physical marker size.  Stored for reference; NOT used
        during detection (detection is image-space only).
    """

    dictionary: int = cv2.aruco.DICT_4X4_50
    marker_size_mm: float = 50.0

    # Internal OpenCV objects — created lazily on first detect()
    _aruco_dict: object = field(default=None, init=False, repr=False)
    _aruco_params: object = field(default=None, init=False, repr=False)

    def _ensure_initialised(self) -> None:
        """Create OpenCV ArUco objects on first use."""
        if self._aruco_dict is not None:
            return

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)

        self._aruco_params = cv2.aruco.DetectorParameters()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> List[Detection]:
        """Find ArUco markers in a frame and return their image centres.

        Parameters
        ----------
        frame : Frame
            Input image frame.

        Returns
        -------
        list of Detection
            One Detection per visible marker.  The detection
            ``u``, ``v`` are the marker centre in pixels.
            ``marker_id`` carries the ArUco marker ID.
            ``metadata`` stores the raw corner positions.
        """
        self._ensure_initialised()

        gray = frame.image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray,
            self._aruco_dict,
            parameters=self._aruco_params,
        )

        detections: List[Detection] = []

        if ids is None or len(ids) == 0:
            return detections

        for i in range(len(ids)):
            marker_id = int(ids[i][0])

            # corners[i] has shape (1, 4, 2) — the four corners
            corner_set = corners[i][0]  # shape (4, 2)

            # Marker centre as the mean of the four corner points
            u = float(np.mean(corner_set[:, 0]))
            v = float(np.mean(corner_set[:, 1]))

            # Axis-aligned bounding box from the four corners
            x_min = float(np.min(corner_set[:, 0]))
            y_min = float(np.min(corner_set[:, 1]))
            x_max = float(np.max(corner_set[:, 0]))
            y_max = float(np.max(corner_set[:, 1]))
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            detections.append(
                Detection(
                    u=u,
                    v=v,
                    confidence=1.0,
                    timestamp=frame.timestamp,
                    marker_id=marker_id,
                    bbox=bbox,
                    metadata={
                        "corners": corner_set.tolist(),
                        "marker_size_mm": self.marker_size_mm,
                    },
                )
            )

        return detections
