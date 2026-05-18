"""Detection dataclass — image-space observation of a target.

A Detection records where in the image a target was found
and with what confidence.  It is intentionally restricted to
image space: it carries NO world coordinates, NO rays, and
NO geometry assumptions.

Conversion from (u, v) pixel to a world ray is the job of
the geometry layer, not the detection layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Detection:
    """An image-space detection of a target.

    Attributes
    ----------
    u : float
        Horizontal pixel coordinate of the detection centre.
        Origin at the top-left corner, increasing rightwards.
    v : float
        Vertical pixel coordinate of the detection centre.
        Origin at the top-left corner, increasing downwards.
    confidence : float
        Detection confidence in [0, 1].  Higher values mean
        the detector is more certain this is a true target.
    timestamp : float
        Acquisition time in seconds, tied to the frame timestamp.
    marker_id : int or None
        ArUco / fiducial marker ID, if this detection came from
        a marker-based detector.  None for generic detections.
    bbox : tuple of float or None
        Axis-aligned bounding box (x_min, y_min, width, height)
        in pixel coordinates.  None if the detector does not
        provide a bounding box.
    metadata : dict or None
        Detector-specific extra information (e.g. marker corners,
        blob size).  Consumers should treat this as opaque.
    """

    u: float
    v: float
    confidence: float
    timestamp: float
    marker_id: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Optional[Dict[str, Any]] = field(default=None, compare=False)

    def __post_init__(self):
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")

    @property
    def pixel(self) -> Tuple[float, float]:
        """The (u, v) pixel coordinate as a plain tuple.

        Returns
        -------
        (float, float)
        """
        return (self.u, self.v)

    @property
    def centre(self) -> Tuple[float, float]:
        """Alias for ``pixel``.  The detection centre in image space."""
        return self.pixel
