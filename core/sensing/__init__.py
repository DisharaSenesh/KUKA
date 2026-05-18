"""Sensing layer — camera acquisition, detection, and timestamp management.

The sensing layer is responsible ONLY for:
- acquiring sensor data (frames from camera)
- producing image-space target detections
- managing timestamps and synchronisation

It MUST NOT perform trajectory optimisation, ray geometry,
tracking, or world-space reconstruction.
"""

from .camera.frame import Frame
from .camera.timestamps import (
    monotonic_timestamp,
    wall_timestamp,
    timestamp_generator,
    stamp_now,
)
from .camera.intrinsics import CameraIntrinsics
from .camera.capture import CameraCapture
from .camera.calibration import (
    CalibrationResult,
    calibrate_checkerboard,
    undistort_points,
)
from .detection.detections import Detection
from .detection.detector import BaseDetector
from .detection.aruco_detector import ArUcoDetector
from .detection.blob_detector import BlobDetector
from .synchronization.clock import Clock
from .synchronization.alignment import (
    nearest_timestamp,
    alignment_offset,
    align_series,
)
from .synchronization.buffering import SensorBuffer

__all__ = [
    "Frame",
    "CameraIntrinsics",
    "CameraCapture",
    "CalibrationResult",
    "calibrate_checkerboard",
    "undistort_points",
    "monotonic_timestamp",
    "wall_timestamp",
    "timestamp_generator",
    "stamp_now",
    "Detection",
    "BaseDetector",
    "ArUcoDetector",
    "BlobDetector",
    "Clock",
    "nearest_timestamp",
    "alignment_offset",
    "align_series",
    "SensorBuffer",
]
