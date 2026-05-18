"""
Process-layer types: detection objects and re-exports of canonical types.

The Detection type lives here (image-space only, from the camera process).
The canonical SynchronizedMeasurement is defined in core.synchronization
and re-exported here for convenience.

Pipeline:

  Camera Process → Detection         → detection_queue
  Robot Process  → (X,Y,Z,A,B,C,t)  → pose_queue
                                          ↓
  Synchronization Thread ← consumes both queues
     (uses Synchronizer from core.synchronization)
                                          ↓
  SynchronizedMeasurement → sync_queue → tracking / CSV recording
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export the canonical synchronized measurement type
from core.synchronization.synchronized_measurement import (
    SynchronizedMeasurement,
    CSV_COLUMNS,
)

# Re-export the synchronizer and its diagnostics
from core.synchronization.synchronizer import Synchronizer, SyncDiagnostics

# Re-export pose buffer types
from core.synchronization.pose_buffer import TimedPose


@dataclass(frozen=True)
class Detection:
    """
    A raw monocular detection from the camera process.

    Contains the pixel observation and the timestamp at which the
    frame was acquired. No geometry, world coordinates, or rays.

    Attributes
    ----------
    u : float
        Horizontal pixel coordinate of the detected target center.
    v : float
        Vertical pixel coordinate of the detected target center.
    t : float
        Frame acquisition timestamp (time.perf_counter, seconds).
    fx, fy, cx, cy : float
        Camera intrinsic parameters (carried through from calibration).
    frame_id : int
        Monotonically increasing frame counter for diagnostics.
    """

    u: float
    v: float
    t: float
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    frame_id: int = 0


# ---- Raw robot pose (for inter-process communication) ----

@dataclass(frozen=True)
class RawRobotPose:
    """
    Raw robot TCP pose for inter-process queue transfer.

    Carries the KUKA-native representation:
      - position in mm (X, Y, Z)
      - orientation in degrees (A, B, C)
      - timestamp (time.perf_counter)

    This is the data type that flows through the pose_queue.
    The sync thread converts it to TimedPose entries in the buffer.
    """
    X_mm: float
    Y_mm: float
    Z_mm: float
    A_deg: float
    B_deg: float
    C_deg: float
    timestamp: float
