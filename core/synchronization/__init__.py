"""
Synchronization module: temporal alignment of asynchronous sensor streams.

Produces the canonical SynchronizedMeasurement data type that all
downstream modules consume. This enables online/offline unification.

Architecture:
  - synchronized_measurement: canonical data type (frame_id, t, u, v, X, Y, Z, A, B, C, sync_error)
  - pose_buffer: rolling time-ordered buffer of robot pose samples
  - matcher: temporal alignment strategies (nearest-neighbor, linear interpolation)
  - synchronizer: high-level orchestrator (accept_pose + synchronize)
"""

from .synchronized_measurement import SynchronizedMeasurement, CSV_COLUMNS
from .pose_buffer import PoseBuffer, TimedPose
from .matcher import match_nearest, match_linear_translation
from .synchronizer import Synchronizer, SyncDiagnostics

__all__ = [
    "SynchronizedMeasurement",
    "CSV_COLUMNS",
    "PoseBuffer",
    "TimedPose",
    "match_nearest",
    "match_linear_translation",
    "Synchronizer",
    "SyncDiagnostics",
]
