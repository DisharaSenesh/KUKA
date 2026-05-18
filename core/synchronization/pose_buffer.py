"""
PoseBuffer: rolling time-ordered buffer of robot pose samples.

Maintains a bounded deque of pose entries, each containing:
  - (X, Y, Z) in mm
  - (A, B, C) in degrees (KUKA Euler angles)
  - timestamp (perf_counter, seconds)

Supports:
  - nearest-neighbor temporal lookup
  - bracketing-pair retrieval (for interpolation)
  - bounded memory (oldest evicted when full)

This is a pure data structure — no threading, no queues.
Used by both the synchronization thread and the synchronizer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimedPose:
    """
    A single robot pose sample with timestamp.

    Stores raw KUKA coordinates directly:
      - positions in mm
      - euler angles in degrees

    This is intentionally kept in robot-native units.
    Conversion to meters/radians happens in the geometry layer.

    Attributes
    ----------
    X_mm, Y_mm, Z_mm : float
        TCP position in robot base frame (millimeters).
    A_deg, B_deg, C_deg : float
        KUKA Euler angles (degrees, Z-Y-X intrinsic).
    timestamp : float
        Pose acquisition time (time.perf_counter, seconds).
    """

    X_mm: float
    Y_mm: float
    Z_mm: float
    A_deg: float
    B_deg: float
    C_deg: float
    timestamp: float


class PoseBuffer:
    """
    Rolling buffer of robot poses, ordered by timestamp.

    Provides O(n) nearest-neighbor lookup over the buffer.
    For typical sizes (~200 entries), linear scan is fast enough.
    Use binary search for very large buffers.

    Attributes
    ----------
    max_size : int
        Maximum number of poses retained. Oldest are evicted.
    _poses : deque
        Time-ordered pose entries (oldest left, newest right).
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._poses: deque = deque()

    def push(self, X_mm: float, Y_mm: float, Z_mm: float,
             A_deg: float, B_deg: float, C_deg: float,
             timestamp: float) -> None:
        """
        Add a robot pose sample to the buffer.

        Assumes poses arrive in monotonically increasing time order.
        Out-of-order samples are silently discarded.

        Parameters
        ----------
        X_mm, Y_mm, Z_mm : float
            Position in mm.
        A_deg, B_deg, C_deg : float
            Orientation in degrees.
        timestamp : float
            Pose acquisition timestamp.
        """
        # Enforce time ordering (skip out-of-order)
        if len(self._poses) > 0 and timestamp < self._poses[-1].timestamp:
            return

        entry = TimedPose(
            X_mm=X_mm, Y_mm=Y_mm, Z_mm=Z_mm,
            A_deg=A_deg, B_deg=B_deg, C_deg=C_deg,
            timestamp=timestamp,
        )
        self._poses.append(entry)

        # Evict oldest if over capacity
        while len(self._poses) > self.max_size:
            self._poses.popleft()

    def find_nearest(self, target_time: float) -> Optional[TimedPose]:
        """
        Find the pose entry closest in time to target_time.

        Linear scan over the buffer.

        Parameters
        ----------
        target_time : float
            Target timestamp (seconds).

        Returns
        -------
        TimedPose or None
            Nearest pose, or None if buffer is empty.
        """
        if len(self._poses) == 0:
            return None

        best = None
        best_dt = float("inf")

        for entry in self._poses:
            dt = abs(entry.timestamp - target_time)
            if dt < best_dt:
                best_dt = dt
                best = entry

        return best

    def find_bracketing(
        self, target_time: float
    ) -> tuple:
        """
        Find the two poses that bracket target_time.

        Returns (before, after) where:
          before.timestamp ≤ target_time ≤ after.timestamp

        For extrapolation (target before first pose or after last):
          Returns (None, first_pose) or (last_pose, None) respectively.

        Parameters
        ----------
        target_time : float
            Target timestamp.

        Returns
        -------
        tuple of (TimedPose or None, TimedPose or None)
            Bracketing pose pair.
        """
        if len(self._poses) == 0:
            return (None, None)

        before = None
        after = None

        for entry in self._poses:
            if entry.timestamp <= target_time:
                before = entry
            if entry.timestamp >= target_time and after is None:
                after = entry

        return (before, after)

    def latest(self) -> Optional[TimedPose]:
        """Return the most recent pose in the buffer."""
        if len(self._poses) == 0:
            return None
        return self._poses[-1]

    def earliest(self) -> Optional[TimedPose]:
        """Return the oldest pose in the buffer."""
        if len(self._poses) == 0:
            return None
        return self._poses[0]

    def __len__(self) -> int:
        return len(self._poses)
