"""
Synchronizer: high-level temporal alignment orchestrator.

Takes individual detections and a pose buffer, and produces
SynchronizedMeasurement objects. This is the primary interface
for creating synchronized data — whether from live processes
or from replay.

Design:
  - Pure orchestrator: owns a PoseBuffer and a matching strategy
  - accept_pose() feeds the buffer
  - synchronize() aligns a detection with the buffer
  - Stateless between calls (the buffer is the only state)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

from .pose_buffer import PoseBuffer, TimedPose
from .matcher import match_nearest, match_linear_translation
from .synchronized_measurement import SynchronizedMeasurement


@dataclass
class SyncDiagnostics:
    """
    Rolling diagnostics for monitoring synchronization quality.

    Attributes
    ----------
    total_detections : int
        Cumulative count of detections fed to the synchronizer.
    total_synchronized : int
        Cumulative count of valid synchronized measurements produced.
    total_dropped : int
        Count of detections dropped due to sync failure (no pose, etc.).
    sync_errors_s : deque
        Rolling window of recent sync errors.
    max_history : int
        Maximum rolling window size.
    """

    total_detections: int = 0
    total_synchronized: int = 0
    total_dropped: int = 0
    sync_errors_s: deque = field(default_factory=deque)
    max_history: int = 1000

    def record_sync(self, error_s: float) -> None:
        self.sync_errors_s.append(error_s)
        while len(self.sync_errors_s) > self.max_history:
            self.sync_errors_s.popleft()

    def mean_error_s(self) -> float:
        if not self.sync_errors_s:
            return 0.0
        return sum(self.sync_errors_s) / len(self.sync_errors_s)

    def max_error_s(self) -> float:
        if not self.sync_errors_s:
            return 0.0
        return max(self.sync_errors_s)


@dataclass
class Synchronizer:
    """
    Temporal alignment engine for monocular observations.

    Maintains a rolling buffer of robot poses and matches incoming
    detections against it to produce SynchronizedMeasurement objects.

    Usage:

        sync = Synchronizer(use_interpolation=False)
        sync.accept_pose(X_mm=500, Y_mm=0, Z_mm=800,
                         A_deg=0, B_deg=45, C_deg=0, timestamp=1.0)
        result = sync.synchronize(
            frame_id=0, t=1.05, u=320.0, v=240.0,
            fx=600, fy=600, cx=320, cy=240
        )

    Attributes
    ----------
    pose_buffer : PoseBuffer
        Rolling buffer of robot pose samples.
    use_interpolation : bool
        If True, use linear translation interpolation.
        If False, use nearest-neighbor matching.
    sync_tolerance_s : float
        Maximum acceptable sync error. Exceeding detections are
        still produced but marked invalid.
    diagnostics : SyncDiagnostics
        Rolling metrics for monitoring.
    """

    pose_buffer: PoseBuffer = field(default_factory=lambda: PoseBuffer(max_size=200))
    use_interpolation: bool = False
    sync_tolerance_s: float = 0.05
    diagnostics: SyncDiagnostics = field(default_factory=SyncDiagnostics)

    def accept_pose(
        self,
        X_mm: float,
        Y_mm: float,
        Z_mm: float,
        A_deg: float,
        B_deg: float,
        C_deg: float,
        timestamp: float,
    ) -> None:
        """
        Feed a robot pose sample into the buffer.

        Parameters
        ----------
        X_mm, Y_mm, Z_mm : float
            Robot TCP position (mm, base frame).
        A_deg, B_deg, C_deg : float
            KUKA Euler angles (degrees).
        timestamp : float
            Pose acquisition time (time.perf_counter).
        """
        self.pose_buffer.push(
            X_mm=X_mm, Y_mm=Y_mm, Z_mm=Z_mm,
            A_deg=A_deg, B_deg=B_deg, C_deg=C_deg,
            timestamp=timestamp,
        )

    def synchronize(
        self,
        frame_id: int,
        u: float,
        v: float,
        t_frame: float,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ) -> SynchronizedMeasurement:
        """
        Align a detection with the pose buffer.

        Finds the robot pose closest to t_frame (or interpolates)
        and produces a SynchronizedMeasurement.

        Parameters
        ----------
        frame_id : int
            Frame counter.
        u, v : float
            Pixel coordinates of the detection.
        t_frame : float
            Frame acquisition timestamp.
        fx, fy, cx, cy : float
            Camera intrinsics.

        Returns
        -------
        SynchronizedMeasurement
            Always returns a measurement. Check .is_valid for quality.
        """
        self.diagnostics.total_detections += 1

        # ---- Match ----
        if self.use_interpolation:
            timed_pose, sync_error, method = match_linear_translation(
                self.pose_buffer, t_frame
            )
        else:
            timed_pose, sync_error, method = match_nearest(
                self.pose_buffer, t_frame
            )

        # ---- Build result ----
        if timed_pose is None:
            # No pose available — produce invalid measurement with zeros
            self.diagnostics.total_dropped += 1
            return SynchronizedMeasurement(
                frame_id=frame_id,
                timestamp=t_frame,
                u=u, v=v,
                X_mm=0.0, Y_mm=0.0, Z_mm=0.0,
                A_deg=0.0, B_deg=0.0, C_deg=0.0,
                sync_error_s=float("inf"),
                sync_method="none",
                is_valid=False,
                fx=fx, fy=fy, cx=cx, cy=cy,
            )

        is_valid = sync_error <= self.sync_tolerance_s

        if is_valid:
            self.diagnostics.total_synchronized += 1
        else:
            self.diagnostics.total_dropped += 1

        self.diagnostics.record_sync(sync_error)

        return SynchronizedMeasurement(
            frame_id=frame_id,
            timestamp=t_frame,
            u=u, v=v,
            X_mm=timed_pose.X_mm,
            Y_mm=timed_pose.Y_mm,
            Z_mm=timed_pose.Z_mm,
            A_deg=timed_pose.A_deg,
            B_deg=timed_pose.B_deg,
            C_deg=timed_pose.C_deg,
            sync_error_s=sync_error,
            sync_method=method,
            is_valid=is_valid,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
