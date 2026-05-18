"""
Final output layer: standardized object pose estimate for downstream consumers.

This module is the single interface between the perception/tracking pipeline
and any future robotics modules (grasp planning, trajectory recording,
multi-object tracking, visualization).

The ObjectPoseEstimate is the CANONICAL output type. Every consumer
receives this identical structure.

Architecture:

    camera_process → tracker → sync → interpolation → pose estimation
                                                           ↓
                                                   ObjectPoseEstimate
                                                           ↓
                                              (grasp, record, visualize, ...)

Usage example:

    from core.output import ObjectPoseEstimate, TrackingState

    estimate = ObjectPoseEstimate(
        timestamp=1.234,
        frame_id=42,
        object_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.5, v=240.3,
        robot_pose=RobotPose6D(x=500.0, y=0.0, z=800.0, rx=0.0, ry=45.0, rz=0.0),
        object_pose_base=RobotPose6D(x=512.0, y=-5.0, z=825.0, rx=0.0, ry=45.0, rz=0.0),
        confidence=0.92,
    )

    # Serialize for downstream
    payload = estimate.to_dict()
    json_str = estimate.to_json()
    print(estimate.pretty())
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple


# ============================================================
# Tracking State
# ============================================================

class TrackingState(Enum):
    """
    Status of the current object tracking estimate.

    TRACKED        — estimate produced from valid synchronized measurements.
    LOST           — estimate unavailable (object not detected, tracking failed).
    INTERPOLATED   — estimate produced via temporal interpolation between tracked frames.
    INITIALIZING   — building initial measurement set, not yet solvable.
    INVALID        — estimate exists but fails quality checks (high residual, etc.).
    """
    TRACKED = auto()
    LOST = auto()
    INTERPOLATED = auto()
    INITIALIZING = auto()
    INVALID = auto()

    @classmethod
    def from_string(cls, s: str) -> "TrackingState":
        """Case-insensitive lookup from a string."""
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.INVALID


# ============================================================
# Robot Pose (6-DOF)
# ============================================================

@dataclass(frozen=True)
class RobotPose6D:
    """
    A 6-DOF pose in robot base frame.

    Position in millimeters, orientation in degrees (KUKA ABC convention).
    This is intentionally kept in robot-native units — the same units
    that the robot controller uses directly.

    Attributes
    ----------
    x, y, z : float
        Translation in millimeters (robot base frame).
    rx, ry, rz : float
        Rotation in degrees about X, Y, Z axes (KUKA ABC: A=rz, B=ry, C=rx).
    """

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return as a flat 6-tuple (x, y, z, rx, ry, rz)."""
        return (self.x, self.y, self.z, self.rx, self.ry, self.rz)

    def to_dict(self) -> Dict[str, float]:
        """Return as a dictionary with named keys."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rx": self.rx,
            "ry": self.ry,
            "rz": self.rz,
        }

    def is_valid(self) -> bool:
        """Check that all position values are finite."""
        return all(math.isfinite(v) for v in [self.x, self.y, self.z])


# ============================================================
# Object Pose Estimate — Canonical Output
# ============================================================

@dataclass
class ObjectPoseEstimate:
    """
    The canonical final output of the perception/tracking pipeline.

    Represents a single synchronized frame: the detected object's 2D
    pixel location, the interpolated robot pose at that instant, and
    the estimated object pose in the robot's base frame.

    This is the STANDARDIZED interface. Every downstream module
    (grasp planning, recording, visualization, multi-object tracking)
    consumes this type.

    Attributes
    ----------
    timestamp : float
        Frame acquisition time (time.perf_counter, seconds).
    frame_id : int
        Monotonically increasing frame counter.
    object_id : int
        Unique object identifier (for future multi-object support).
    tracking_state : TrackingState
        Status of the current estimate.
    u : float
        Horizontal pixel coordinate of the detected object center.
    v : float
        Vertical pixel coordinate of the detected object center.
    robot_pose : RobotPose6D or None
        Interpolated robot TCP pose at frame time (mm, degrees).
        None if no robot pose was available.
    object_pose_base : RobotPose6D or None
        Estimated object pose in the robot base frame (mm, degrees).
        None if tracking has not yet converged.
    confidence : float
        Tracking confidence in [0, 1]. Combines observability,
        reprojection quality, and measurement freshness.
    sync_error_s : float
        Time difference between frame and nearest robot pose (seconds).
    reprojection_error_px : float
        RMS reprojection error of the trajectory estimate (pixels).
        -1.0 if not available.
    polynomial_order : int
        Polynomial order used for the trajectory model. 0 if not available.
    window_size : int
        Number of measurements in the sliding window. 0 if not available.
    """

    timestamp: float
    frame_id: int
    object_id: int = 1
    tracking_state: TrackingState = TrackingState.INITIALIZING

    # Camera detection
    u: float = 0.0
    v: float = 0.0

    # Robot and object poses
    robot_pose: Optional[RobotPose6D] = None
    object_pose_base: Optional[RobotPose6D] = None

    # Quality metrics
    confidence: float = 0.0
    sync_error_s: float = 0.0
    reprojection_error_px: float = -1.0

    # Reconstruction metadata
    polynomial_order: int = 0
    window_size: int = 0

    # ============================================================
    # Validation
    # ============================================================

    def is_valid(self) -> bool:
        """
        Check whether this estimate is usable by downstream consumers.

        A valid estimate has:
          - tracking state TRACKED or INTERPOLATED
          - finite camera pixel coordinates
          - a finite robot pose
          - confidence > 0
        """
        if self.tracking_state in (TrackingState.LOST, TrackingState.INVALID):
            return False
        if not (math.isfinite(self.u) and math.isfinite(self.v)):
            return False
        if self.robot_pose is None or not self.robot_pose.is_valid():
            return False
        if self.confidence <= 0.0:
            return False
        return True

    def has_object_pose(self) -> bool:
        """True if the object pose in base frame is available and finite."""
        if self.object_pose_base is None:
            return False
        return self.object_pose_base.is_valid()

    # ============================================================
    # Serialization
    # ============================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a flat dictionary for JSON serialization.

        Returns
        -------
        dict
            All fields as simple Python types (float, int, str, list).
        """
        result: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "tracking_state": self.tracking_state.name.lower(),
            "camera_uv": [self.u, self.v],
            "confidence": self.confidence,
            "sync_error_s": self.sync_error_s,
            "reprojection_error_px": self.reprojection_error_px,
        }

        if self.robot_pose is not None:
            result["robot_pose"] = self.robot_pose.to_dict()
        else:
            result["robot_pose"] = None

        if self.object_pose_base is not None:
            result["object_pose_base"] = self.object_pose_base.to_dict()
        else:
            result["object_pose_base"] = None

        # Diagnostics (compact, for recording)
        result["diagnostics"] = {
            "polynomial_order": self.polynomial_order,
            "window_size": self.window_size,
        }

        return result

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize to a JSON string.

        Parameters
        ----------
        indent : int
            Indentation level for pretty-printing. Use 0 for compact.

        Returns
        -------
        str
            JSON-encoded string.
        """
        separators = None if indent > 0 else (",", ":")
        return json.dumps(
            self.to_dict(),
            indent=indent if indent > 0 else None,
            separators=separators,
            default=str,
        )

    def pretty(self) -> str:
        """
        Multi-line human-readable representation.

        Returns
        -------
        str
            Formatted string suitable for console/log output.
        """
        lines = [
            "=" * 54,
            f"  Object Pose Estimate  |  frame={self.frame_id:04d}  id={self.object_id}",
            "-" * 54,
            f"  timestamp  : {self.timestamp:.4f} s",
            f"  state      : {self.tracking_state.name}",
            f"  camera_uv  : ({self.u:.1f}, {self.v:.1f}) px",
            f"  confidence : {self.confidence:.3f}",
            f"  sync_error : {self.sync_error_s*1000:.2f} ms",
        ]

        if self.robot_pose is not None:
            rp = self.robot_pose
            lines.append(f"  robot_pose : ({rp.x:.1f}, {rp.y:.1f}, {rp.z:.1f}) mm  "
                         f"({rp.rx:.1f}, {rp.ry:.1f}, {rp.rz:.1f}) deg")
        else:
            lines.append("  robot_pose : None")

        if self.object_pose_base is not None:
            op = self.object_pose_base
            lines.append(f"  object_pose: ({op.x:.1f}, {op.y:.1f}, {op.z:.1f}) mm  "
                         f"({op.rx:.1f}, {op.ry:.1f}, {op.rz:.1f}) deg")
        else:
            lines.append("  object_pose: None")

        if self.reprojection_error_px >= 0:
            lines.append(f"  reproj_err : {self.reprojection_error_px:.2f} px")
            lines.append(f"  poly_order : {self.polynomial_order}")
            lines.append(f"  window     : {self.window_size} meas")

        lines.append("=" * 54)
        return "\n".join(lines)

    def summary_line(self) -> str:
        """
        One-line summary for log streaming.

        Returns
        -------
        str
            Compact single-line representation.
        """
        state_abbr = self.tracking_state.name[:4]
        uv = f"({self.u:.0f},{self.v:.0f})"
        conf = f"c={self.confidence:.2f}"
        err = f"e={self.sync_error_s*1000:.0f}ms"

        if self.robot_pose is not None:
            rp = self.robot_pose
            robot = f"R=({rp.x:.0f},{rp.y:.0f},{rp.z:.0f})"
        else:
            robot = "R=None"

        if self.object_pose_base is not None:
            op = self.object_pose_base
            obj = f"O=({op.x:.0f},{op.y:.0f},{op.z:.0f})"
        else:
            obj = "O=None"

        return f"[{self.frame_id:04d}] {state_abbr} {uv} {conf} {err} {robot} {obj}"

    # ============================================================
    # Factory constructors for convenience
    # ============================================================

    @classmethod
    def lost(cls, frame_id: int, timestamp: float, object_id: int = 1) -> "ObjectPoseEstimate":
        """Create an estimate representing a lost/dropped track."""
        return cls(
            timestamp=timestamp,
            frame_id=frame_id,
            object_id=object_id,
            tracking_state=TrackingState.LOST,
            confidence=0.0,
        )

    @classmethod
    def initializing(cls, frame_id: int, timestamp: float,
                     u: float = 0.0, v: float = 0.0,
                     object_id: int = 1) -> "ObjectPoseEstimate":
        """Create an estimate while building initial measurements."""
        return cls(
            timestamp=timestamp,
            frame_id=frame_id,
            object_id=object_id,
            tracking_state=TrackingState.INITIALIZING,
            u=u, v=v,
            confidence=0.1,
        )

    @classmethod
    def interpolated(cls, frame_id: int, timestamp: float,
                     robot_pose: RobotPose6D,
                     object_pose_base: Optional[RobotPose6D] = None,
                     object_id: int = 1) -> "ObjectPoseEstimate":
        """Create an interpolated estimate between tracked frames."""
        return cls(
            timestamp=timestamp,
            frame_id=frame_id,
            object_id=object_id,
            tracking_state=TrackingState.INTERPOLATED,
            robot_pose=robot_pose,
            object_pose_base=object_pose_base,
            confidence=0.5,
        )


# ============================================================
# Builder — constructs estimates from pipeline output
# ============================================================

@dataclass
class OutputBuilder:
    """
    Convenience builder for constructing ObjectPoseEstimate from
    pipeline objects (SynchronizedMeasurement, TrajectoryEstimate).

    Keeps output construction separate from the data type itself
    for modularity. The builder can be extended for multi-object
    or multi-sensor configurations.

    Usage:

        builder = OutputBuilder()
        estimate = builder.build_from_sync(sync_meas)
        estimate = builder.build_from_track(sync_meas, trajectory_est)
    """

    def build_from_sync(
        self,
        sync_meas: "SynchronizedMeasurement",
        object_id: int = 1,
    ) -> ObjectPoseEstimate:
        """
        Build an estimate from a synchronized measurement only
        (before trajectory reconstruction is available).

        Object pose is not yet computed; tracking state is INITIALIZING.

        Parameters
        ----------
        sync_meas : SynchronizedMeasurement
            The synchronized measurement from the pipeline.
        object_id : int
            Object identifier.

        Returns
        -------
        ObjectPoseEstimate
        """
        robot_pose = RobotPose6D(
            x=sync_meas.X_mm,
            y=sync_meas.Y_mm,
            z=sync_meas.Z_mm,
            rx=sync_meas.C_deg,
            ry=sync_meas.B_deg,
            rz=sync_meas.A_deg,
        )

        return ObjectPoseEstimate(
            timestamp=sync_meas.timestamp,
            frame_id=sync_meas.frame_id,
            object_id=object_id,
            tracking_state=TrackingState.INITIALIZING,
            u=sync_meas.u,
            v=sync_meas.v,
            robot_pose=robot_pose,
            sync_error_s=sync_meas.sync_error_s,
            confidence=0.1 if sync_meas.is_valid else 0.0,
        )

    def build_from_track(
        self,
        sync_meas: "SynchronizedMeasurement",
        trajectory_est: "TrajectoryEstimate",
        object_id: int = 1,
        confidence: Optional[float] = None,
    ) -> ObjectPoseEstimate:
        """
        Build a full estimate from a synchronized measurement and
        a trajectory estimate.

        The object pose in base frame is computed from the trajectory
        evaluated at the frame timestamp.

        Parameters
        ----------
        sync_meas : SynchronizedMeasurement
        trajectory_est : TrajectoryEstimate
            Trajectory estimate from the reconstruction stage.
        object_id : int
        confidence : float or None
            If None, auto-computed from reprojection quality and
            measurement freshness.

        Returns
        -------
        ObjectPoseEstimate
        """
        # Robot pose from the synchronized measurement
        robot_pose = RobotPose6D(
            x=sync_meas.X_mm,
            y=sync_meas.Y_mm,
            z=sync_meas.Z_mm,
            rx=sync_meas.C_deg,
            ry=sync_meas.B_deg,
            rz=sync_meas.A_deg,
        )

        # Object pose from trajectory estimate
        object_pose_base = None
        if trajectory_est.solvable:
            import numpy as np
            obj_pos_world = trajectory_est.evaluate(sync_meas.timestamp)

            # Convert world position (meters) to robot base frame (mm)
            KUKA_POSITION_SCALE = 0.001
            obj_x_mm = float(obj_pos_world[0] / KUKA_POSITION_SCALE)
            obj_y_mm = float(obj_pos_world[1] / KUKA_POSITION_SCALE)
            obj_z_mm = float(obj_pos_world[2] / KUKA_POSITION_SCALE)

            # Orientation: use the same as the robot (for position-only tracking)
            # Full 6-DOF tracking would estimate orientation separately
            object_pose_base = RobotPose6D(
                x=obj_x_mm,
                y=obj_y_mm,
                z=obj_z_mm,
                rx=sync_meas.C_deg,
                ry=sync_meas.B_deg,
                rz=sync_meas.A_deg,
            )

        # Auto-compute confidence if not provided
        if confidence is None:
            confidence = self._compute_confidence(
                trajectory_est, sync_meas.sync_error_s
            )

        # Determine tracking state
        if trajectory_est.solvable and confidence > 0.3:
            state = TrackingState.TRACKED
        elif trajectory_est.solvable:
            state = TrackingState.INVALID
        else:
            state = TrackingState.INITIALIZING

        return ObjectPoseEstimate(
            timestamp=sync_meas.timestamp,
            frame_id=sync_meas.frame_id,
            object_id=object_id,
            tracking_state=state,
            u=sync_meas.u,
            v=sync_meas.v,
            robot_pose=robot_pose,
            object_pose_base=object_pose_base,
            confidence=confidence,
            sync_error_s=sync_meas.sync_error_s,
            reprojection_error_px=trajectory_est.reprojection_rms
            if trajectory_est.solvable else -1.0,
            polynomial_order=trajectory_est.order
            if trajectory_est.solvable else 0,
            window_size=trajectory_est.window_size
            if trajectory_est.solvable else 0,
        )

    @staticmethod
    def _compute_confidence(
        trajectory_est: "TrajectoryEstimate",
        sync_error_s: float,
    ) -> float:
        """
        Heuristic confidence from reprojection quality and sync error.

        Parameters
        ----------
        trajectory_est : TrajectoryEstimate
        sync_error_s : float

        Returns
        -------
        float
            Confidence in [0, 1].
        """
        if not trajectory_est.solvable:
            return 0.0

        # Reprojection quality: RMS error below 1 px → confidence near 1.0
        reproj_rms = trajectory_est.reprojection_rms
        if reproj_rms <= 0.0 or not math.isfinite(reproj_rms):
            reproj_score = 1.0
        else:
            # Exponential decay: RMS > 5 px → score ≈ 0
            reproj_score = math.exp(-reproj_rms / 2.0)

        # Sync quality: error below 5 ms → score near 1.0
        sync_score = math.exp(-sync_error_s / 0.02)

        # Combine: geometric mean
        confidence = math.sqrt(max(0.0, reproj_score) * max(0.0, sync_score))

        return max(0.0, min(1.0, float(confidence)))
