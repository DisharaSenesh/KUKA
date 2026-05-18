"""
SynchronizedMeasurement: the canonical system data type.

This is the MOST important data structure in the system. It represents
a single monocular observation that has been temporally aligned with
the robot pose stream.

After synchronization, ALL downstream modules consume ONLY this type.
This enables online/offline unification: whether the data comes from
live sensors or CSV replay, the downstream pipeline sees identical objects.

Fields follow the existing KUKA convention for readability:
  - (X, Y, Z): robot/camera translation in mm (or native robot units)
  - (A, B, C): KUKA Euler angles in degrees (Z-Y-X intrinsic)
  - (u, v): image pixel coordinates of the detection

Raw Euler angles are kept intentionally (not converted to rotation matrices
or quaternions) because:
  1. The robot provides them directly
  2. They are human-readable for debugging
  3. They match the CSV format exactly
  4. Conversion to rotation matrices happens downstream in the geometry layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SynchronizedMeasurement:
    """
    A temporally aligned monocular observation.

    This is the canonical data type. Every downstream consumer
    (tracking, optimization, prediction) receives this structure.

    Attributes
    ----------
    frame_id : int
        Monotonically increasing frame counter from the camera process.
    timestamp : float
        Frame acquisition time (time.perf_counter, seconds).
    u : float
        Horizontal pixel coordinate of the detected target.
    v : float
        Vertical pixel coordinate of the detected target.
    X_mm : float
        Robot/camera X translation at this timestamp (robot base frame, mm).
    Y_mm : float
        Robot/camera Y translation (robot base frame, mm).
    Z_mm : float
        Robot/camera Z translation (robot base frame, mm).
    A_deg : float
        KUKA Euler angle A (rotation about Z, degrees).
    B_deg : float
        KUKA Euler angle B (rotation about Y, degrees).
    C_deg : float
        KUKA Euler angle C (rotation about X, degrees).
    sync_error_s : float
        Absolute time difference between frame timestamp and the nearest
        robot pose timestamp (seconds). 0.0 means exact match.
    sync_method : str
        Method used for temporal alignment: "nearest", "linear_translation",
        or "none" if no pose was available.
    is_valid : bool
        False if sync_error exceeds the configured tolerance or if no
        pose was available at all.
    fx : float
        Camera focal length in pixels (x). Carried through from calibration.
    fy : float
        Camera focal length in pixels (y).
    cx : float
        Camera principal point x (pixels).
    cy : float
        Camera principal point y (pixels).
    """

    frame_id: int
    timestamp: float
    u: float
    v: float
    X_mm: float
    Y_mm: float
    Z_mm: float
    A_deg: float
    B_deg: float
    C_deg: float
    sync_error_s: float = 0.0
    sync_method: str = "nearest"
    is_valid: bool = True
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0

    @property
    def position_mm(self):
        """Return (X, Y, Z) as a tuple in mm."""
        return (self.X_mm, self.Y_mm, self.Z_mm)

    @property
    def orientation_deg(self):
        """Return (A, B, C) Euler angles as a tuple in degrees."""
        return (self.A_deg, self.B_deg, self.C_deg)

    @property
    def pixel(self):
        """Return (u, v) pixel coordinates."""
        return (self.u, self.v)

    def is_position_valid(self) -> bool:
        """Check if position values are finite (not NaN/inf from failed sync)."""
        import math
        return all(math.isfinite(v) for v in [self.X_mm, self.Y_mm, self.Z_mm])

    def to_csv_row(self) -> tuple:
        """
        Convert to a tuple matching the standard CSV column order.

        Returns
        -------
        tuple
            (frame_id, timestamp, u, v, X, Y, Z, A, B, C, sync_error_s)
        """
        return (
            self.frame_id,
            self.timestamp,
            self.u,
            self.v,
            self.X_mm,
            self.Y_mm,
            self.Z_mm,
            self.A_deg,
            self.B_deg,
            self.C_deg,
            self.sync_error_s,
        )

    @classmethod
    def from_csv_row(cls, row: tuple) -> "SynchronizedMeasurement":
        """
        Reconstruct from a CSV row tuple.

        Parameters
        ----------
        row : tuple
            (frame_id, timestamp, u, v, X, Y, Z, A, B, C, sync_error_s)

        Returns
        -------
        SynchronizedMeasurement
        """
        return cls(
            frame_id=int(row[0]),
            timestamp=float(row[1]),
            u=float(row[2]),
            v=float(row[3]),
            X_mm=float(row[4]),
            Y_mm=float(row[5]),
            Z_mm=float(row[6]),
            A_deg=float(row[7]),
            B_deg=float(row[8]),
            C_deg=float(row[9]),
            sync_error_s=float(row[10]),
            sync_method="csv_replay",
            is_valid=True,
        )

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"[SyncMeas frame={self.frame_id:04d} t={self.timestamp:.4f} "
            f"uv=({self.u:.1f},{self.v:.1f}) "
            f"XYZ=({self.X_mm:.1f},{self.Y_mm:.1f},{self.Z_mm:.1f}) "
            f"err={self.sync_error_s*1000:.2f}ms]"
        )


# CSV column names in standard order
CSV_COLUMNS = [
    "frame_id",
    "timestamp",
    "u",
    "v",
    "X_mm",
    "Y_mm",
    "Z_mm",
    "A_deg",
    "B_deg",
    "C_deg",
    "sync_error_s",
]
"""Standard CSV column headers for synchronized measurement files."""
