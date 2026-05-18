"""
Temporal matcher: aligns detection timestamps with the robot pose stream.

Provides two strategies:
  1. nearest-neighbor — find the closest pose in time
  2. linear_translation — interpolate between bracketing poses

The matcher is a pure function: it takes a PoseBuffer and a target
timestamp, and returns the estimated robot pose at that time.

All timestamps use time.perf_counter() (monotonic, high-resolution).
"""

from __future__ import annotations

from typing import Optional

from .pose_buffer import PoseBuffer, TimedPose


def match_nearest(
    pose_buffer: PoseBuffer,
    target_time: float,
) -> tuple:
    """
    Nearest-neighbor temporal matching.

    Finds the robot pose closest in time to target_time and returns
    it along with the synchronization error.

    Parameters
    ----------
    pose_buffer : PoseBuffer
        Rolling buffer of timed robot poses.
    target_time : float
        Detection timestamp to match against.

    Returns
    -------
    timed_pose : TimedPose or None
        The nearest pose, or None if buffer is empty.
    sync_error_s : float
        Absolute time difference (seconds). float('inf') if no pose.
    method : str
        Always "nearest".
    """
    nearest = pose_buffer.find_nearest(target_time)

    if nearest is None:
        return (None, float("inf"), "nearest")

    sync_error = abs(nearest.timestamp - target_time)
    return (nearest, sync_error, "nearest")


def match_linear_translation(
    pose_buffer: PoseBuffer,
    target_time: float,
) -> tuple:
    """
    Linear translation interpolation between bracketing poses.

    Finds the two poses that bracket target_time and linearly blends
    the translation (X, Y, Z). Euler angles are taken from the
    nearer bracketing pose (no SLERP yet).

    The architecture is ready for future rotation interpolation.

    Parameters
    ----------
    pose_buffer : PoseBuffer
        Rolling buffer of timed robot poses.
    target_time : float
        Detection timestamp.

    Returns
    -------
    timed_pose : TimedPose or None
        Interpolated pose, or None if no bracketing pair exists.
    sync_error_s : float
        Maximum time difference to either bracketing pose.
    method : str
        "linear_translation" or "nearest" (fallback).
    """
    before, after = pose_buffer.find_bracketing(target_time)

    if before is None and after is None:
        return (None, float("inf"), "linear_translation")

    # Extrapolation fallback: use the available pose
    if before is None:
        sync_error = abs(after.timestamp - target_time)
        return (after, sync_error, "linear_translation")
    if after is None:
        sync_error = abs(before.timestamp - target_time)
        return (before, sync_error, "linear_translation")

    # Interpolation
    dt = after.timestamp - before.timestamp

    if dt < 1e-10:
        # Bracketing poses have the same timestamp
        sync_error = abs(before.timestamp - target_time)
        return (before, sync_error, "linear_translation")

    # Fraction between before and after
    alpha = (target_time - before.timestamp) / dt
    alpha = max(0.0, min(1.0, alpha))  # clamp for safety

    # Linearly blend translation
    X_interp = before.X_mm + alpha * (after.X_mm - before.X_mm)
    Y_interp = before.Y_mm + alpha * (after.Y_mm - before.Y_mm)
    Z_interp = before.Z_mm + alpha * (after.Z_mm - before.Z_mm)

    # Use the nearer pose's orientation
    if alpha < 0.5:
        A_interp, B_interp, C_interp = before.A_deg, before.B_deg, before.C_deg
    else:
        A_interp, B_interp, C_interp = after.A_deg, after.B_deg, after.C_deg

    interp_pose = TimedPose(
        X_mm=X_interp, Y_mm=Y_interp, Z_mm=Z_interp,
        A_deg=A_interp, B_deg=B_interp, C_deg=C_interp,
        timestamp=target_time,
    )

    sync_error = max(
        abs(target_time - before.timestamp),
        abs(target_time - after.timestamp),
    )

    return (interp_pose, sync_error, "linear_translation")
