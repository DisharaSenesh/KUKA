"""
Camera baseline validation filters.

Evaluates the spatial separation (baseline) between camera centers.
A small camera baseline produces weak parallax, degrading monocular
reconstruction quality.

Mathematical foundation:

Given two camera centers C1, C2 in world frame:

    b = ‖C2 - C1‖

The baseline distance directly affects depth resolution:
    σ_depth ∝ depth² / (f * b)

where f is the focal length. A larger baseline b reduces depth
uncertainty (improves triangulation quality).

All functions operate on world-frame camera centers from Pose objects.
"""

from __future__ import annotations

import numpy as np

from core.types.pose import Pose


def baseline_distance(pose1: Pose, pose2: Pose) -> float:
    """
    Compute the Euclidean distance between two camera centers.

    b = ‖C2 - C1‖

    Parameters
    ----------
    pose1 : Pose
        First camera pose (world frame).
    pose2 : Pose
        Second camera pose (world frame).

    Returns
    -------
    float
        Camera baseline distance in world units (e.g., meters).
    """
    C1 = np.asarray(pose1.C, dtype=np.float64)
    C2 = np.asarray(pose2.C, dtype=np.float64)
    return np.linalg.norm(C2 - C1)


def has_valid_baseline(
    pose1: Pose,
    pose2: Pose,
    min_baseline: float
) -> bool:
    """
    Check whether two camera poses provide sufficient baseline.

    Parameters
    ----------
    pose1 : Pose
        First camera pose.
    pose2 : Pose
        Second camera pose.
    min_baseline : float
        Minimum acceptable baseline distance (world units).

    Returns
    -------
    bool
        True if the baseline is at least min_baseline.
    """
    b = baseline_distance(pose1, pose2)
    return b >= min_baseline


def minimum_pairwise_baseline(poses: list) -> float:
    """
    Find the smallest baseline among all pairs of camera poses.

    Useful for identifying the weakest stereo pair in a multi-view setup.

    Parameters
    ----------
    poses : list of Pose
        List of camera poses (world frame).

    Returns
    -------
    float
        Minimum pairwise baseline distance.
    """
    n = len(poses)
    if n < 2:
        return 0.0

    min_b = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            b = baseline_distance(poses[i], poses[j])
            if b < min_b:
                min_b = b
    
    return min_b


def baseline_quality_score(
    poses: list,
    reference_depth: float = 10.0
) -> float:
    """
    Compute a heuristic baseline quality score for multi-view reconstruction.

    The score is based on the ratio of minimum baseline to a reference
    depth. Larger baselines relative to depth produce better quality.

    quality ∈ [0, 1] where:
      0  → baseline too small for given depth (degenerate)
      1  → baseline is well-matched to depth

    Parameters
    ----------
    poses : list of Pose
        Camera poses.
    reference_depth : float
        Approximate depth to the target (world units).

    Returns
    -------
    float
        Quality score in [0, 1].

    Notes
    -----
    The score function is a smooth saturation:
        score = min_baseline / (reference_depth + min_baseline)

    This gives score ≈ 0.5 when baseline ≈ depth, and asymptotically
    approaches 1.0 for large baselines.
    """
    if len(poses) < 2:
        return 0.0

    min_b = minimum_pairwise_baseline(poses)
    if min_b <= 0.0:
        return 0.0

    # Smooth saturation toward 1.0
    score = min_b / (reference_depth + min_b)
    return float(score)


# ---------------------------------------------------------------------------
# SynchronizedMeasurement → camera-centre extraction
# ---------------------------------------------------------------------------
# These bridge functions extract camera centres from SynchronizedMeasurement
# objects (X_mm, Y_mm, Z_mm → metres) so the existing baseline functions can
# evaluate parallax geometry without needing the full Pose pipeline.
#
# X_mm, Y_mm, Z_mm are the robot end-effector position in mm.
# Multiplying by 0.001 converts to metres.
# ---------------------------------------------------------------------------


def measurement_camera_centre(
    sm: "SynchronizedMeasurement",
) -> np.ndarray:
    """Extract the camera centre in world frame from a synchronised measurement.

    The camera centre is approximated as the robot end-effector position
    (no TCP-to-camera offset applied — use the full ``compute_camera_pose``
    pipeline if a calibrated offset is needed).

    Parameters
    ----------
    sm : SynchronizedMeasurement
        Synchronised measurement with X_mm, Y_mm, Z_mm.

    Returns
    -------
    np.ndarray, shape (3,)
        Camera centre in metres (world frame).
    """
    return np.array(
        [sm.X_mm * 0.001, sm.Y_mm * 0.001, sm.Z_mm * 0.001],
        dtype=np.float64,
    )


def baseline_between_measurements(
    sm1: "SynchronizedMeasurement",
    sm2: "SynchronizedMeasurement",
) -> float:
    """Camera baseline distance (metres) between two synchronised measurements.

    Parameters
    ----------
    sm1, sm2 : SynchronizedMeasurement

    Returns
    -------
    float
        Euclidean distance between camera centres in metres.
    """
    C1 = measurement_camera_centre(sm1)
    C2 = measurement_camera_centre(sm2)
    return float(np.linalg.norm(C2 - C1))


def has_valid_baseline_between(
    sm1: "SynchronizedMeasurement",
    sm2: "SynchronizedMeasurement",
    min_baseline: float,
) -> bool:
    """Check whether two synchronised measurements have sufficient camera baseline.

    Parameters
    ----------
    sm1, sm2 : SynchronizedMeasurement
    min_baseline : float
        Minimum acceptable baseline distance (metres).

    Returns
    -------
    bool
    """
    b = baseline_between_measurements(sm1, sm2)
    return b >= min_baseline


def minimum_pairwise_baseline_between(
    measurements: list,
) -> float:
    """Smallest camera baseline (metres) among a list of synchronised measurements.

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement

    Returns
    -------
    float
        Minimum pairwise baseline distance in metres.
    """
    n = len(measurements)
    if n < 2:
        return 0.0

    min_b = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            b = baseline_between_measurements(measurements[i], measurements[j])
            if b < min_b:
                min_b = b
    return float(min_b)


def baseline_quality_between(
    measurements: list,
    reference_depth: float = 10.0,
) -> float:
    """Heuristic baseline quality score from synchronised measurements.

    quality ∈ [0, 1]; see ``baseline_quality_score`` for the formula.

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement
    reference_depth : float
        Approximate depth to target (metres).

    Returns
    -------
    float
        Quality score in [0, 1].
    """
    if len(measurements) < 2:
        return 0.0
    min_b = minimum_pairwise_baseline_between(measurements)
    if min_b <= 0.0:
        return 0.0
    return float(min_b / (reference_depth + min_b))
