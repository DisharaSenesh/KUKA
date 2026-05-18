"""
Reprojection validation: evaluate trajectory estimates by projecting back
into the image plane.

For each filtered measurement with observed pixel (u_i, v_i) and estimated
trajectory point X_hat(t_i):

  1. Transform X_hat to camera frame:  X_cam = R @ (X_hat - C)
  2. Project to image:  u_hat = fx * X_cam.x / X_cam.z + cx
                        v_hat = fy * X_cam.y / X_cam.z + cy
  3. Compute reprojection error:
        e_i² = (u_i - u_hat)² + (v_i - v_hat)²

Total reprojection cost:  E = Σ e_i²

Reprojection error is in pixel units and is the primary quality metric
for trajectory estimation.
"""

from typing import List

import numpy as np

from .sliding_window import FilteredMeasurement
from .ray_builder import synchronized_to_measurement


def reprojection_error_single(
    filtered_meas: FilteredMeasurement,
    coefficients: list,
    t0: float,
) -> float:
    """
    Compute the reprojection error for a single measurement.

    e = sqrt( (u_obs - u_proj)² + (v_obs - v_proj)² )

    Parameters
    ----------
    filtered_meas : FilteredMeasurement
    coefficients : list of np.ndarray
        Trajectory coefficient vectors [a0, a1, ...].
    t0 : float
        Reference time.

    Returns
    -------
    float
        Reprojection error in pixels. float('inf') on failure.
    """
    sm = filtered_meas.synchronized

    try:
        # Convert to measurement to extract ray and intrinsics
        meas = synchronized_to_measurement(sm)
    except Exception:
        return float("inf")

    # Camera center and rotation
    C = meas.ray.origin
    # Reconstruct rotation from Euler angles for the projection
    from core.robotics.kinematics.transforms import euler_abc_to_rotation
    a_rad = np.radians(sm.A_deg)
    b_rad = np.radians(sm.B_deg)
    c_rad = np.radians(sm.C_deg)
    R = euler_abc_to_rotation(a_rad, b_rad, c_rad)

    # Evaluate trajectory at measurement time
    dt = sm.timestamp - t0
    X_hat = np.zeros(3, dtype=np.float64)
    for k, a_k in enumerate(coefficients):
        X_hat += a_k * (dt ** k)

    # R (from euler_abc_to_rotation) maps world→camera under Pose convention.
    # X_cam = R @ (X_world - C) is the world-to-camera point transform.
    X_cam = R @ (X_hat - C)

    # Check the point is in front of the camera
    if X_cam[2] <= 1e-6:
        return float("inf")

    # Project to image
    u_proj = sm.fx * (X_cam[0] / X_cam[2]) + sm.cx
    v_proj = sm.fy * (X_cam[1] / X_cam[2]) + sm.cy

    # Reprojection error
    du = sm.u - u_proj
    dv = sm.v - v_proj

    return float(np.sqrt(du * du + dv * dv))


def compute_reprojection_errors(
    filtered_measurements: List[FilteredMeasurement],
    coefficients: list,
    t0: float,
) -> np.ndarray:
    """
    Compute reprojection errors for all measurements.

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
    coefficients : list of np.ndarray
    t0 : float

    Returns
    -------
    np.ndarray, shape (n,)
        Reprojection errors in pixels.
    """
    errors = np.zeros(len(filtered_measurements), dtype=np.float64)
    for i, fm in enumerate(filtered_measurements):
        errors[i] = reprojection_error_single(fm, coefficients, t0)
    return errors


def total_reprojection_cost(
    filtered_measurements: List[FilteredMeasurement],
    coefficients: list,
    t0: float,
) -> float:
    """
    Sum of squared reprojection errors.

    E = Σ (u_i - u_proj)² + Σ (v_i - v_proj)²

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
    coefficients : list of np.ndarray
    t0 : float

    Returns
    -------
    float
        Total squared reprojection error (pixels²).
    """
    total = 0.0
    for fm in filtered_measurements:
        err = reprojection_error_single(fm, coefficients, t0)
        if np.isfinite(err):
            total += err * err
    return total


def rms_reprojection_error(
    filtered_measurements: List[FilteredMeasurement],
    coefficients: list,
    t0: float,
) -> float:
    """
    RMS reprojection error over all valid measurements.

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
    coefficients : list of np.ndarray
    t0 : float

    Returns
    -------
    float
        RMS reprojection error in pixels.
    """
    errors = compute_reprojection_errors(filtered_measurements, coefficients, t0)
    finite = errors[np.isfinite(errors)]
    if len(finite) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(finite ** 2)))
