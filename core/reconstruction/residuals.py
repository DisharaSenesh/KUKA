"""
Geometric residuals: evaluate constraint violation for a trajectory estimate.

For each measurement i with world-frame ray (C_i, d_i) and estimated
trajectory X(t), the geometric residual is:

    r_i = ‖ d_i × ( X(t_i) - C_i ) ‖

This measures how far the estimated trajectory point is from the
viewing ray. Zero residual means the point lies exactly on the ray.
"""

from typing import List

import numpy as np

from core.types.trajectory_state import TrajectoryState
from .sliding_window import FilteredMeasurement
from .ray_builder import synchronized_to_world_ray


def compute_geometric_residual(
    filtered_meas: FilteredMeasurement,
    coefficients: list,
    t0: float,
) -> float:
    """
    Compute the scalar geometric residual for a single measurement.

    r = ‖ d × ( X(t) - C ) ‖

    Parameters
    ----------
    filtered_meas : FilteredMeasurement
        The measurement to evaluate.
    coefficients : list of np.ndarray
        Trajectory coefficient vectors [a0, a1, ...].
    t0 : float
        Reference time of the polynomial model.

    Returns
    -------
    float
        Scalar residual (meters). float('inf') if computation fails.
    """
    try:
        ray = synchronized_to_world_ray(filtered_meas.synchronized)
    except Exception:
        return float("inf")

    d = ray.direction
    C = ray.origin
    t_meas = filtered_meas.synchronized.timestamp

    # Evaluate trajectory at measurement time
    dt = t_meas - t0
    X_est = np.zeros(3, dtype=np.float64)
    for k, a_k in enumerate(coefficients):
        X_est += a_k * (dt ** k)

    # Cross-product constraint magnitude
    residual = np.linalg.norm(np.cross(d, X_est - C))
    return float(residual)


def compute_all_residuals(
    filtered_measurements: List[FilteredMeasurement],
    coefficients: list,
    t0: float,
) -> np.ndarray:
    """
    Compute geometric residuals for all measurements.

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
    coefficients : list of np.ndarray
    t0 : float

    Returns
    -------
    np.ndarray, shape (n,)
        Scalar residuals for each measurement (meters).
    """
    residuals = np.zeros(len(filtered_measurements), dtype=np.float64)
    for i, fm in enumerate(filtered_measurements):
        residuals[i] = compute_geometric_residual(fm, coefficients, t0)
    return residuals


def rms_residual(
    filtered_measurements: List[FilteredMeasurement],
    coefficients: list,
    t0: float,
) -> float:
    """
    Root-mean-square geometric residual over all measurements.

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
    coefficients : list of np.ndarray
    t0 : float

    Returns
    -------
    float
        RMS residual in meters.
    """
    residuals = compute_all_residuals(filtered_measurements, coefficients, t0)
    finite_residuals = residuals[np.isfinite(residuals)]
    if len(finite_residuals) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(finite_residuals ** 2)))


def max_residual(
    filtered_measurements: List[FilteredMeasurement],
    coefficients: list,
    t0: float,
) -> float:
    """
    Maximum geometric residual among all measurements.

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
    coefficients : list of np.ndarray
    t0 : float

    Returns
    -------
    float
        Maximum residual in meters.
    """
    residuals = compute_all_residuals(filtered_measurements, coefficients, t0)
    finite_residuals = residuals[np.isfinite(residuals)]
    if len(finite_residuals) == 0:
        return float("inf")
    return float(np.max(finite_residuals))
