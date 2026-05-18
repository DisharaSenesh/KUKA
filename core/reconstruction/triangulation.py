"""
Triangulation: continuous-time trajectory estimation from monocular rays.

This is the core reconstruction engine. It:
  1. Takes a set of FilteredMeasurements from the sliding window
  2. Converts them to world-space rays via the ray builder
  3. Builds the geometric constraint linear system
  4. Solves for polynomial trajectory coefficients via least squares
  5. Returns the estimated trajectory

The core geometric constraint for each measurement i at time t_i:
    d_i × ( X(t_i) - C_i ) = 0

where X(t) = a₀ + a₁·(t-t₀) + a₂·(t-t₀)² + ... is the polynomial trajectory.

Delegates to:
  - core.constraints.geometric.build_linear_system (matrix construction)
  - core.optimization.least_squares.solve_least_squares (SVD solver)
  - core.optimization.least_squares.unflatten_coefficients (vector → list)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from core.types.measurement import Measurement
from core.constraints.geometric import build_linear_system
from core.optimization.least_squares import solve_least_squares, unflatten_coefficients

from .sliding_window import FilteredMeasurement
from .ray_builder import synchronized_to_measurement


def triangulate(
    filtered_measurements: List[FilteredMeasurement],
    order: int,
    t0: Optional[float] = None,
) -> Tuple[Optional[list], Optional[np.ndarray]]:
    """
    Estimate a polynomial trajectory from a set of filtered measurements.

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
        Valid filtered measurements from the sliding window.
    order : int
        Polynomial order (1 = linear, 2 = quadratic, 3 = cubic).
    t0 : float or None
        Reference time for the polynomial model.
        If None, uses the latest measurement timestamp.

    Returns
    -------
    coefficients : list of np.ndarray or None
        List of 3D coefficient vectors [a0, a1, ..., a_order], or None
        if triangulation failed (too few measurements, numerical failure).
    x_opt : np.ndarray or None
        Flattened coefficient vector from the solver (for diagnostics).
    """
    # Convert to core Measurements
    measurements: List[Measurement] = []
    for fm in filtered_measurements:
        if not fm.is_valid:
            continue
        try:
            meas = synchronized_to_measurement(fm.synchronized)
            measurements.append(meas)
        except Exception:
            continue

    # Check minimum requirements
    min_required = order + 1
    if len(measurements) < min_required:
        return (None, None)

    # Set reference time to latest measurement
    if t0 is None:
        t0 = measurements[-1].t

    # Build the geometric constraint linear system
    A, b = build_linear_system(
        measurements=measurements,
        order=order,
        t0=t0,
    )

    # Solve via SVD-based least squares
    try:
        x_opt = solve_least_squares(A, b)
    except np.linalg.LinAlgError:
        return (None, None)

    # Unflatten into list of 3D coefficient vectors
    coefficients = unflatten_coefficients(x_opt, order)

    return (coefficients, x_opt)
