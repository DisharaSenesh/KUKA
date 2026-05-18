"""
Geometric constraints for monocular trajectory reconstruction.

The fundamental constraint: for each measurement i, the true 3D position
X(t_i) must lie on the world-frame ray (origin C_i, direction d_i).

This is expressed as the cross-product constraint:

    d_i × (X(t_i) - C_i) = 0

Substituting the polynomial trajectory model X(t) = Σ a_k * (t - t0)^k:

    d_i × (Σ_k a_k * (t_i - t0)^k - C_i) = 0

    Σ_k (t_i - t0)^k * (d_i × a_k) = d_i × C_i

Using the skew-symmetric matrix [d_i]× (where [d_i]× v = d_i × v):

    Σ_k (t_i - t0)^k * [d_i]× * a_k = [d_i]× * C_i

This is a linear system in the unknown coefficient vectors a_k.
"""

import numpy as np

from core.types.measurement import Measurement
from core.types.trajectory_state import TrajectoryState


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Build the 3x3 skew-symmetric (cross-product) matrix for vector v.

    [v]× = [[ 0,  -v_z,  v_y],
            [v_z,   0, -v_x],
            [-v_y, v_x,   0]]

    Satisfies: [v]× * w = v × w for any vector w.

    Parameters
    ----------
    v : np.ndarray, shape (3,)
        Input vector.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Skew-symmetric matrix.
    """
    v = np.asarray(v, dtype=np.float64)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def build_linear_system(
    measurements: list,
    order: int,
    t0: float
) -> tuple:
    """
    Build the linear system A * x = b for monocular trajectory estimation.

    The unknown x is the flattened coefficient vector:
        x = [a0_x, a0_y, a0_z, a1_x, a1_y, a1_z, ...]

    Each measurement contributes 3 rows (the three scalar equations from
    the cross-product constraint), though effectively only 2 are independent.
    The least-squares solver handles this redundancy naturally.

    Parameters
    ----------
    measurements : list of Measurement
        List of measurements, each containing a world-frame ray and timestamp.
    order : int
        Polynomial order for the trajectory model.
    t0 : float
        Reference time for the polynomial.

    Returns
    -------
    A : np.ndarray, shape (3*n_measurements, 3*(order+1))
        Design matrix.
    b : np.ndarray, shape (3*n_measurements,)
        Right-hand side vector.

    Notes
    -----
    For measurement i with ray (C_i, d_i) at time t_i:
        Row block 3i : 3i+2 of A contains [d_i]× scaled by (t_i - t0)^k
        for each coefficient column block.
        b[3i:3i+3] = [d_i]× * C_i
    """
    n_meas = len(measurements)
    n_coeffs = order + 1  # number of coefficient vectors
    n_unknowns = 3 * n_coeffs

    # Allocate system matrices
    A = np.zeros((3 * n_meas, n_unknowns), dtype=np.float64)
    b = np.zeros(3 * n_meas, dtype=np.float64)

    for i, meas in enumerate(measurements):
        # Extract ray parameters
        d = meas.ray.direction  # world-frame ray direction (unit)
        C = meas.ray.origin      # world-frame ray origin (camera center)
        dt = meas.t - t0

        # Skew-symmetric matrix for cross product with direction
        d_cross = skew_symmetric(d)

        # Right-hand side: [d]× * C
        row_start = 3 * i
        b[row_start:row_start + 3] = d_cross @ C

        # Fill the design matrix row block
        for k in range(n_coeffs):
            col_start = 3 * k
            # Block = (dt)^k * [d]×
            A[row_start:row_start + 3, col_start:col_start + 3] = (dt ** k) * d_cross

    return A, b


def compute_residuals(
    measurements: list,
    state: TrajectoryState
) -> np.ndarray:
    """
    Compute the geometric residual vector for all measurements.

    For each measurement i:
        r_i = d_i × (X(t_i) - C_i)

    where X(t_i) = state.evaluate(t_i).

    Parameters
    ----------
    measurements : list of Measurement
        Measurements with world-frame rays and timestamps.
    state : TrajectoryState
        Current trajectory estimate.

    Returns
    -------
    np.ndarray, shape (3*n_measurements,)
        Residual vector (stacked 3-vectors for each measurement).
    """
    n_meas = len(measurements)
    residuals = np.zeros(3 * n_meas, dtype=np.float64)

    for i, meas in enumerate(measurements):
        # Evaluate trajectory at measurement time
        X_est = state.evaluate(meas.t)

        # Compute cross-product constraint violation
        d = meas.ray.direction
        C = meas.ray.origin

        # r_i = d_i × (X_est - C_i)
        row_start = 3 * i
        residuals[row_start:row_start + 3] = np.cross(d, X_est - C)

    return residuals


def compute_scalar_residuals(
    measurements: list,
    state: TrajectoryState
) -> np.ndarray:
    """
    Compute scalar (Euclidean norm) residuals per measurement.

    Useful for outlier detection and visualization.

    Parameters
    ----------
    measurements : list of Measurement
        Measurements.
    state : TrajectoryState
        Current trajectory estimate.

    Returns
    -------
    np.ndarray, shape (n_measurements,)
        Per-measurement residual norms ‖d_i × (X(t_i) - C_i)‖.
    """
    res_3d = compute_residuals(measurements, state)
    n_meas = len(measurements)

    scalar_res = np.zeros(n_meas, dtype=np.float64)
    for i in range(n_meas):
        scalar_res[i] = np.linalg.norm(res_3d[3 * i:3 * i + 3])

    return scalar_res
