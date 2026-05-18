"""
Polynomial trajectory models for continuous-time 3D reconstruction.

The object trajectory is modeled as three independent scalar polynomials:
  x(t) = a0_x + a1_x * dt + a2_x * dt² + ...
  y(t) = a0_y + a1_y * dt + a2_y * dt² + ...
  z(t) = a0_z + a1_z * dt + a2_z * dt² + ...

where dt = t - t0 (time relative to the model's reference time).

Each scalar polynomial shares the same basis functions but has its own
coefficients. This is equivalent to a vector-valued polynomial with
coefficient vectors a_k ∈ R³.

Model orders:
  order = 0: constant position   (1 coefficient per axis)
  order = 1: constant velocity   (2 coefficients per axis)
  order = 2: constant accel      (3 coefficients per axis)
  order = 3: constant jerk       (4 coefficients per axis)
"""

import numpy as np


def build_polynomial_basis(
    times: np.ndarray,
    t0: float,
    order: int,
) -> np.ndarray:
    """
    Build the Vandermonde matrix for a set of time values.

    Returns matrix M of shape (n_times, order + 1) where:
      M[i, k] = (t_i - t0)^k

    This is used to construct the design matrix for the full 3D system.

    Parameters
    ----------
    times : np.ndarray, shape (n,)
        Measurement timestamps.
    t0 : float
        Reference time for the polynomial model.
    order : int
        Polynomial order (0 = constant, 1 = linear, etc.).

    Returns
    -------
    np.ndarray, shape (n, order + 1)
        Vandermonde basis matrix.
    """
    dt = np.asarray(times, dtype=np.float64) - t0
    n = len(times)
    M = np.zeros((n, order + 1), dtype=np.float64)

    for k in range(order + 1):
        M[:, k] = dt ** k

    return M


def evaluate_trajectory_at_time(
    coefficients_x: np.ndarray,
    coefficients_y: np.ndarray,
    coefficients_z: np.ndarray,
    t: float,
    t0: float,
) -> np.ndarray:
    """
    Evaluate the 3D trajectory at a single time t.

    x(t) = Σ c_x[k] * (t-t0)^k, similarly for y, z.

    Parameters
    ----------
    coefficients_x, coefficients_y, coefficients_z : np.ndarray
        Coefficient arrays for each axis, length = order + 1.
    t : float
        Evaluation time.
    t0 : float
        Reference time of the model.

    Returns
    -------
    np.ndarray, shape (3,)
        3D position at time t.
    """
    dt = t - t0
    pos = np.zeros(3, dtype=np.float64)

    for k in range(len(coefficients_x)):
        basis = dt ** k
        pos[0] += coefficients_x[k] * basis
        pos[1] += coefficients_y[k] * basis
        pos[2] += coefficients_z[k] * basis

    return pos


def evaluate_trajectory_vectorized(
    coefficients_x: np.ndarray,
    coefficients_y: np.ndarray,
    coefficients_z: np.ndarray,
    times: np.ndarray,
    t0: float,
) -> np.ndarray:
    """
    Evaluate the trajectory at multiple times.

    Parameters
    ----------
    coefficients_x, coefficients_y, coefficients_z : np.ndarray
        Coefficient arrays for each axis.
    times : np.ndarray, shape (n,)
        Evaluation times.
    t0 : float
        Reference time.

    Returns
    -------
    np.ndarray, shape (n, 3)
        Array of 3D positions at each time.
    """
    dt = times - t0
    n = len(times)
    positions = np.zeros((n, 3), dtype=np.float64)

    for k in range(len(coefficients_x)):
        basis = dt ** k
        positions[:, 0] += coefficients_x[k] * basis
        positions[:, 1] += coefficients_y[k] * basis
        positions[:, 2] += coefficients_z[k] * basis

    return positions


def num_coefficients(order: int) -> int:
    """
    Number of scalar coefficients per axis for a given polynomial order.

    order=0 → 1, order=1 → 2, order=2 → 3, order=3 → 4, etc.

    Parameters
    ----------
    order : int
        Polynomial order.

    Returns
    -------
    int
        Number of coefficients per axis (= order + 1).
    """
    return order + 1


def num_unknowns(order: int) -> int:
    """
    Total number of unknowns in the 3D trajectory system.

    = 3 * (order + 1)  (coefficients for x, y, z axes)

    Parameters
    ----------
    order : int
        Polynomial order.

    Returns
    -------
    int
        Total unknown count for the linear system.
    """
    return 3 * (order + 1)
