"""
Polynomial basis functions for trajectory modeling.

The polynomial basis of order `order` evaluated at scalar time t (relative to t0):

    b(t) = [1, dt, dt², ..., dt^order]

where dt = t - t0.

For a 3D trajectory with coefficient vectors [a0, a1, ..., a_order],
each a_k ∈ R³, the position is:

    X(t) = Σ a_k * dt^k = A @ b(t)

where A is a 3 x (order+1) matrix stacking the a_k as columns.

For the linear system, we use the Kronecker structure:
For each 3D constraint, the design matrix block is b(t) ⊗ I_3.
"""

import numpy as np


def evaluate_polynomial_basis(dt: float, order: int) -> np.ndarray:
    """
    Evaluate the scalar polynomial basis at relative time dt.

    Returns the row vector: [1, dt, dt², ..., dt^order]

    Parameters
    ----------
    dt : float
        Time difference (t - t0).
    order : int
        Polynomial order.

    Returns
    -------
    np.ndarray, shape (order+1,)
        Basis vector evaluated at dt.
    """
    basis = np.zeros(order + 1, dtype=np.float64)
    for k in range(order + 1):
        basis[k] = dt ** k
    return basis


def evaluate_polynomial_basis_matrix(
    times: np.ndarray,
    t0: float,
    order: int
) -> np.ndarray:
    """
    Build the polynomial design matrix for a set of timestamps.

    Returns matrix M of shape (n_measurements, order+1) where:
        M[i, k] = (t_i - t0)^k

    Used to construct the full Kronecker system in the constraints module.

    Parameters
    ----------
    times : np.ndarray, shape (n,)
        Array of measurement timestamps.
    t0 : float
        Reference time for the polynomial model.
    order : int
        Polynomial order.

    Returns
    -------
    np.ndarray, shape (n, order+1)
        Design matrix where each row is the basis evaluated at t_i.
    """
    n = len(times)
    dt = times - t0

    basis_matrix = np.zeros((n, order + 1), dtype=np.float64)
    for k in range(order + 1):
        basis_matrix[:, k] = dt ** k

    return basis_matrix


def evaluate_trajectory(
    coefficients: list,
    t: float,
    t0: float
) -> np.ndarray:
    """
    Evaluate a 3D polynomial trajectory at time t.

    X(t) = Σ a_k * (t - t0)^k

    Parameters
    ----------
    coefficients : list of np.ndarray
        List of 3D coefficient vectors [a0, a1, ..., a_order].
    t : float
        Evaluation time (seconds).
    t0 : float
        Reference time of the model.

    Returns
    -------
    np.ndarray, shape (3,)
        3D position at time t.
    """
    dt = t - t0
    position = np.zeros(3, dtype=np.float64)
    for k, a_k in enumerate(coefficients):
        position += a_k * (dt ** k)
    return position
