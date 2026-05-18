"""
Least squares solvers for trajectory coefficient estimation.

Provides both standard and weighted least squares.

Standard:
    minimize ‖A x - b‖²
    solution: x = (Aᵀ A)⁻¹ Aᵀ b

Weighted:
    minimize ‖W (A x - b)‖²
    solution: x = (Aᵀ W² A)⁻¹ Aᵀ W² b

where W = diag(w_1, w_1, w_1, w_2, w_2, w_2, ...)
with each measurement getting a scalar weight w_i replicated across
its 3 constraint rows.
"""

from typing import Optional

import numpy as np


def solve_least_squares(
    A: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    Solve the standard least squares problem: minimize ‖A x - b‖².

    Uses np.linalg.lstsq for numerical stability on potentially
    ill-conditioned polynomial systems.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Design matrix from the constraints module.
    b : np.ndarray, shape (m,)
        Right-hand side vector.

    Returns
    -------
    np.ndarray, shape (n,)
        Optimal coefficient vector x.

    Notes
    -----
    np.linalg.lstsq uses an SVD-based approach that handles
    rank-deficient systems gracefully.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    return x


def solve_weighted_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Solve weighted least squares: minimize ‖W (A x - b)‖².

    Each measurement gets a scalar weight w_i, which is applied to all
    3 rows of its constraint block.

    Parameters
    ----------
    A : np.ndarray, shape (3*n_meas, n)
        Design matrix.
    b : np.ndarray, shape (3*n_meas,)
        Right-hand side vector.
    weights : np.ndarray, shape (n_meas,)
        Per-measurement weights w_i (positive).

    Returns
    -------
    np.ndarray, shape (n,)
        Optimal coefficient vector x.

    Notes
    -----
    The weight matrix W is constructed such that each measurement's
    3-row block is multiplied by the same weight.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_meas = len(weights)
    assert A.shape[0] == 3 * n_meas, \
        f"A has {A.shape[0]} rows, expected {3 * n_meas} (3 per measurement)"

    # Build the diagonal weight matrix
    # Each weight w_i is repeated 3 times for the 3 constraint rows
    W_flat = np.repeat(weights, 3)

    # Weight the system: WA x = Wb
    A_weighted = W_flat[:, np.newaxis] * A
    b_weighted = W_flat * b

    # Solve using SVD-based least squares
    x, residuals, rank, singular_values = np.linalg.lstsq(
        A_weighted, b_weighted, rcond=None
    )

    return x


def unflatten_coefficients(
    x: np.ndarray,
    order: int
) -> list:
    """
    Convert a flattened coefficient vector back into a list of 3D vectors.

    x = [a0_x, a0_y, a0_z, a1_x, ..., a_order_z]
    →
    [np.array([a0_x, a0_y, a0_z]), np.array([a1_x, a1_y, a1_z]), ...]

    Parameters
    ----------
    x : np.ndarray, shape (3*(order+1),)
        Flattened coefficient vector.
    order : int
        Polynomial order.

    Returns
    -------
    list of np.ndarray
        List of 3D coefficient vectors, length = order + 1.
    """
    x = np.asarray(x, dtype=np.float64)
    n_coeffs = order + 1

    coefficients = []
    for k in range(n_coeffs):
        coeff_k = x[3 * k: 3 * k + 3]
        coefficients.append(coeff_k.copy())

    return coefficients
