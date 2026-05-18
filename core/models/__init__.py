"""
Polynomial trajectory models.

The object trajectory is modeled as a vector-valued polynomial:

    X(t) = Σ a_k * (t - t0)^k   for k = 0 .. order

where each a_k ∈ R³ is a coefficient vector.

This module provides:
  - polynomial basis evaluation (for constructing the design matrix)
  - trajectory evaluation from coefficients
"""

from .polynomial import (
    evaluate_polynomial_basis,
    evaluate_polynomial_basis_matrix,
    evaluate_trajectory,
)

__all__ = [
    "evaluate_polynomial_basis",
    "evaluate_polynomial_basis_matrix",
    "evaluate_trajectory",
]
