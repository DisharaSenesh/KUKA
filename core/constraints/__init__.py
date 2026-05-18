"""
Constraints module: convert measurements into linear systems for optimization.

Provides:
  - skew_symmetric: cross-product matrix construction
  - build_linear_system: measurement set → A, b for least squares
  - compute_residuals: evaluate constraint violation
  - compute_scalar_residuals: per-measurement scalar errors

The core constraint is the ray coincidence condition:
    d_i × (X(t_i) - C_i) = 0
"""

from .geometric import (
    skew_symmetric,
    build_linear_system,
    compute_residuals,
    compute_scalar_residuals,
)

__all__ = [
    "skew_symmetric",
    "build_linear_system",
    "compute_residuals",
    "compute_scalar_residuals",
]
