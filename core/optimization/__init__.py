"""
Optimization module: least squares solvers for trajectory estimation.

Solvers operate on pre-built linear systems (A, b) from the constraints module
and produce polynomial coefficient estimates.

The problem: minimize ‖ A * x - b ‖²

Where x = [a0_x, a0_y, a0_z, a1_x, a1_y, a1_z, ...] are the
flattened polynomial coefficients.
"""

from .least_squares import (
    solve_least_squares,
    solve_weighted_least_squares,
)

__all__ = [
    "solve_least_squares",
    "solve_weighted_least_squares",
]
