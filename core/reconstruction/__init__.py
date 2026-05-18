"""
Reconstruction module: continuous-time monocular trajectory estimation.

The reconstruction stage sits between filtering and prediction.
It takes FilteredMeasurements from a sliding window, builds geometric
ray constraints, and estimates a polynomial trajectory via least squares.

Pipeline:
  FilteredMeasurement → SlidingWindow → Triangulation → TrajectoryEstimate
                                              ↓
                                    Model Selection (auto order)
                                              ↓
                                    Reprojection Validation

Supports:
  - Manual polynomial order selection (debugging/experiments)
  - Automatic model selection with complexity penalty
  - Online (live stream) and offline (CSV replay) compatibility
"""

from .reconstruction_config import ReconstructionConfig
from .sliding_window import SlidingWindow, FilteredMeasurement
from .ray_builder import (
    synchronized_to_world_ray,
    synchronized_to_measurement,
    batch_to_measurements,
)
from .trajectory_model import (
    build_polynomial_basis,
    evaluate_trajectory_at_time,
    evaluate_trajectory_vectorized,
    num_coefficients,
    num_unknowns,
)
from .triangulation import triangulate
from .reprojection import (
    reprojection_error_single,
    compute_reprojection_errors,
    total_reprojection_cost,
    rms_reprojection_error,
)
from .residuals import (
    compute_geometric_residual,
    compute_all_residuals,
    rms_residual,
    max_residual,
)
from .model_selection import select_model_order, ModelSelectionResult
from .trajectory_estimate import TrajectoryEstimate

__all__ = [
    "ReconstructionConfig",
    "SlidingWindow",
    "FilteredMeasurement",
    "synchronized_to_world_ray",
    "synchronized_to_measurement",
    "batch_to_measurements",
    "build_polynomial_basis",
    "evaluate_trajectory_at_time",
    "evaluate_trajectory_vectorized",
    "num_coefficients",
    "num_unknowns",
    "triangulate",
    "reprojection_error_single",
    "compute_reprojection_errors",
    "total_reprojection_cost",
    "rms_reprojection_error",
    "compute_geometric_residual",
    "compute_all_residuals",
    "rms_residual",
    "max_residual",
    "select_model_order",
    "ModelSelectionResult",
    "TrajectoryEstimate",
]
