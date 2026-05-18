"""
Prediction module: forward extrapolation of the trajectory estimate.

Depends only on the trajectory state (polynomial coefficients).
Does NOT depend on cameras, measurements, or optimization.

Provides:
  - single-point prediction
  - trajectory sampling at regular intervals
  - time-to-target estimation for interception
"""

from .extrapolator import (
    predict_position,
    predict_trajectory,
    predict_interception_time,
)

__all__ = [
    "predict_position",
    "predict_trajectory",
    "predict_interception_time",
]
