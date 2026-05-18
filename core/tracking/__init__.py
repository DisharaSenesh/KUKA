"""
Tracking module: time-windowed trajectory estimation.

Orchestrates the estimation pipeline:
  1. Collect monocular measurements within a time window
  2. Build the geometric constraint system
  3. Solve for polynomial trajectory coefficients
  4. Update the trajectory state estimate

The tracker does NOT implement geometry or optimization internally.
It delegates to the constraints and optimization modules.
"""

from .buffer import MeasurementBuffer
from .sliding_window import SlidingWindowTracker

__all__ = [
    "MeasurementBuffer",
    "SlidingWindowTracker",
]
