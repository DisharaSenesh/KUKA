"""
SlidingWindowTracker: online trajectory estimator.

Orchestrates the full estimation pipeline for a sliding window of
monocular measurements:

  1. Accept measurements into the buffer
  2. When enough measurements exist, build the linear constraint system
  3. Solve for optimal polynomial coefficients
  4. Update the trajectory state

The tracker stores the current estimate and provides evaluation
at arbitrary timestamps (including future times for prediction).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.types.measurement import Measurement
from core.types.trajectory_state import TrajectoryState
from core.constraints.geometric import build_linear_system
from core.optimization.least_squares import (
    solve_least_squares,
    solve_weighted_least_squares,
    unflatten_coefficients,
)
from .buffer import MeasurementBuffer


@dataclass
class TrackerConfig:
    """
    Configuration for the sliding window tracker.

    Attributes
    ----------
    window_duration : float
        Duration of the sliding time window in seconds.
    polynomial_order : int
        Order of the polynomial trajectory model.
        order=1: constant velocity (2 coefficients)
        order=2: constant acceleration (3 coefficients)
        order=3: constant jerk (4 coefficients)
    min_measurements : int
        Minimum number of measurements required before solving.
        Must be at least polynomial_order + 1.
    """
    window_duration: float = 1.0
    polynomial_order: int = 2
    min_measurements: int = 3

    def __post_init__(self):
        if self.min_measurements < self.polynomial_order + 1:
            raise ValueError(
                f"min_measurements ({self.min_measurements}) must be >= "
                f"polynomial_order + 1 ({self.polynomial_order + 1})"
            )


@dataclass
class SlidingWindowTracker:
    """
    Online sliding-window trajectory estimator.

    Manages a measurement buffer, builds constraint systems from
    the current window, solves for polynomial coefficients, and
    maintains the current trajectory state estimate.

    Attributes
    ----------
    config : TrackerConfig
        Tracker configuration parameters.
    buffer : MeasurementBuffer
        Sliding window measurement buffer.
    state : Optional[TrajectoryState]
        Current trajectory estimate (None until first solve).
    last_solve_time : float
        Timestamp of the most recent successful solve.
    """

    config: TrackerConfig = field(default_factory=TrackerConfig)
    buffer: MeasurementBuffer = field(init=False)
    state: Optional[TrajectoryState] = None
    last_solve_time: float = 0.0

    def __post_init__(self):
        self.buffer = MeasurementBuffer(window_duration=self.config.window_duration)

    def add_measurement(self, measurement: Measurement) -> None:
        """
        Add a monocular measurement to the sliding window.

        Parameters
        ----------
        measurement : Measurement
            World-frame ray measurement with timestamp.
        """
        self.buffer.add_measurement(measurement)

    def has_estimate(self) -> bool:
        """Check if a trajectory estimate is available."""
        return self.state is not None

    def can_solve(self) -> bool:
        """Check if enough measurements are available to solve."""
        return len(self.buffer) >= self.config.min_measurements

    def solve(self) -> TrajectoryState:
        """
        Build the constraint system and solve for trajectory coefficients.

        Uses the current measurement buffer to construct the geometric
        linear system and solve via least squares.

        Returns
        -------
        TrajectoryState
            The updated trajectory estimate.

        Raises
        ------
        RuntimeError
            If insufficient measurements are available.
        """
        if not self.can_solve():
            raise RuntimeError(
                f"Not enough measurements: have {len(self.buffer)}, "
                f"need {self.config.min_measurements}"
            )

        measurements = self.buffer.get_measurements()

        # Use the latest measurement time as the polynomial reference
        t0 = measurements[-1].t

        # Build the geometric constraint linear system
        A, b = build_linear_system(
            measurements=measurements,
            order=self.config.polynomial_order,
            t0=t0
        )

        # Solve for flattened coefficients
        x_opt = solve_least_squares(A, b)

        # Unflatten into list of 3D coefficient vectors
        coefficients = unflatten_coefficients(x_opt, self.config.polynomial_order)

        # Update trajectory state
        self.state = TrajectoryState(coefficients=coefficients, t0=t0)
        self.last_solve_time = measurements[-1].t

        return self.state

    def solve_weighted(self, weights: np.ndarray) -> TrajectoryState:
        """
        Solve with per-measurement weights.

        Parameters
        ----------
        weights : np.ndarray, shape (n_measurements,)
            Positive weights for each measurement.

        Returns
        -------
        TrajectoryState
            The updated trajectory estimate.
        """
        if not self.can_solve():
            raise RuntimeError(
                f"Not enough measurements: have {len(self.buffer)}, "
                f"need {self.config.min_measurements}"
            )

        measurements = self.buffer.get_measurements()
        t0 = measurements[-1].t

        A, b = build_linear_system(
            measurements=measurements,
            order=self.config.polynomial_order,
            t0=t0
        )

        x_opt = solve_weighted_least_squares(A, b, weights)

        coefficients = unflatten_coefficients(x_opt, self.config.polynomial_order)

        self.state = TrajectoryState(coefficients=coefficients, t0=t0)
        self.last_solve_time = measurements[-1].t

        return self.state

    def get_position(self, t: float) -> Optional[np.ndarray]:
        """
        Evaluate the estimated trajectory at time t.

        Parameters
        ----------
        t : float
            Evaluation time (can be in the future for prediction).

        Returns
        -------
        np.ndarray or None
            3D position estimate, or None if no estimate exists.
        """
        if self.state is None:
            return None
        return self.state.evaluate(t)

    def get_velocity(self, t: float) -> Optional[np.ndarray]:
        """
        Evaluate the estimated velocity at time t.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        np.ndarray or None
            3D velocity estimate, or None if no estimate exists.
        """
        if self.state is None:
            return None
        return self.state.evaluate_velocity(t)

    def get_state(self) -> Optional[TrajectoryState]:
        """Return the current trajectory state estimate."""
        return self.state

    def reset(self) -> None:
        """Clear all measurements and reset the state estimate."""
        self.buffer.clear()
        self.state = None
        self.last_solve_time = 0.0
