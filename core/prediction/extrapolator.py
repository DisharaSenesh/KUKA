"""
Trajectory extrapolation from polynomial state estimates.

All functions are pure and depend only on the TrajectoryState.
No camera geometry or optimization logic here.
"""

from typing import List

import numpy as np

from core.types.trajectory_state import TrajectoryState


def predict_position(
    state: TrajectoryState,
    t_future: float
) -> np.ndarray:
    """
    Predict the 3D position at a future time using the polynomial model.

    Evaluates: X(t_future) = Σ a_k * (t_future - t0)^k

    Parameters
    ----------
    state : TrajectoryState
        Current trajectory estimate (polynomial coefficients and t0).
    t_future : float
        Future time at which to predict (must be >= state.t0).

    Returns
    -------
    np.ndarray, shape (3,)
        Predicted 3D position in world coordinates.
    """
    return state.evaluate(t_future)


def predict_trajectory(
    state: TrajectoryState,
    t_start: float,
    t_end: float,
    n_samples: int
) -> np.ndarray:
    """
    Sample the predicted trajectory at regular intervals.

    Parameters
    ----------
    state : TrajectoryState
        Current trajectory estimate.
    t_start : float
        First sample time.
    t_end : float
        Last sample time.
    n_samples : int
        Number of samples (must be >= 2).

    Returns
    -------
    np.ndarray, shape (n_samples, 3)
        Array of 3D positions along the predicted trajectory.
    """
    times = np.linspace(t_start, t_end, n_samples)
    positions = np.zeros((n_samples, 3), dtype=np.float64)

    for i, t in enumerate(times):
        positions[i] = state.evaluate(t)

    return positions


def predict_interception_time(
    state: TrajectoryState,
    target_point: np.ndarray,
    t_search_start: float,
    t_search_end: float,
    tolerance: float = 1e-3,
    max_iterations: int = 20
) -> float:
    """
    Estimate the time when the trajectory passes closest to a target point.

    Uses bisection on the derivative of squared distance to find the
    minimum-distance time within the search interval.

    Parameters
    ----------
    state : TrajectoryState
        Current trajectory estimate.
    target_point : np.ndarray, shape (3,)
        Target point in world coordinates.
    t_search_start : float
        Start of the search interval.
    t_search_end : float
        End of the search interval.
    tolerance : float
        Time convergence tolerance in seconds.
    max_iterations : int
        Maximum number of bisection iterations.

    Returns
    -------
    float
        Estimated time of closest approach within the search interval.
    """
    target = np.asarray(target_point, dtype=np.float64)
    t_a = t_search_start
    t_b = t_search_end

    # Evaluate distance-squared at endpoints
    da2 = np.sum((state.evaluate(t_a) - target) ** 2)
    db2 = np.sum((state.evaluate(t_b) - target) ** 2)

    # If one endpoint is closer, shrink toward it
    for _ in range(max_iterations):
        if abs(t_b - t_a) < tolerance:
            break

        t_mid = (t_a + t_b) / 2.0
        dm2 = np.sum((state.evaluate(t_mid) - target) ** 2)

        # Use golden-section style refinement: choose the better half
        t_quarter_a = (3 * t_a + t_b) / 4.0
        t_quarter_b = (t_a + 3 * t_b) / 4.0

        dqa2 = np.sum((state.evaluate(t_quarter_a) - target) ** 2)
        dqb2 = np.sum((state.evaluate(t_quarter_b) - target) ** 2)

        if dqa2 < dqb2:
            t_b = t_quarter_b
        else:
            t_a = t_quarter_a

    best_t = (t_a + t_b) / 2.0
    return best_t
