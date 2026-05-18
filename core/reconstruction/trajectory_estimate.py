"""
Trajectory estimate: the output of the reconstruction stage.

Contains the estimated polynomial trajectory, fitting diagnostics,
and window metadata. This is the data type consumed by future
prediction, robot control, and evaluation stages.

The estimate represents a continuous trajectory over the current
sliding window — not a single triangulated point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TrajectoryEstimate:
    """
    A continuous-time polynomial trajectory estimate from local reconstruction.

    Attributes
    ----------
    order : int
        Polynomial order used (1 = linear, 2 = quadratic, 3 = cubic).
    t0 : float
        Reference time for the polynomial model.
    coefficients : list of np.ndarray
        List of 3D coefficient vectors [a0, a1, ..., a_order].
    reprojection_rms : float
        RMS reprojection error in pixels.
    geometric_rms : float
        RMS geometric residual in meters.
    max_geometric_residual : float
        Maximum geometric residual in meters.
    window_size : int
        Number of measurements used in the reconstruction.
    window_time_span : float
        Time span covered by the window (seconds).
    window_t_start : float
        Earliest measurement timestamp in the window.
    window_t_end : float
        Latest measurement timestamp in the window.
    solvable : bool
        True if the reconstruction produced valid coefficients.
    failure_reason : Optional[str]
        Explanation if reconstruction failed (e.g., too few measurements).
    """

    order: int
    t0: float
    coefficients: List[np.ndarray] = field(default_factory=list)
    reprojection_rms: float = float("inf")
    geometric_rms: float = float("inf")
    max_geometric_residual: float = float("inf")
    window_size: int = 0
    window_time_span: float = 0.0
    window_t_start: float = 0.0
    window_t_end: float = 0.0
    solvable: bool = False
    failure_reason: Optional[str] = None

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate the trajectory at time t.

        X(t) = Σ a_k * (t - t0)^k

        Parameters
        ----------
        t : float
            Evaluation time (seconds).

        Returns
        -------
        np.ndarray, shape (3,)
            3D position at time t. Returns zeros if not solvable.
        """
        if not self.solvable:
            return np.zeros(3, dtype=np.float64)

        dt = t - self.t0
        pos = np.zeros(3, dtype=np.float64)
        for k, a_k in enumerate(self.coefficients):
            pos += a_k * (dt ** k)
        return pos

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """
        Evaluate trajectory velocity (first derivative).

        Returns
        -------
        np.ndarray, shape (3,)
        """
        if not self.solvable or len(self.coefficients) < 2:
            return np.zeros(3, dtype=np.float64)

        dt = t - self.t0
        vel = np.zeros(3, dtype=np.float64)
        for k in range(1, len(self.coefficients)):
            vel += float(k) * self.coefficients[k] * (dt ** (k - 1))
        return vel

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.solvable:
            return f"[TrajEst] FAILED: {self.failure_reason}"

        return (
            f"[TrajEst] order={self.order} "
            f"reproj_rms={self.reprojection_rms:.2f}px "
            f"geom_rms={self.geometric_rms*1000:.2f}mm "
            f"window=[{self.window_t_start:.3f}, {self.window_t_end:.3f}] "
            f"n={self.window_size}"
        )

    @classmethod
    def failed(cls, reason: str) -> "TrajectoryEstimate":
        """Create a TrajectoryEstimate representing a failed reconstruction."""
        return cls(
            order=0,
            t0=0.0,
            solvable=False,
            failure_reason=reason,
        )
