"""
TrajectoryState: polynomial trajectory representation.

The object's 3D motion is modeled as a polynomial in time:
  X(t) = a0 + a1*t + a2*t² + a3*t³ + ...

where each a_k ∈ R³ is a 3D coefficient vector.

The state stores the coefficients {a0, a1, ..., a_k} for a polynomial
of order `order`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TrajectoryState:
    """
    Polynomial trajectory model for a moving 3D object.

    Attributes
    ----------
    coefficients : List[np.ndarray]
        List of 3D coefficient vectors: [a0, a1, a2, ...].
        a0 is the constant term (position at t=0 of the local model).
        a1 is the velocity.
        a2 is proportional to acceleration, etc.
    t0 : float
        Reference time for the local polynomial model.
        Evaluations use t' = t - t0 internally.
    order : int
        Polynomial order = number of coefficients - 1.
    """

    coefficients: List[np.ndarray]
    t0: float = 0.0

    @property
    def order(self) -> int:
        """Polynomial order: number of coefficients minus one."""
        return len(self.coefficients) - 1

    def __post_init__(self):
        self.coefficients = [np.asarray(c, dtype=np.float64) for c in self.coefficients]

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate the trajectory at time t (world coordinates).

        X(t) = Σ a_k * (t - t0)^k

        Parameters
        ----------
        t : float
            Time at which to evaluate the trajectory (seconds).

        Returns
        -------
        np.ndarray, shape (3,)
            3D position in world coordinates at time t.
        """
        dt = t - self.t0
        position = np.zeros(3, dtype=np.float64)
        for k, a_k in enumerate(self.coefficients):
            position += a_k * (dt ** k)
        return position

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """
        Evaluate the velocity (first derivative) at time t.

        dX/dt = Σ k * a_k * (t - t0)^(k-1)

        Parameters
        ----------
        t : float
            Time at which to evaluate velocity (seconds).

        Returns
        -------
        np.ndarray, shape (3,)
            Velocity vector in world coordinates.
        """
        dt = t - self.t0
        velocity = np.zeros(3, dtype=np.float64)
        for k in range(1, len(self.coefficients)):
            a_k = self.coefficients[k]
            velocity += k * a_k * (dt ** (k - 1))
        return velocity

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """
        Evaluate the acceleration (second derivative) at time t.

        d²X/dt² = Σ k*(k-1) * a_k * (t - t0)^(k-2)

        Parameters
        ----------
        t : float
            Time at which to evaluate acceleration (seconds).

        Returns
        -------
        np.ndarray, shape (3,)
            Acceleration vector in world coordinates.
        """
        dt = t - self.t0
        acceleration = np.zeros(3, dtype=np.float64)
        for k in range(2, len(self.coefficients)):
            a_k = self.coefficients[k]
            acceleration += k * (k - 1) * a_k * (dt ** (k - 2))
        return acceleration

    def retime(self, new_t0: float) -> TrajectoryState:
        """
        Re-express the trajectory relative to a new reference time.

        This changes the coefficients so that the trajectory still represents
        the same function X(t), but with a shifted t0.

        Parameters
        ----------
        new_t0 : float
            New reference time.

        Returns
        -------
        TrajectoryState
            Equivalent trajectory with shifted reference time.
        """
        n = len(self.coefficients)
        new_coeffs = [np.zeros(3, dtype=np.float64) for _ in range(n)]

        for k in range(n):
            for m in range(k, n):
                a_m = self.coefficients[m]
                binom = np.math.comb(m, k)
                shift = (new_t0 - self.t0) ** (m - k)
                new_coeffs[k] += binom * shift * a_m

        return TrajectoryState(coefficients=new_coeffs, t0=new_t0)
