"""
Sliding window: bounded collection of filtered measurements for local reconstruction.

The sliding window stores FilteredMeasurement objects, each wrapping a
SynchronizedMeasurement with observability and filtering metadata.

The window has configurable minimum and maximum sizes:
  - min_window_size: below this, reconstruction is skipped (weak geometry)
  - max_window_size: oldest measurements are evicted above this limit

The window is NOT system memory or logging. It IS the local geometric
constraint set for trajectory triangulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from core.synchronization.synchronized_measurement import SynchronizedMeasurement


# ---- FilteredMeasurement ----

@dataclass(frozen=True)
class FilteredMeasurement:
    """
    A synchronized measurement that has passed the geometric filtering gates.

    Wraps the canonical SynchronizedMeasurement with additional filtering
    metadata: observability scores, gate status, ray-angle diagnostics.

    This is the input type for the sliding window and trajectory reconstruction.

    Attributes
    ----------
    synchronized : SynchronizedMeasurement
        The underlying synchronized monocular observation.
    passed_ray_angle : bool
        True if the ray-angle gate was satisfied.
    passed_baseline : bool
        True if the baseline gate was satisfied.
    observability_score : float
        Combined observability score in [0, 1] from the filtering stage.
    min_ray_angle_rad : float
        Smallest angular separation to any accepted ray in the gate pool.
    """

    synchronized: SynchronizedMeasurement
    passed_ray_angle: bool = True
    passed_baseline: bool = True
    observability_score: float = 0.0
    min_ray_angle_rad: float = 0.0

    @property
    def is_valid(self) -> bool:
        """A filtered measurement is valid if it passed all gates."""
        return self.synchronized.is_valid and self.passed_ray_angle and self.passed_baseline


# ---- Sliding Window ----

@dataclass
class SlidingWindow:
    """
    Bounded sliding window of filtered measurements for local reconstruction.

    Maintains a time-ordered collection of FilteredMeasurement objects.
    Provides the local dataset for trajectory triangulation.

    Attributes
    ----------
    min_size : int
        Minimum measurements required before reconstruction is attempted.
    max_size : int
        Maximum measurements retained. Oldest evicted when exceeded.
    _measurements : deque
        Time-ordered filtered measurements (oldest → newest).
    """

    min_size: int = 5
    max_size: int = 50
    _measurements: deque = field(default_factory=deque)

    def add(self, filtered: FilteredMeasurement) -> None:
        """
        Add a filtered measurement to the window.

        Evicts oldest entries if the window exceeds max_size.

        Parameters
        ----------
        filtered : FilteredMeasurement
            A measurement that passed filtering gates.
        """
        self._measurements.append(filtered)
        while len(self._measurements) > self.max_size:
            self._measurements.popleft()

    def can_reconstruct(self) -> bool:
        """
        Check if enough measurements exist for reconstruction.

        Returns
        -------
        bool
            True if the window has at least min_size entries.
        """
        return len(self._measurements) >= self.min_size

    def get_measurements(self) -> List[FilteredMeasurement]:
        """
        Return all measurements currently in the window.

        Returns
        -------
        list of FilteredMeasurement
            Time-ordered measurements (oldest → newest).
        """
        return list(self._measurements)

    def get_valid_measurements(self) -> List[FilteredMeasurement]:
        """
        Return only measurements that passed all filtering gates.

        Returns
        -------
        list of FilteredMeasurement
        """
        return [m for m in self._measurements if m.is_valid]

    def get_synchronized(self) -> List[SynchronizedMeasurement]:
        """
        Return the underlying SynchronizedMeasurement objects.

        Returns
        -------
        list of SynchronizedMeasurement
        """
        return [m.synchronized for m in self._measurements]

    def time_span(self) -> float:
        """
        Time span covered by the window.

        Returns
        -------
        float
            latest_time - earliest_time, or 0.0 if empty.
        """
        if len(self._measurements) < 2:
            return 0.0
        return self._measurements[-1].synchronized.timestamp - self._measurements[0].synchronized.timestamp

    def latest_time(self) -> Optional[float]:
        """Timestamp of the most recent measurement."""
        if len(self._measurements) == 0:
            return None
        return self._measurements[-1].synchronized.timestamp

    def clear(self) -> None:
        """Remove all measurements from the window."""
        self._measurements.clear()

    def __len__(self) -> int:
        return len(self._measurements)

    def __bool__(self) -> bool:
        return self.can_reconstruct()
