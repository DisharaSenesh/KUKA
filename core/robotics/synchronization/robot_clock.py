"""
Robot clock: centralized timestamp management.

In a real-time system, timestamps must be consistent across all
subsystems. This module provides the canonical time source for
robot measurements, avoiding scattered `time.time()` calls.

Uses Python's `time.perf_counter()` for monotonic, high-resolution
timing suitable for robotics control loops.
"""

import time
from dataclasses import dataclass, field


@dataclass
class RobotClock:
    """
    Monotonic clock for consistent timestamp generation.

    Provides a single time source for all robot measurements.
    Uses perf_counter() which is monotonic and high-resolution.

    Attributes
    ----------
    t_start : float
        Clock reading at initialization (reference zero).
    offset : float
        Optional offset to align with an external time source.
    """

    t_start: float = field(default_factory=time.perf_counter)
    offset: float = 0.0

    def now(self) -> float:
        """
        Get the current robot time in seconds.

        Returns
        -------
        float
            Monotonic time in seconds since initialization (plus offset).
        """
        return time.perf_counter() - self.t_start + self.offset

    def reset(self) -> None:
        """Reset the clock to zero at the current instant."""
        self.t_start = time.perf_counter()
        self.offset = 0.0

    def set_offset(self, offset_seconds: float) -> None:
        """
        Apply a fixed offset to align with an external clock.

        Parameters
        ----------
        offset_seconds : float
            Offset to add to all future time readings.
        """
        self.offset = offset_seconds


def wall_time_now() -> float:
    """
    Get the current wall-clock time.

    Returns
    -------
    float
        Current time in seconds since the Unix epoch.
    """
    return time.time()
