"""
MeasurementBuffer: stores and manages time-ordered measurements.

Maintains a sliding window of monocular measurements, automatically
evicting old measurements that fall outside the window.

The buffer uses a fixed-size deque for O(1) append and O(1) pop from left.
Measurements are assumed to arrive in monotonically increasing time order.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List

from core.types.measurement import Measurement


@dataclass
class MeasurementBuffer:
    """
    A sliding-window buffer of monocular measurements.

    Attributes
    ----------
    window_duration : float
        Time window length in seconds. Measurements older than
        (latest_time - window_duration) are evicted.
    measurements : deque
        Time-ordered measurements within the current window.
    """

    window_duration: float
    measurements: deque = field(default_factory=deque)

    @property
    def latest_time(self) -> float:
        """Timestamp of the most recent measurement."""
        if len(self.measurements) == 0:
            return 0.0
        return self.measurements[-1].t

    @property
    def earliest_time(self) -> float:
        """Timestamp of the oldest measurement in the buffer."""
        if len(self.measurements) == 0:
            return 0.0
        return self.measurements[0].t

    def __len__(self) -> int:
        return len(self.measurements)

    def add_measurement(self, measurement: Measurement) -> None:
        """
        Add a new measurement and evict old ones outside the window.

        Parameters
        ----------
        measurement : Measurement
            New measurement to add. Must have timestamp >= latest_time.

        Raises
        ------
        ValueError
            If measurement timestamp is before the latest timestamp
            (out-of-order measurements are not supported).
        """
        if len(self.measurements) > 0 and measurement.t < self.measurements[-1].t:
            raise ValueError(
                f"Out-of-order measurement: t={measurement.t:.4f} < "
                f"latest={self.measurements[-1].t:.4f}"
            )

        self.measurements.append(measurement)
        self._evict_old()

    def _evict_old(self) -> None:
        """Remove measurements that fall outside the sliding window."""
        cutoff_time = self.latest_time - self.window_duration
        while len(self.measurements) > 0 and self.measurements[0].t < cutoff_time:
            self.measurements.popleft()

    def get_measurements(self) -> List[Measurement]:
        """
        Return all measurements currently in the buffer.

        Returns
        -------
        list of Measurement
            Time-ordered measurements within the window.
        """
        return list(self.measurements)

    def clear(self) -> None:
        """Remove all measurements from the buffer."""
        self.measurements.clear()
