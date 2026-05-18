"""Simple ring buffer for timestamped sensor data.

A lightweight buffer that maintains a sliding window of recent
frames or detections, keyed by timestamp.  Useful for bridging
asynchronous sensor streams before alignment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SensorBuffer:
    """Sliding-window buffer for timestamped items.

    Items are inserted with an associated timestamp and evicted
    automatically when they fall outside the configured window.

    Attributes
    ----------
    window_duration : float
        How long (in seconds) items are retained.
    _items : deque
        Internal storage of (timestamp, payload) pairs.
    """

    window_duration: float
    _items: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self):
        if self.window_duration <= 0:
            raise ValueError("window_duration must be positive")
        self._items = deque()

    def push(self, timestamp: float, payload: Any) -> None:
        """Insert an item and evict stale entries.

        Parameters
        ----------
        timestamp : float
            Item timestamp.
        payload : any
            Associated data (e.g. Frame, Detection).
        """
        self._items.append((timestamp, payload))
        self._evict(timestamp)

    def _evict(self, current_time: float) -> None:
        """Remove items older than ``current_time - window_duration``."""
        cutoff = current_time - self.window_duration
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()

    def snapshot(self) -> List[Any]:
        """Return all payloads currently in the buffer, oldest first.

        Returns
        -------
        list
            Payloads in chronological order.
        """
        return [item for _, item in self._items]

    def latest(self) -> Optional[Any]:
        """Return the most recent payload, or None if empty."""
        if not self._items:
            return None
        return self._items[-1][1]

    def oldest(self) -> Optional[Any]:
        """Return the oldest payload, or None if empty."""
        if not self._items:
            return None
        return self._items[0][1]

    def clear(self) -> None:
        """Empty the buffer."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def latest_timestamp(self) -> Optional[float]:
        """Timestamp of the most recent item, or None."""
        if not self._items:
            return None
        return self._items[-1][0]

    @property
    def earliest_timestamp(self) -> Optional[float]:
        """Timestamp of the oldest item, or None."""
        if not self._items:
            return None
        return self._items[0][0]
