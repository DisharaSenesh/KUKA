"""Centralised clock abstraction.

A Clock is a monotonic time source used to assign timestamps
to frames and detections.  Having a single clock object makes
it straightforward to inject a simulated clock during testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Clock:
    """Monotonic clock for timestamp generation.

    The clock records an epoch and returns elapsed seconds
    since that epoch on each call.  This ensures that all
    timestamps produced by a single acquisition run share
    a common time reference.

    Attributes
    ----------
    epoch : float
        Clock start time (seconds).  Set to ``time.monotonic()``
        at construction.
    """

    epoch: float = field(default_factory=time.monotonic)

    def now(self) -> float:
        """Current time in seconds since this clock's epoch.

        Returns
        -------
        float
            Elapsed seconds.
        """
        return time.monotonic() - self.epoch

    def reset(self) -> None:
        """Reset the epoch to the current monotonic time."""
        self.epoch = time.monotonic()

    def stamp(self) -> float:
        """Synonym for ``now``.  Produce a timestamp."""
        return self.now()
