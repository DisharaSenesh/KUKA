"""Timestamp generation and management utilities.

Timestamps are monotonically non-decreasing float values
recording seconds since an epoch. The epoch is chosen by
the acquisition source (e.g. system time, camera clock).
"""

import time
from typing import Generator


def monotonic_timestamp() -> float:
    """Return a monotonic timestamp in seconds.

    Uses ``time.monotonic()`` so that the result is guaranteed
    never to decrease even if the system clock is adjusted.

    Returns
    -------
    float
        Current monotonic time in seconds.
    """
    return time.monotonic()


def wall_timestamp() -> float:
    """Return a wall-clock timestamp in seconds.

    Uses ``time.time()`` which may jump under NTP adjustments.

    Returns
    -------
    float
        Current wall-clock time in seconds.
    """
    return time.time()


def timestamp_generator(
    start_time: float,
    frame_rate: float = 30.0,
) -> Generator[float, None, None]:
    """Infinite generator of synthetic timestamps at a fixed rate.

    Useful for processing offline video files where the camera
    driver does not report per-frame timestamps.

    Parameters
    ----------
    start_time : float
        Timestamp assigned to frame 0.
    frame_rate : float
        Frames per second. Default 30.0.

    Yields
    ------
    float
        Next timestamp in the sequence.
    """
    dt = 1.0 / frame_rate
    t = start_time
    while True:
        yield t
        t += dt


def stamp_now() -> float:
    """One-shot convenience alias for ``monotonic_timestamp``.

    Returns
    -------
    float
        Current monotonic time in seconds.
    """
    return monotonic_timestamp()
