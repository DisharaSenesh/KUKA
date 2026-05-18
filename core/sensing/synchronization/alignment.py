"""Timestamp alignment across asynchronous sensor streams.

When measurements from multiple cameras (or a camera and an
external trigger) arrive with independent clocks, alignment
finds corresponding timestamps and optionally estimates a
linear offset between two time bases.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def nearest_timestamp(
    target: float,
    candidates: np.ndarray,
) -> Tuple[int, float, float]:
    """Find the candidate timestamp closest to the target.

    Parameters
    ----------
    target : float
        Desired timestamp.
    candidates : np.ndarray
        1-D array of available timestamps, shape (N,).

    Returns
    -------
    idx : int
        Index of the closest candidate.
    value : float
        Timestamp value of the closest candidate.
    delta : float
        Absolute difference ``|candidates[idx] - target|``.
    """
    deltas = np.abs(np.asarray(candidates, dtype=np.float64) - target)
    idx = int(np.argmin(deltas))
    return idx, float(candidates[idx]), float(deltas[idx])


def alignment_offset(
    timestamps_a: np.ndarray,
    timestamps_b: np.ndarray,
) -> float:
    """Estimate a constant offset between two timestamp sequences.

    Assumes ``t_b ≈ offset + t_a`` and estimates the offset
    by averaging pairwise differences.  The sequences must be
    the same length and correspond to the same events measured
    by independent clocks.

    Parameters
    ----------
    timestamps_a : np.ndarray
        Reference timestamps, shape (N,).
    timestamps_b : np.ndarray
        Secondary timestamps for the same events, shape (N,).

    Returns
    -------
    offset : float
        Mean of (timestamps_b - timestamps_a).
    """
    a = np.asarray(timestamps_a, dtype=np.float64)
    b = np.asarray(timestamps_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("timestamp arrays must have the same length")
    return float(np.mean(b - a))


def align_series(
    queries: np.ndarray,
    reference: np.ndarray,
    max_delta: float = 0.05,
) -> List[Optional[int]]:
    """Match each query timestamp to an index in the reference stream.

    A query is considered matched if there exists a reference
    timestamp within ``max_delta`` seconds; otherwise it is
    recorded as ``None``.

    Parameters
    ----------
    queries : np.ndarray
        Query timestamps, shape (M,).
    reference : np.ndarray
        Reference timestamps, shape (N,).
    max_delta : float
        Maximum allowed temporal distance for a match (seconds).

    Returns
    -------
    matches : list of int or None
        One entry per query: the index of the best-matching
        reference timestamp, or None if no match within range.
    """
    q = np.asarray(queries, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)

    matches: List[Optional[int]] = []
    for t in q:
        deltas = np.abs(r - t)
        idx = int(np.argmin(deltas))
        if deltas[idx] <= max_delta:
            matches.append(idx)
        else:
            matches.append(None)
    return matches
