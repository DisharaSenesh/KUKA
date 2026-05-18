"""FilteredMeasurement: a synchronised observation after geometric gating.

A FilteredMeasurement wraps the original SynchronizedMeasurement
with the outcome of geometric quality evaluation.  It records WHY
a measurement was accepted or rejected, together with the specific
metric values that drove the decision.

This type is the output of the filtering stage and the input to
downstream tracking / optimisation.  It provides the diagnostic
information needed for threshold tuning, offline analysis, and
debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.synchronization.synchronized_measurement import SynchronizedMeasurement


@dataclass(frozen=True)
class FilteredMeasurement:
    """A synchronised measurement after geometric quality gating.

    Attributes
    ----------
    measurement : SynchronizedMeasurement
        The original synchronised observation (temporal alignment already done).
    accepted : bool
        True if this measurement passed all geometric quality gates.
    ray_angle_deg : float or None
        Angular separation (degrees) to the nearest accepted ray direction.
        None for the first accepted measurement (no peer to compare against).
    baseline_m : float or None
        Camera baseline (metres) to the nearest accepted camera centre.
        None for the first accepted measurement.
    observability_score : float or None
        Overall observability score in [0, 1] for the accepted pool AFTER
        this measurement was evaluated.
    sync_error_s : float
        Temporal synchronisation error carried through from the raw
        measurement (seconds).
    rejection_reason : str or None
        Human-readable explanation of why the measurement was rejected.
        None if accepted.
    """

    measurement: SynchronizedMeasurement
    accepted: bool
    ray_angle_deg: Optional[float] = None
    baseline_m: Optional[float] = None
    observability_score: Optional[float] = None
    sync_error_s: float = 0.0
    rejection_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience accessors (delegate to wrapped measurement)
    # ------------------------------------------------------------------

    @property
    def frame_id(self) -> int:
        return self.measurement.frame_id

    @property
    def timestamp(self) -> float:
        return self.measurement.timestamp

    @property
    def u(self) -> float:
        return self.measurement.u

    @property
    def v(self) -> float:
        return self.measurement.v

    @property
    def pixel(self) -> tuple:
        return self.measurement.pixel

    def summary(self) -> str:
        """One-line human-readable summary including gate decision."""
        status = "ACCEPT" if self.accepted else "REJECT"
        parts = [
            f"[Filtered {status} frame={self.frame_id:04d}",
            f"uv=({self.u:.1f},{self.v:.1f})",
        ]
        if self.ray_angle_deg is not None:
            parts.append(f"angle={self.ray_angle_deg:.2f}°")
        if self.baseline_m is not None:
            parts.append(f"base={self.baseline_m:.4f}m")
        if self.rejection_reason is not None:
            parts.append(f"({self.rejection_reason})")
        return " ".join(parts) + "]"

    def to_csv_row(self) -> tuple:
        """Convert to a CSV row that includes filtering diagnostics.

        Returns
        -------
        tuple
            Extends the standard SynchronizedMeasurement CSV row with
            filtering diagnostics.
        """
        base = self.measurement.to_csv_row()
        return base + (
            int(self.accepted),
            self.ray_angle_deg,
            self.baseline_m,
            self.observability_score,
            self.rejection_reason or "",
        )


# Extended CSV header for filtered measurements
FILTERED_CSV_HEADER_EXTRA = [
    "accepted",
    "ray_angle_deg",
    "baseline_m",
    "observability_score",
    "rejection_reason",
]
