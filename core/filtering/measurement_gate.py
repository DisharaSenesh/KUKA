"""
MeasurementGate: accepts or rejects monocular measurements.

A small stateful orchestration class that evaluates incoming measurements
against geometric quality thresholds. The gate maintains a pool of
previously accepted measurements and checks new candidates for:
  - sufficient ray angular diversity
  - adequate camera baseline
  - overall geometric observability

This is the ONLY class in the filtering layer with mutable state.
All threshold logic is explicit and configurable.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from core.types.measurement import Measurement
from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from .ray_filters import ray_angle, smallest_pairwise_angle
from .baseline_filters import baseline_distance, minimum_pairwise_baseline
from .observability import compute_observability, ObservabilityScore
from .filtered_measurement import FilteredMeasurement


@dataclass
class GateConfig:
    """
    Configuration for the measurement gate.

    Attributes
    ----------
    min_ray_angle_rad : float
        Minimum angular separation between any two accepted ray directions.
        Default 0.017 rad ≈ 1°. Below this, triangulation is unstable.
    min_baseline : float
        Minimum distance between any two camera centers (world units).
        Default 0.1 m. Below this, parallax is insufficient.
    max_measurements : int
        Maximum measurements to retain in the gate.
        Default 100. Older measurements are evicted when full.
    require_observability : bool
        If True, run the full observability check on every accept.
        Slightly more expensive but catches subtle degeneracies.
    """
    min_ray_angle_rad: float = 0.017
    min_baseline: float = 0.1
    max_measurements: int = 100
    require_observability: bool = True


@dataclass
class GateResult:
    """
    Result of a gate evaluation.

    Attributes
    ----------
    accepted : bool
        Whether the measurement passed all gates.
    reason : str
        Explanation of accept/reject decision.
    angle_to_nearest_rad : float
        Angular separation to the nearest accepted ray direction.
    baseline_to_nearest : float
        Baseline distance to the nearest accepted camera center.
    observability : ObservabilityScore or None
        Full observability score if computed.
    """
    accepted: bool
    reason: str
    angle_to_nearest_rad: float = 0.0
    baseline_to_nearest: float = 0.0
    observability: ObservabilityScore = None


@dataclass
class MeasurementGate:
    """
    Gating mechanism for monocular measurement acceptance.

    Maintains a pool of previously accepted measurements and evaluates
    new candidates against geometric quality thresholds.

    The gate does NOT:
      - modify rays or measurements
      - build constraint systems
      - solve optimization problems
      - depend on tracking internals

    Attributes
    ----------
    config : GateConfig
        Threshold and behavior configuration.
    accepted : List[Measurement]
        Pool of previously accepted measurements.
    rejected_count : int
        Total number of rejected measurements (diagnostic).
    """

    config: GateConfig = field(default_factory=GateConfig)
    accepted: List[Measurement] = field(default_factory=list)
    rejected_count: int = 0

    def accept(self, measurement: Measurement) -> GateResult:
        """
        Evaluate a single measurement for acceptance.

        The measurement must pass:
          1. Minimum ray-angle check against ALL accepted rays
          2. Minimum baseline check against ALL accepted camera centers
          3. (Optionally) full observability check on the candidate set

        The first measurement is always accepted (no peers to check against).

        Parameters
        ----------
        measurement : Measurement
            Candidate measurement to evaluate.

        Returns
        -------
        GateResult
            Evaluation result with accept/reject and diagnostic info.
        """
        # First measurement: always accept (need a seed)
        if len(self.accepted) == 0:
            self.accepted.append(measurement)
            return GateResult(
                accepted=True,
                reason="First measurement (no peers to check)",
            )

        d_new = measurement.ray.direction
        C_new = measurement.ray.origin

        # ---- Ray-angle gate ----
        # Find the smallest angular separation to any accepted ray
        min_angle_found = np.pi
        for existing in self.accepted:
            angle = ray_angle(d_new, existing.ray.direction)
            if angle < min_angle_found:
                min_angle_found = angle

        if min_angle_found < self.config.min_ray_angle_rad:
            self.rejected_count += 1
            return GateResult(
                accepted=False,
                reason=(
                    f"Ray angle ({np.degrees(min_angle_found):.2f}°) "
                    f"below minimum ({np.degrees(self.config.min_ray_angle_rad):.2f}°)"
                ),
                angle_to_nearest_rad=min_angle_found,
            )

        # ---- Baseline gate ----
        # Find the smallest baseline to any accepted camera center
        min_baseline_found = np.inf
        for existing in self.accepted:
            b = np.linalg.norm(C_new - existing.ray.origin)
            if b < min_baseline_found:
                min_baseline_found = b

        if min_baseline_found < self.config.min_baseline:
            self.rejected_count += 1
            return GateResult(
                accepted=False,
                reason=(
                    f"Camera baseline ({min_baseline_found:.4f}) "
                    f"below minimum ({self.config.min_baseline:.4f})"
                ),
                angle_to_nearest_rad=min_angle_found,
                baseline_to_nearest=min_baseline_found,
            )

        # ---- Full observability check (optional) ----
        if self.config.require_observability:
            candidate_set = self.accepted + [measurement]
            obs = compute_observability(
                candidate_set,
                min_angle_rad=self.config.min_ray_angle_rad,
                min_baseline=self.config.min_baseline,
            )
            if not obs.is_observable:
                self.rejected_count += 1
                return GateResult(
                    accepted=False,
                    reason=f"Observability failed: {obs.failure_reason}",
                    angle_to_nearest_rad=min_angle_found,
                    baseline_to_nearest=min_baseline_found,
                    observability=obs,
                )

        # ---- Accept ----
        self.accepted.append(measurement)

        # Evict oldest if over capacity
        while len(self.accepted) > self.config.max_measurements:
            self.accepted.pop(0)

        obs_result = None
        if self.config.require_observability and len(self.accepted) >= 2:
            obs_result = compute_observability(
                self.accepted,
                min_angle_rad=self.config.min_ray_angle_rad,
                min_baseline=self.config.min_baseline,
            )

        return GateResult(
            accepted=True,
            reason="Passed all geometric quality gates",
            angle_to_nearest_rad=min_angle_found,
            baseline_to_nearest=min_baseline_found,
            observability=obs_result,
        )

    def accept_batch(
        self,
        measurements: List[Measurement]
    ) -> Tuple[List[Measurement], List[Measurement]]:
        """
        Evaluate and partition a batch of measurements.

        Parameters
        ----------
        measurements : list of Measurement
            Batch of candidate measurements.

        Returns
        -------
        accepted : list of Measurement
            Measurements that passed the gate.
        rejected : list of Measurement
            Measurements that were rejected.
        """
        accepted_list = []
        rejected_list = []

        for meas in measurements:
            result = self.accept(meas)
            if result.accepted:
                accepted_list.append(meas)
            else:
                rejected_list.append(meas)

        return accepted_list, rejected_list

    def current_observability(self) -> ObservabilityScore:
        """
        Compute the observability score for the current accepted set.

        Returns
        -------
        ObservabilityScore
            Evaluation of the current measurement pool.
        """
        return compute_observability(
            self.accepted,
            min_angle_rad=self.config.min_ray_angle_rad,
            min_baseline=self.config.min_baseline,
        )

    def clear(self) -> None:
        """Reset the gate: clear all accepted measurements and counters."""
        self.accepted.clear()
        self.rejected_count = 0

    def __len__(self) -> int:
        return len(self.accepted)


# ======================================================================
# SynchronizedMeasurementGate — operates on SynchronizedMeasurement
# ======================================================================
# This gate is the FIRST geometric quality stage after synchronisation.
# It consumes SynchronizedMeasurement objects (temporal alignment done),
# evaluates geometric observability, and produces FilteredMeasurement
# objects with full diagnostics.
#
# The gate internally constructs world-ray directions and camera centres
# from the Euler-angle / mm position fields.  Rotation matrices are
# temporary — Euler angles remain the primary representation.
# ======================================================================


@dataclass
class SyncGateConfig:
    """Configuration for the synchronised-measurement gate.

    Attributes
    ----------
    min_ray_angle_deg : float
        Minimum angular separation between any two accepted ray directions
        (degrees).  Default 1.0°.  Nearly parallel rays produce unstable
        triangulation.
    min_baseline_m : float
        Minimum camera baseline (metres).  Default 0.01 m.  A tiny baseline
        means the camera barely moved — parallax is insufficient.
    max_measurements : int
        Maximum measurements retained in the gate.  Default 100.
    require_observability : bool
        If True, run the full observability check on every accept.
    max_sync_error_s : float
        Maximum acceptable synchronisation error (seconds).  Measurements
        whose temporal alignment error exceeds this threshold are rejected
        regardless of geometry.  Default 0.1 s.
    """

    min_ray_angle_deg: float = 1.0
    min_baseline_m: float = 0.01
    max_measurements: int = 100
    require_observability: bool = True
    max_sync_error_s: float = 0.1


@dataclass
class SynchronizedMeasurementGate:
    """Gating mechanism for synchronised measurements.

    Evaluates incoming ``SynchronizedMeasurement`` objects against
    geometric quality thresholds.  Produces ``FilteredMeasurement``
    objects with acceptance status and diagnostic metrics.

    The gate is the ONLY class in the filtering layer that stores
    mutable state (the pool of accepted measurements).  All threshold
    logic is explicit and configurable via ``SyncGateConfig``.

    Attributes
    ----------
    config : SyncGateConfig
        Threshold and behaviour configuration.
    accepted : list of SynchronizedMeasurement
        Pool of measurements that passed all quality gates.
    rejected_count : int
        Total number of rejected measurements (diagnostic counter).
    """

    config: SyncGateConfig = field(default_factory=SyncGateConfig)
    accepted: list = field(default_factory=list)
    rejected_count: int = 0

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    def accept(
        self,
        measurement: SynchronizedMeasurement,
    ) -> FilteredMeasurement:
        """Evaluate a single synchronised measurement for geometric quality.

        The measurement must pass:
          1. Synchronisation error gate (temporal alignment check)
          2. Minimum ray-angle check against all accepted rays
          3. Minimum baseline check against all accepted camera centres
          4. (Optionally) full observability check on the extended pool

        The first measurement is always accepted (no peers to compare).

        Parameters
        ----------
        measurement : SynchronizedMeasurement
            Candidate measurement after temporal synchronisation.

        Returns
        -------
        FilteredMeasurement
            Wrapped measurement with acceptance decision and diagnostics.
        """
        from .ray_filters import measurement_ray_direction, ray_angle
        from .baseline_filters import measurement_camera_centre
        from .filtered_measurement import FilteredMeasurement

        # ---- Synchronisation error gate ----
        if measurement.sync_error_s > self.config.max_sync_error_s:
            self.rejected_count += 1
            return FilteredMeasurement(
                measurement=measurement,
                accepted=False,
                sync_error_s=measurement.sync_error_s,
                rejection_reason=(
                    f"Sync error ({measurement.sync_error_s*1000:.1f} ms) "
                    f"exceeds max ({self.config.max_sync_error_s*1000:.1f} ms)"
                ),
            )

        # ---- First measurement: always accept (seeding) ----
        if len(self.accepted) == 0:
            self.accepted.append(measurement)
            return FilteredMeasurement(
                measurement=measurement,
                accepted=True,
                sync_error_s=measurement.sync_error_s,
                ray_angle_deg=None,          # no peer to compare
                baseline_m=None,              # no peer to compare
                observability_score=None,     # single measurement — not observable yet
            )

        # ---- Extract geometric primitives from the candidate ----
        d_new = measurement_ray_direction(measurement)
        C_new = measurement_camera_centre(measurement)

        min_angle_rad_threshold = np.radians(self.config.min_ray_angle_deg)

        # ---- Ray-angle gate ----
        # Find the nearest accepted ray direction — the CLOSEST pair
        # determines whether triangulation would be degenerate.
        min_angle_found_rad = np.pi
        for existing in self.accepted:
            d_existing = measurement_ray_direction(existing)
            angle = ray_angle(d_new, d_existing)
            if angle < min_angle_found_rad:
                min_angle_found_rad = angle

        min_angle_found_deg = np.degrees(min_angle_found_rad)

        if min_angle_found_rad < min_angle_rad_threshold:
            self.rejected_count += 1
            return FilteredMeasurement(
                measurement=measurement,
                accepted=False,
                ray_angle_deg=min_angle_found_deg,
                sync_error_s=measurement.sync_error_s,
                rejection_reason=(
                    f"Ray angle ({min_angle_found_deg:.2f}°) "
                    f"below minimum ({self.config.min_ray_angle_deg:.2f}°). "
                    f"Nearly parallel rays give unstable depth triangulation."
                ),
            )

        # ---- Baseline gate ----
        # The baseline is the Euclidean distance between camera centres.
        # A camera that has barely moved provides no parallax.
        min_baseline_found = float("inf")
        for existing in self.accepted:
            C_existing = measurement_camera_centre(existing)
            b = float(np.linalg.norm(C_new - C_existing))
            if b < min_baseline_found:
                min_baseline_found = b

        if min_baseline_found < self.config.min_baseline_m:
            self.rejected_count += 1
            return FilteredMeasurement(
                measurement=measurement,
                accepted=False,
                ray_angle_deg=min_angle_found_deg,
                baseline_m=min_baseline_found,
                sync_error_s=measurement.sync_error_s,
                rejection_reason=(
                    f"Camera baseline ({min_baseline_found:.4f} m) "
                    f"below minimum ({self.config.min_baseline_m:.4f} m). "
                    f"Insufficient parallax for reconstruction."
                ),
            )

        # ---- Full observability check (optional) ----
        obs_result = None
        if self.config.require_observability:
            from .observability import compute_observability_from_synchronized

            obs_result = compute_observability_from_synchronized(
                self.accepted + [measurement],
                min_angle_rad=min_angle_rad_threshold,
                min_baseline=self.config.min_baseline_m,
            )
            if not obs_result.is_observable:
                self.rejected_count += 1
                return FilteredMeasurement(
                    measurement=measurement,
                    accepted=False,
                    ray_angle_deg=min_angle_found_deg,
                    baseline_m=min_baseline_found,
                    observability_score=obs_result.overall_score,
                    sync_error_s=measurement.sync_error_s,
                    rejection_reason=f"Observability failed: {obs_result.failure_reason}",
                )

        # ---- Accept ----
        self.accepted.append(measurement)

        # Evict oldest if over capacity
        while len(self.accepted) > self.config.max_measurements:
            self.accepted.pop(0)

        # Compute post-accept observability for diagnostics
        post_obs = None
        if self.config.require_observability and len(self.accepted) >= 2:
            from .observability import compute_observability_from_synchronized
            post_obs = compute_observability_from_synchronized(
                self.accepted,
                min_angle_rad=min_angle_rad_threshold,
                min_baseline=self.config.min_baseline_m,
            )

        return FilteredMeasurement(
            measurement=measurement,
            accepted=True,
            ray_angle_deg=min_angle_found_deg,
            baseline_m=min_baseline_found,
            observability_score=post_obs.overall_score if post_obs else None,
            sync_error_s=measurement.sync_error_s,
        )

    # ------------------------------------------------------------------
    # Batch processing (offline replay)
    # ------------------------------------------------------------------

    def accept_batch(
        self,
        measurements: list,
    ) -> Tuple[List[FilteredMeasurement], List[FilteredMeasurement]]:
        """Evaluate and partition a batch of synchronised measurements.

        Parameters
        ----------
        measurements : list of SynchronizedMeasurement

        Returns
        -------
        accepted : list of FilteredMeasurement
        rejected : list of FilteredMeasurement
        """
        accepted_list = []
        rejected_list = []

        for m in measurements:
            result = self.accept(m)
            if result.accepted:
                accepted_list.append(result)
            else:
                rejected_list.append(result)

        return accepted_list, rejected_list

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def current_observability(self) -> ObservabilityScore:
        """Observability score for the current accepted pool.

        Returns
        -------
        ObservabilityScore
        """
        from .observability import compute_observability_from_synchronized

        min_angle_rad = np.radians(self.config.min_ray_angle_deg)
        return compute_observability_from_synchronized(
            self.accepted,
            min_angle_rad=min_angle_rad,
            min_baseline=self.config.min_baseline_m,
        )

    def clear(self) -> None:
        """Reset the gate: clear accepted pool and rejection counter."""
        self.accepted.clear()
        self.rejected_count = 0

    def __len__(self) -> int:
        return len(self.accepted)
