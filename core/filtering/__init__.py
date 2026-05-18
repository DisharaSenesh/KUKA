"""
Filtering layer: geometric observability and ray-quality validation.

Sits between measurement synchronisation and tracking/optimisation.
Evaluates measurements before they enter the solver.

Modules:
  - ray_filters:       angular separation between ray directions
  - baseline_filters:  camera baseline distance validation
  - observability:     combined geometric quality scoring
  - measurement_gate:  accept/reject orchestrators
  - filtered_measurement: output type with full diagnostics
"""

# ---- Ray-angle filters ----
from .ray_filters import (
    ray_angle,
    has_valid_ray_angle,
    smallest_pairwise_angle,
    angular_spread,
    # SynchronizedMeasurement bridges
    measurement_ray_direction,
    measurement_ray_directions,
    ray_angle_between_measurements,
    has_valid_ray_angle_between,
    smallest_pairwise_angle_between,
    angular_spread_between,
)

# ---- Baseline filters ----
from .baseline_filters import (
    baseline_distance,
    has_valid_baseline,
    minimum_pairwise_baseline,
    baseline_quality_score,
    # SynchronizedMeasurement bridges
    measurement_camera_centre,
    baseline_between_measurements,
    has_valid_baseline_between,
    minimum_pairwise_baseline_between,
    baseline_quality_between,
)

# ---- Observability ----
from .observability import (
    ObservabilityScore,
    compute_observability,
    is_geometrically_observable,
    condition_number_estimate,
    # SynchronizedMeasurement entry points
    compute_observability_from_synchronized,
    is_geometrically_observable_from_synchronized,
)

# ---- Gates ----
from .measurement_gate import (
    GateConfig,
    GateResult,
    MeasurementGate,
    SyncGateConfig,
    SynchronizedMeasurementGate,
)

# ---- Filtered output ----
from .filtered_measurement import (
    FilteredMeasurement,
    FILTERED_CSV_HEADER_EXTRA,
)

__all__ = [
    # Ray filters (pure)
    "ray_angle",
    "has_valid_ray_angle",
    "smallest_pairwise_angle",
    "angular_spread",
    # Ray filters (sync bridges)
    "measurement_ray_direction",
    "measurement_ray_directions",
    "ray_angle_between_measurements",
    "has_valid_ray_angle_between",
    "smallest_pairwise_angle_between",
    "angular_spread_between",
    # Baseline filters (pure)
    "baseline_distance",
    "has_valid_baseline",
    "minimum_pairwise_baseline",
    "baseline_quality_score",
    # Baseline filters (sync bridges)
    "measurement_camera_centre",
    "baseline_between_measurements",
    "has_valid_baseline_between",
    "minimum_pairwise_baseline_between",
    "baseline_quality_between",
    # Observability
    "ObservabilityScore",
    "compute_observability",
    "is_geometrically_observable",
    "condition_number_estimate",
    "compute_observability_from_synchronized",
    "is_geometrically_observable_from_synchronized",
    # Gates
    "GateConfig",
    "GateResult",
    "MeasurementGate",
    "SyncGateConfig",
    "SynchronizedMeasurementGate",
    # Filtered output
    "FilteredMeasurement",
    "FILTERED_CSV_HEADER_EXTRA",
]
