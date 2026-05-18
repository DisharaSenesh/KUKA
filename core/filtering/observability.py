"""
Geometric observability scoring for monocular multi-view reconstruction.

Observability in this context means: how well can the 3D trajectory be
recovered from the available measurements?

A measurement set is "observable" when:
  1. Ray directions have sufficient angular diversity
  2. Camera baselines provide adequate parallax
  3. No degenerate near-parallel ray pairs exist
  4. The combined geometry avoids rank-deficient constraint matrices

This module computes quantitative scores for each of these dimensions
and combines them into an overall observability assessment.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.types.measurement import Measurement
from core.types.pose import Pose
from .ray_filters import (
    angular_spread,
    smallest_pairwise_angle,
)
from .baseline_filters import (
    minimum_pairwise_baseline,
    baseline_quality_score,
)


@dataclass
class ObservabilityScore:
    """
    Quantitative observability assessment for a set of measurements.

    Attributes
    ----------
    angular_spread_rad : float
        Angular spread of ray directions (larger is better).
    min_pairwise_angle_rad : float
        Smallest angular separation between any two ray directions.
    min_baseline : float
        Minimum camera baseline distance (world units).
    baseline_quality : float
        Baseline quality score in [0, 1].
    n_measurements : int
        Number of measurements evaluated.
    overall_score : float
        Combined observability score in [0, 1], geometrically weighted.
    is_observable : bool
        True if all individual checks pass their thresholds.
    failure_reason : Optional[str]
        Explanation of why observability failed, if applicable.
    """

    angular_spread_rad: float = 0.0
    min_pairwise_angle_rad: float = 0.0
    min_baseline: float = 0.0
    baseline_quality: float = 0.0
    n_measurements: int = 0
    overall_score: float = 0.0
    is_observable: bool = False
    failure_reason: Optional[str] = None


def compute_observability(
    measurements: List[Measurement],
    min_angle_rad: float = 0.017,
    min_baseline: float = 0.1,
    reference_depth: float = 10.0
) -> ObservabilityScore:
    """
    Compute a full observability assessment for a set of measurements.

    Evaluates:
      1. Angular spread of world-frame ray directions
      2. Minimum pairwise ray angle (degeneracy check)
      3. Camera baseline quality
      4. Combined observability score

    Parameters
    ----------
    measurements : list of Measurement
        Measurements with world-frame rays.
    min_angle_rad : float
        Minimum acceptable ray angle (default ≈ 1°).
    min_baseline : float
        Minimum acceptable camera baseline (world units).
    reference_depth : float
        Approximate depth to target for baseline quality scaling.

    Returns
    -------
    ObservabilityScore
        Complete observability assessment.
    """
    n = len(measurements)
    if n < 2:
        return ObservabilityScore(
            n_measurements=n,
            is_observable=False,
            failure_reason=f"Need at least 2 measurements, got {n}"
        )

    # ---- Extract geometric primitives ----

    # Ray direction vectors (world frame, unit norm)
    directions = np.array([m.ray.direction for m in measurements], dtype=np.float64)

    # Camera centers from ray origins (world frame)
    origins = np.array([m.ray.origin for m in measurements], dtype=np.float64)

    # ---- Ray angular metrics ----

    spread = angular_spread(directions)
    min_angle = smallest_pairwise_angle(directions)

    # ---- Baseline metrics ----

    # Construct temporary Pose objects for baseline computation
    # (ray origin == camera center; rotation is not needed for baseline)
    poses = []
    for m in measurements:
        # Use identity rotation since we only need C for baseline
        R_identity = np.eye(3, dtype=np.float64)
        C = m.ray.origin
        poses.append(Pose(R=R_identity, C=C, t=m.t))

    min_base = minimum_pairwise_baseline(poses)
    base_quality = baseline_quality_score(poses, reference_depth)

    # ---- Combined score ----

    # Angle score: how spread out the rays are
    # Map [0, π/2] → [0, 1]: spread of 90° gives score 1.0
    angle_score = np.clip(spread / (np.pi / 2.0), 0.0, 1.0)

    # Min angle score: penalize near-parallel pairs
    # Map min_angle from [0, min_angle_rad] → [0, 1]
    if min_angle_rad > 0.0:
        min_angle_score = np.clip(min_angle / min_angle_rad, 0.0, 1.0)
    else:
        min_angle_score = 1.0

    # Combine: geometric mean of component scores
    overall = float(
        (angle_score * min_angle_score * base_quality) ** (1.0 / 3.0)
    )

    # ---- Gate checks ----

    is_observable = True
    failure_reason = None

    if min_angle < min_angle_rad:
        is_observable = False
        failure_reason = (
            f"Minimum pairwise ray angle ({np.degrees(min_angle):.2f}°) "
            f"below threshold ({np.degrees(min_angle_rad):.2f}°)"
        )
    elif min_base < min_baseline:
        is_observable = False
        failure_reason = (
            f"Minimum camera baseline ({min_base:.4f}) "
            f"below threshold ({min_baseline:.4f})"
        )
    elif n < 2:
        is_observable = False
        failure_reason = f"Too few measurements ({n})"

    return ObservabilityScore(
        angular_spread_rad=spread,
        min_pairwise_angle_rad=min_angle,
        min_baseline=min_base,
        baseline_quality=base_quality,
        n_measurements=n,
        overall_score=overall,
        is_observable=is_observable,
        failure_reason=failure_reason,
    )


def is_geometrically_observable(
    measurements: List[Measurement],
    min_angle_rad: float = 0.017,
    min_baseline: float = 0.1,
    min_measurements: int = 3
) -> bool:
    """
    Quick boolean check: is this measurement set geometrically sufficient?

    Parameters
    ----------
    measurements : list of Measurement
        Measurements to evaluate.
    min_angle_rad : float
        Minimum ray angular separation.
    min_baseline : float
        Minimum camera baseline.
    min_measurements : int
        Minimum number of measurements for a solvable system.

    Returns
    -------
    bool
        True if the set passes all geometric observability gates.
    """
    n = len(measurements)
    if n < min_measurements:
        return False

    score = compute_observability(measurements, min_angle_rad, min_baseline)
    return score.is_observable


def compute_observability_from_synchronized(
    measurements: list,
    min_angle_rad: float = 0.017,
    min_baseline: float = 0.1,
    reference_depth: float = 10.0,
) -> ObservabilityScore:
    """Compute observability directly from a list of SynchronizedMeasurement objects.

    This function internally extracts world-ray directions and camera centres
    from the synchronised measurements, then delegates to ``compute_observability``
    (which expects ``Measurement`` objects with pre-computed rays).

    The sync layer has already resolved temporal alignment; this function
    evaluates ONLY the geometric quality of the resulting view-set.

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement
        Synchronised measurements (temporal alignment already done).
    min_angle_rad : float
        Minimum acceptable ray angular separation (default ≈ 1°).
    min_baseline : float
        Minimum acceptable camera baseline (metres).
    reference_depth : float
        Approximate depth to target for baseline quality scaling.

    Returns
    -------
    ObservabilityScore
    """
    n = len(measurements)
    if n < 2:
        return ObservabilityScore(
            n_measurements=n,
            is_observable=False,
            failure_reason=f"Need at least 2 measurements, got {n}",
        )

    # ---- Extract world-frame ray directions ----
    from .ray_filters import measurement_ray_directions, angular_spread, smallest_pairwise_angle

    directions = measurement_ray_directions(measurements)
    spread = angular_spread(directions)
    min_angle = smallest_pairwise_angle(directions)

    # ---- Extract camera centres and compute baselines ----
    from .baseline_filters import (
        minimum_pairwise_baseline_between,
        baseline_quality_between,
    )

    min_base = minimum_pairwise_baseline_between(measurements)
    base_quality = baseline_quality_between(measurements, reference_depth)

    # ---- Combined score (same formula as compute_observability) ----
    angle_score = np.clip(spread / (np.pi / 2.0), 0.0, 1.0)

    if min_angle_rad > 0.0:
        min_angle_score = np.clip(min_angle / min_angle_rad, 0.0, 1.0)
    else:
        min_angle_score = 1.0

    overall = float((angle_score * min_angle_score * base_quality) ** (1.0 / 3.0))

    # ---- Gate checks ----
    is_observable = True
    failure_reason = None

    if min_angle < min_angle_rad:
        is_observable = False
        failure_reason = (
            f"Minimum pairwise ray angle ({np.degrees(min_angle):.2f}°) "
            f"below threshold ({np.degrees(min_angle_rad):.2f}°)"
        )
    elif min_base < min_baseline:
        is_observable = False
        failure_reason = (
            f"Minimum camera baseline ({min_base:.4f}) "
            f"below threshold ({min_baseline:.4f})"
        )

    return ObservabilityScore(
        angular_spread_rad=spread,
        min_pairwise_angle_rad=min_angle,
        min_baseline=min_base,
        baseline_quality=base_quality,
        n_measurements=n,
        overall_score=overall,
        is_observable=is_observable,
        failure_reason=failure_reason,
    )


def is_geometrically_observable_from_synchronized(
    measurements: list,
    min_angle_rad: float = 0.017,
    min_baseline: float = 0.1,
    min_measurements: int = 3,
) -> bool:
    """Quick boolean gate: are synchronised measurements geometrically sufficient?

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement
    min_angle_rad : float
    min_baseline : float
    min_measurements : int

    Returns
    -------
    bool
    """
    n = len(measurements)
    if n < min_measurements:
        return False
    score = compute_observability_from_synchronized(
        measurements, min_angle_rad, min_baseline
    )
    return score.is_observable


def condition_number_estimate(
    measurements: List[Measurement],
    order: int = 1,
    t0: Optional[float] = None
) -> float:
    """
    Estimate the condition number of the geometric constraint system.

    Uses the design matrix from the constraints module to assess
    numerical conditioning. A high condition number indicates
    near-degenerate geometry.

    This is an explicit observability metric: if the constraint matrix
    is ill-conditioned, the least-squares solution will be unreliable.

    Parameters
    ----------
    measurements : list of Measurement
        Measurements with world-frame rays.
    order : int
        Polynomial order for the trajectory model.
    t0 : float or None
        Reference time. If None, uses the latest measurement time.

    Returns
    -------
    float
        Estimated condition number (ratio of largest to smallest
        non-zero singular value). Returns ∞ for degenerate systems.

    Notes
    -----
    Delegates to constraints.geometric.build_linear_system for matrix
    construction. This is the only observability function that touches
    the constraints module (read-only: it only builds the system,
    does NOT solve it).
    """
    if len(measurements) < order + 1:
        return np.inf

    if t0 is None:
        t0 = measurements[-1].t

    # Import locally to avoid circular dependency at module level
    from core.constraints.geometric import build_linear_system

    A, _ = build_linear_system(measurements, order=order, t0=t0)

    # SVD for condition number analysis
    _, s, _ = np.linalg.svd(A, full_matrices=False)

    # Condition number: ratio of largest to smallest non-zero singular value
    # Use a tolerance to identify numerical zero
    tol = s[0] * max(A.shape) * np.finfo(float).eps
    nonzero = s[s > tol]

    if len(nonzero) == 0:
        return np.inf

    return float(s[0] / nonzero[-1])
