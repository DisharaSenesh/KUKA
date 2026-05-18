"""
Ray-angle quality filters.

Evaluates angular separation between world-frame ray directions.
Rays with small angular separation produce unstable triangulation
because depth estimation becomes ill-conditioned.

Mathematical foundation:

Given two unit direction vectors d1, d2 ∈ S² (world frame):

    cos(θ) = d1 · d2
    θ = arccos( clip(cos(θ), -1, 1) )

Interpretation:
  θ ≈ 0°    → rays nearly parallel → depth ambiguous (bad)
  θ ≈ 90°   → rays orthogonal      → good triangulation
  θ ≈ 180°  → rays opposite        → good (wide baseline)

All functions are pure and operate on unit direction vectors.
"""

import numpy as np


def ray_angle(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    Compute the 3D angular separation between two unit ray directions.

    θ = arccos(d1 · d2)

    Parameters
    ----------
    d1 : np.ndarray, shape (3,)
        First unit direction vector (world frame).
    d2 : np.ndarray, shape (3,)
        Second unit direction vector (world frame).

    Returns
    -------
    float
        Angle in radians, in [0, π].

    Notes
    -----
    Uses explicit clipping before arccos for numerical stability
    when the dot product slightly exceeds [-1, 1] due to floating point.
    """
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    # Dot product of unit vectors gives cos(θ)
    cos_theta = np.dot(d1, d2)

    # Clamp to valid arccos domain for numerical safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)


def has_valid_ray_angle(
    d1: np.ndarray,
    d2: np.ndarray,
    min_angle_rad: float
) -> bool:
    """
    Check whether two ray directions have sufficient angular separation.

    Parameters
    ----------
    d1 : np.ndarray, shape (3,)
        First unit direction vector.
    d2 : np.ndarray, shape (3,)
        Second unit direction vector.
    min_angle_rad : float
        Minimum acceptable angle in radians (e.g., 0.017 ≈ 1°).

    Returns
    -------
    bool
        True if the angular separation is at least min_angle_rad.
    """
    angle = ray_angle(d1, d2)
    return angle >= min_angle_rad


def smallest_pairwise_angle(directions: np.ndarray) -> float:
    """
    Find the smallest angular separation among all pairs of ray directions.

    This is a key observability metric: if ANY pair of rays is nearly
    parallel, the overall reconstruction is at risk.

    Parameters
    ----------
    directions : np.ndarray, shape (n, 3)
        Array of n unit direction vectors (world frame).

    Returns
    -------
    float
        Minimum pairwise angular separation in radians.

    Notes
    -----
    For n directions, there are n*(n-1)/2 pairs. This function evaluates
    all pairs explicitly (O(n²)) for clarity. For large n, consider
    subsampling or incremental evaluation.
    """
    n = len(directions)
    if n < 2:
        return 0.0

    min_angle = np.pi  # worst-case initialization

    for i in range(n):
        for j in range(i + 1, n):
            angle = ray_angle(directions[i], directions[j])
            if angle < min_angle:
                min_angle = angle

    return min_angle


def angular_spread(directions: np.ndarray) -> float:
    """
    Compute the angular spread of a set of ray directions.

    Angular spread is defined as the angle of the smallest cone that
    contains all direction vectors. A larger spread indicates better
    triangulation geometry.

    Uses the principal eigenvalue analysis: the spread is measured by
    the complement of the largest dot product between any direction
    and the mean direction (Riemannian centroid approximation).

    Parameters
    ----------
    directions : np.ndarray, shape (n, 3)
        Array of unit direction vectors.

    Returns
    -------
    float
        Angular spread in radians, in [0, π].
        Spread = 0 means all rays are parallel.
        Spread = π means rays span the full sphere.
    """
    n = len(directions)
    if n < 2:
        return 0.0

    # Compute mean direction via Euclidean average then renormalize
    mean_dir = np.mean(directions, axis=0)
    mean_norm = np.linalg.norm(mean_dir)

    if mean_norm < 1e-10:
        # Directions cancel out (antipodal symmetry)
        # Fall back to max pairwise angle
        return _max_pairwise_angle(directions)

    mean_dir = mean_dir / mean_norm

    # Find the maximum deviation from the mean direction
    max_cos = 1.0
    for i in range(n):
        cos_angle = np.clip(np.dot(directions[i], mean_dir), -1.0, 1.0)
        if cos_angle < max_cos:
            max_cos = cos_angle

    # The spread is the angular radius of the enclosing cone
    spread = np.arccos(max_cos)
    return spread


def _max_pairwise_angle(directions: np.ndarray) -> float:
    """Compute maximum pairwise angular separation (internal helper)."""
    n = len(directions)
    max_angle = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            angle = ray_angle(directions[i], directions[j])
            if angle > max_angle:
                max_angle = angle
    return max_angle


# ---------------------------------------------------------------------------
# SynchronizedMeasurement → world-ray direction extraction
# ---------------------------------------------------------------------------
# These bridge functions convert the flat SynchronizedMeasurement fields
# (u, v, A_deg, B_deg, C_deg, fx, fy, cx, cy) into world-frame unit ray
# directions that the existing pure ray_filters functions can consume.
#
# The conversion follows the same convention as pixel_to_world_ray:
#   1. Camera-frame direction from (u, v) via the pinhole model
#   2. Euler angles → rotation matrix R (KUKA intrinsic Z-Y-X)
#   3. World direction d_world = R.T @ d_camera
#
# Euler angles are kept in degrees in the SynchronizedMeasurement and
# converted to radians internally.  Rotation matrices are temporary.
# ---------------------------------------------------------------------------


def measurement_ray_direction(
    sm: "SynchronizedMeasurement",
) -> np.ndarray:
    """Extract the world-frame unit ray direction from a synchronised measurement.

    Parameters
    ----------
    sm : SynchronizedMeasurement
        Synchronised measurement with u, v, fx, fy, cx, cy, A_deg, B_deg, C_deg.

    Returns
    -------
    np.ndarray, shape (3,)
        Unit direction vector in world frame.
    """
    # Camera-frame normalised direction via pinhole back-projection
    x_norm = (sm.u - sm.cx) / sm.fx
    y_norm = (sm.v - sm.cy) / sm.fy
    d_camera = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    d_camera /= np.linalg.norm(d_camera)

    # KUKA Euler angles → rotation matrix
    a_rad = np.radians(sm.A_deg)
    b_rad = np.radians(sm.B_deg)
    c_rad = np.radians(sm.C_deg)

    # Avoid circular import: import euler_abc_to_rotation locally
    from core.robotics.kinematics.transforms import euler_abc_to_rotation
    R = euler_abc_to_rotation(a_rad, b_rad, c_rad)

    # Transform camera-frame direction → world frame
    # Follows the Pose.camera_ray_to_world convention: d_world = R.T @ d_camera
    d_world = R.T @ d_camera
    return d_world


def measurement_ray_directions(
    measurements: list,
) -> np.ndarray:
    """Extract world-frame ray directions from a list of synchronised measurements.

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement

    Returns
    -------
    np.ndarray, shape (n, 3)
        Stacked unit direction vectors in world frame.
    """
    dirs = [measurement_ray_direction(m) for m in measurements]
    if not dirs:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(dirs, dtype=np.float64)


def ray_angle_between_measurements(
    sm1: "SynchronizedMeasurement",
    sm2: "SynchronizedMeasurement",
) -> float:
    """Angular separation (radians) between two synchronised measurements.

    Parameters
    ----------
    sm1, sm2 : SynchronizedMeasurement

    Returns
    -------
    float
        Angle in radians, in [0, π].
    """
    d1 = measurement_ray_direction(sm1)
    d2 = measurement_ray_direction(sm2)
    return ray_angle(d1, d2)


def has_valid_ray_angle_between(
    sm1: "SynchronizedMeasurement",
    sm2: "SynchronizedMeasurement",
    min_angle_rad: float,
) -> bool:
    """Check whether two synchronised measurements have sufficient angular separation.

    Parameters
    ----------
    sm1, sm2 : SynchronizedMeasurement
    min_angle_rad : float
        Minimum acceptable angle in radians.

    Returns
    -------
    bool
    """
    angle = ray_angle_between_measurements(sm1, sm2)
    return angle >= min_angle_rad


def smallest_pairwise_angle_between(
    measurements: list,
) -> float:
    """Minimum pairwise ray angle (radians) among a list of synchronised measurements.

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement

    Returns
    -------
    float
        Minimum pairwise angular separation in radians.
    """
    directions = measurement_ray_directions(measurements)
    return smallest_pairwise_angle(directions)


def angular_spread_between(
    measurements: list,
) -> float:
    """Angular spread (radians) of ray directions from synchronised measurements.

    Parameters
    ----------
    measurements : list of SynchronizedMeasurement

    Returns
    -------
    float
        Angular spread in radians, in [0, π].
    """
    directions = measurement_ray_directions(measurements)
    return angular_spread(directions)
