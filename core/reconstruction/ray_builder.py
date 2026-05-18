"""
Ray builder: converts synchronized measurements into world-space viewing rays.

Each SynchronizedMeasurement carries:
  - pixel coordinates (u, v) and camera intrinsics (fx, fy, cx, cy)
  - robot pose in KUKA-native units (X,Y,Z in mm; A,B,C in deg)
  - frame timestamp

The ray builder converts these into a world-frame Ray:
  - origin: camera center C in world coordinates (meters)
  - direction: unit vector from camera through the pixel (world frame)

It also produces a Measurement (from core.types) for the existing
constraint and optimization infrastructure.

Euler angles are converted to rotation matrices temporarily for the
ray computation, then discarded. The stored data remains as raw angles.
"""

import numpy as np

from core.types.ray import Ray
from core.types.measurement import Measurement
from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from core.robotics.kinematics.transforms import (
    euler_abc_to_rotation,
    KUKA_POSITION_SCALE,
)


def synchronized_to_world_ray(
    sm: SynchronizedMeasurement,
) -> Ray:
    """
    Convert a synchronized measurement into a world-frame viewing ray.

    Steps:
      1. Convert KUKA Euler angles (deg) to rotation matrix
      2. Convert position from mm to meters
      3. Backproject pixel → camera-frame direction
      4. Rotate direction into world frame
      5. Set ray origin = camera center (world)

    Parameters
    ----------
    sm : SynchronizedMeasurement
        A valid synchronized measurement with finite pose.

    Returns
    -------
    Ray
        World-frame ray: origin = C, direction = unit vector.

    Notes
    -----
    The rotation matrix is constructed from (A, B, C) and then used
    only for this computation. Euler angles in the storage remain
    untouched.
    """
    # Step 1: Build camera-to-world rotation matrix from Euler angles
    a_rad = np.radians(sm.A_deg)
    b_rad = np.radians(sm.B_deg)
    c_rad = np.radians(sm.C_deg)

    R_camera_to_world = euler_abc_to_rotation(a_rad, b_rad, c_rad)

    # Step 2: Camera center in world frame (mm → meters)
    C_world = np.array([
        sm.X_mm * KUKA_POSITION_SCALE,
        sm.Y_mm * KUKA_POSITION_SCALE,
        sm.Z_mm * KUKA_POSITION_SCALE,
    ], dtype=np.float64)

    # Step 3: Backproject pixel to normalized camera-frame direction
    x_norm = (sm.u - sm.cx) / sm.fx
    y_norm = (sm.v - sm.cy) / sm.fy

    d_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    d_cam = d_cam / np.linalg.norm(d_cam)

    # Step 4: Rotate direction into world frame
    # camera-to-world rotation maps camera vectors → world vectors
    d_world = R_camera_to_world.T @ d_cam  # R.T maps camera→world
    d_world = d_world / np.linalg.norm(d_world)

    # Step 5: Build world-frame ray
    return Ray(origin=C_world, direction=d_world, frame="world")


def synchronized_to_measurement(
    sm: SynchronizedMeasurement,
) -> Measurement:
    """
    Convert a synchronized measurement into a core.types.Measurement.

    The Measurement type is what the existing constraint system
    (build_linear_system, compute_residuals) expects.

    Parameters
    ----------
    sm : SynchronizedMeasurement
        A valid synchronized measurement.

    Returns
    -------
    Measurement
        World-frame ray + timestamp + pixel observation.
    """
    from core.types.measurement import PixelObservation

    ray = synchronized_to_world_ray(sm)

    pixel = PixelObservation(
        u=sm.u,
        v=sm.v,
        fx=sm.fx,
        fy=sm.fy,
        cx=sm.cx,
        cy=sm.cy,
    )

    return Measurement(ray=ray, t=sm.timestamp, pixel=pixel)


def batch_to_measurements(
    sync_measurements: list,
) -> list:
    """
    Convert a batch of synchronized measurements to core.types.Measurement objects.

    Skips invalid measurements (those with no valid pose).

    Parameters
    ----------
    sync_measurements : list of SynchronizedMeasurement

    Returns
    -------
    list of Measurement
        Only measurements with valid, finite poses.
    """
    measurements = []
    for sm in sync_measurements:
        if not sm.is_valid or not sm.is_position_valid():
            continue
        try:
            meas = synchronized_to_measurement(sm)
            measurements.append(meas)
        except Exception:
            continue
    return measurements
