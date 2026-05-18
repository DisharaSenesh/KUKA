"""
Kinematic transforms: convert between robot pose representations and
our geometric Pose type.

The KUKA robot reports its flange/tcp position as (X, Y, Z, A, B, C):
  - X, Y, Z : position in millimeters (robot base frame)
  - A, B, C : Euler angles in degrees (intrinsic Z-Y-X convention)

The camera pose in world frame is computed via the kinematic chain:
  T_world_camera = T_world_base @ T_base_flange @ T_flange_tcp @ T_tcp_camera

All angles are in radians internally. Conversions to/from degrees happen
only at the I/O boundary.
"""

import numpy as np

from core.types.pose import Pose

# KUKA uses millimeters internally; we convert to meters for the math layer
KUKA_POSITION_SCALE = 0.001  # mm → m


def euler_abc_to_rotation(a_rad: float, b_rad: float, c_rad: float) -> np.ndarray:
    """
    Convert KUKA ABC Euler angles (intrinsic Z-Y-X) to a 3×3 rotation matrix.

    The KUKA convention uses intrinsic rotations:
      1. Rotate about Z by A
      2. Rotate about new Y' by B
      3. Rotate about new X'' by C

    This is equivalent to extrinsic X-Y-Z rotations:
      R = Rz(A) @ Ry(B) @ Rx(C)

    The resulting matrix R maps vectors from the rotated frame to the
    reference (base) frame: v_base = R @ v_rotated

    Parameters
    ----------
    a_rad : float
        Rotation about Z axis (radians).
    b_rad : float
        Rotation about Y axis (radians).
    c_rad : float
        Rotation about X axis (radians).

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix R (orthonormal, det=1).
    """
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)
    cos_b = np.cos(b_rad)
    sin_b = np.sin(b_rad)
    cos_c = np.cos(c_rad)
    sin_c = np.sin(c_rad)

    # Rz(A)
    Rz = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    # Ry(B)
    Ry = np.array([
        [cos_b,  0.0, sin_b],
        [0.0,    1.0, 0.0],
        [-sin_b, 0.0, cos_b]
    ], dtype=np.float64)

    # Rx(C)
    Rx = np.array([
        [1.0, 0.0,    0.0],
        [0.0, cos_c, -sin_c],
        [0.0, sin_c,  cos_c]
    ], dtype=np.float64)

    # Intrinsic Z-Y-X = Rz(A) @ Ry(B) @ Rx(C)
    return Rz @ Ry @ Rx


def rotation_to_euler_abc(R: np.ndarray) -> tuple:
    """
    Convert a 3×3 rotation matrix to KUKA ABC Euler angles.

    Decomposes R = Rz(A) @ Ry(B) @ Rx(C) to extract A, B, C.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    tuple of (float, float, float)
        Euler angles (A, B, C) in radians.
        B is in [-π/2, π/2]; if |cos(B)| ≈ 0, gimbal lock occurs and
        A + C (or A - C) is returned instead (with B = ±π/2).
    """
    R = np.asarray(R, dtype=np.float64)

    # R = Rz(A) @ Ry(B) @ Rx(C)
    # R[2,0] = -sin(B)
    # R[2,1] = cos(B) * sin(C)
    # R[2,2] = cos(B) * cos(C)
    # R[1,0] = cos(B) * sin(A)
    # R[0,0] = cos(B) * cos(A)

    sin_b = -R[2, 0]
    cos_b = np.sqrt(R[2, 1]**2 + R[2, 2]**2)

    if cos_b > 1e-10:
        # Non-singular case
        b = np.arctan2(sin_b, cos_b)
        a = np.arctan2(R[1, 0], R[0, 0])
        c = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock: |cos(B)| ≈ 0, B ≈ ±π/2
        # Cannot distinguish A and C uniquely; fix C = 0
        b = np.pi / 2.0 if sin_b > 0 else -np.pi / 2.0
        if sin_b > 0:
            # B = +π/2: R[0,1] = -sin(A-C), R[0,2] = cos(A-C)
            #  → A-C = atan2(-R[0,1], R[0,2])
            a_minus_c = np.arctan2(-R[0, 1], R[0, 2])
            a = a_minus_c
            c = 0.0
        else:
            # B = -π/2: R[0,1] = -sin(A+C), R[0,2] = -cos(A+C)
            #  → A+C = atan2(-R[0,1], -R[0,2])
            a_plus_c = np.arctan2(-R[0, 1], -R[0, 2])
            a = a_plus_c
            c = 0.0

    return (float(a), float(b), float(c))


def kuka_pose_to_rotation_translation(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    a_deg: float,
    b_deg: float,
    c_deg: float
) -> tuple:
    """
    Convert a raw KUKA pose reading into rotation matrix and translation.

    Parameters
    ----------
    x_mm, y_mm, z_mm : float
        Position in millimeters (robot base frame).
    a_deg, b_deg, c_deg : float
        Euler angles in degrees (KUKA ABC convention).

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix (flange-to-base).
    t : np.ndarray, shape (3,)
        Translation vector in meters (base frame).
    """
    a_rad = np.radians(a_deg)
    b_rad = np.radians(b_deg)
    c_rad = np.radians(c_deg)

    R = euler_abc_to_rotation(a_rad, b_rad, c_rad)

    t = np.array([
        x_mm * KUKA_POSITION_SCALE,
        y_mm * KUKA_POSITION_SCALE,
        z_mm * KUKA_POSITION_SCALE,
    ], dtype=np.float64)

    return R, t


def compute_camera_pose(
    robot_x_mm: float,
    robot_y_mm: float,
    robot_z_mm: float,
    robot_a_deg: float,
    robot_b_deg: float,
    robot_c_deg: float,
    flange_to_tcp: np.ndarray = None,
    tcp_to_camera_R: np.ndarray = None,
    tcp_to_camera_t: np.ndarray = None,
    timestamp: float = 0.0,
) -> Pose:
    """
    Compute the camera Pose in world frame from a robot TCP reading.

    Kinematic chain:
      T_world_camera = T_world_base @ T_base_flange @ T_flange_tcp @ T_tcp_camera

    By default, world = base (identity) and flange = TCP (identity),
    so the robot reading directly gives the TCP pose in world frame.
    The camera-to-TCP transform is a calibrated static offset.

    Parameters
    ----------
    robot_x_mm, robot_y_mm, robot_z_mm : float
        Robot TCP position in millimeters (base frame).
    robot_a_deg, robot_b_deg, robot_c_deg : float
        Robot TCP orientation in degrees (KUKA ABC convention).
    flange_to_tcp : np.ndarray or None, shape (3,)
        Translation from flange to TCP in meters. None means zero.
    tcp_to_camera_R : np.ndarray or None, shape (3, 3)
        Rotation from camera frame to TCP frame. None means identity.
    tcp_to_camera_t : np.ndarray or None, shape (3,)
        Translation from camera to TCP in meters. None means zero.
    timestamp : float
        Observation timestamp (seconds).

    Returns
    -------
    Pose
        Camera pose in world frame:
          R: maps camera-frame vectors → world-frame vectors
          C: camera optical center in world coordinates
    """
    # Step 1: Robot flange pose in base frame (from KUKA reading)
    R_flange_in_base, t_flange_in_base = kuka_pose_to_rotation_translation(
        x_mm=robot_x_mm,
        y_mm=robot_y_mm,
        z_mm=robot_z_mm,
        a_deg=robot_a_deg,
        b_deg=robot_b_deg,
        c_deg=robot_c_deg,
    )

    # Step 2: Apply flange-to-TCP offset if provided
    if flange_to_tcp is not None:
        t_tcp_in_base = t_flange_in_base + R_flange_in_base @ np.asarray(flange_to_tcp, dtype=np.float64)
    else:
        t_tcp_in_base = t_flange_in_base  # assume TCP = flange (no tool offset)

    # TCP orientation in base frame (same as flange if no tool rotation offset)
    R_tcp_in_base = R_flange_in_base

    # Step 3: Apply TCP-to-camera transform (static calibration)
    if tcp_to_camera_R is not None and tcp_to_camera_t is not None:
        R_cam_in_tcp = np.asarray(tcp_to_camera_R, dtype=np.float64)
        t_cam_in_tcp = np.asarray(tcp_to_camera_t, dtype=np.float64)

        # Camera pose in base frame via chain: base ← flange ← tcp ← camera
        R_camera_in_base = R_tcp_in_base @ R_cam_in_tcp
        t_camera_in_base = t_tcp_in_base + R_tcp_in_base @ t_cam_in_tcp
    else:
        # Default: camera at TCP (no offset)
        R_camera_in_base = R_tcp_in_base
        t_camera_in_base = t_tcp_in_base

    # Step 4: Build Pose in world frame (world ≡ base by default)
    # Pose.R maps camera-frame vectors to world-frame vectors
    # R_camera_in_base maps camera→base, so R = R_camera_in_base
    # Pose.C is the camera center in world coordinates
    C_world = t_camera_in_base

    # The rotation stored in Pose is camera-to-world
    R_camera_to_world = R_camera_in_base

    return Pose(R=R_camera_to_world, C=C_world, t=timestamp)


def kuka_target_from_pose(pose: Pose) -> tuple:
    """
    Convert a world-frame Pose to KUKA target format (mm, degrees).

    Used to send motion targets computed from the trajectory system
    back to the robot.

    Parameters
    ----------
    pose : Pose
        Target pose in world frame (R: camera-to-world, C: camera center).

    Returns
    -------
    tuple of (x_mm, y_mm, z_mm, a_deg, b_deg, c_deg)
        KUKA target in millimeters and degrees.
    """
    # Camera center → position in mm
    x_mm = pose.C[0] / KUKA_POSITION_SCALE
    y_mm = pose.C[1] / KUKA_POSITION_SCALE
    z_mm = pose.C[2] / KUKA_POSITION_SCALE

    # Camera-to-world rotation → Euler angles in degrees
    a_rad, b_rad, c_rad = rotation_to_euler_abc(pose.R)
    a_deg = np.degrees(a_rad)
    b_deg = np.degrees(b_rad)
    c_deg = np.degrees(c_rad)

    return (x_mm, y_mm, z_mm, a_deg, b_deg, c_deg)


def target_3d_to_kuka(position_world: np.ndarray, current_pose: Pose) -> tuple:
    """
    Convert a 3D world position target to KUKA format, preserving
    the current tool orientation.

    Used for position-only tracking (orientation is held fixed).

    Parameters
    ----------
    position_world : np.ndarray, shape (3,)
        Desired world position in meters.
    current_pose : Pose
        Current camera pose (used to extract the current orientation).

    Returns
    -------
    tuple of (x_mm, y_mm, z_mm, a_deg, b_deg, c_deg)
    """
    position_world = np.asarray(position_world, dtype=np.float64)

    # Extract current Euler angles from current pose
    a_rad, b_rad, c_rad = rotation_to_euler_abc(current_pose.R)

    # Position in mm
    x_mm = position_world[0] / KUKA_POSITION_SCALE
    y_mm = position_world[1] / KUKA_POSITION_SCALE
    z_mm = position_world[2] / KUKA_POSITION_SCALE

    a_deg = np.degrees(a_rad)
    b_deg = np.degrees(b_rad)
    c_deg = np.degrees(c_rad)

    return (x_mm, y_mm, z_mm, a_deg, b_deg, c_deg)
