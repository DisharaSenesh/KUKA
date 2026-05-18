"""
Robot safety checks: workspace boundaries, velocity limits, joint limits.

All safety functions are pure: they evaluate a candidate command or state
and return a boolean with an explanatory message. The motion controller
calls these checks before sending any command to the robot.

Safety checks do NOT modify commands. They only evaluate and report.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SafetyResult:
    """
    Result of a safety evaluation.

    Attributes
    ----------
    safe : bool
        True if the check passed.
    message : str
        Explanation of pass or failure.
    """

    safe: bool
    message: str


@dataclass
class WorkspaceLimits:
    """
    Axis-aligned workspace bounding box in world frame.

    All values are in meters. Targets outside this box are rejected.

    Attributes
    ----------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Workspace bounds in meters.
    """

    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0
    z_min: float = 0.0
    z_max: float = 2.0


@dataclass
class VelocityLimits:
    """
    Maximum allowed velocities for robot motion.

    Values are in meters/second (translational) and degrees/second (rotational).
    These are soft limits applied before sending to the controller;
    the hardware may have its own stricter limits.

    Attributes
    ----------
    max_linear_speed : float
        Maximum translational speed (m/s).
    max_angular_speed_deg : float
        Maximum rotational speed (deg/s).
    """

    max_linear_speed: float = 2.0
    max_angular_speed_deg: float = 180.0


@dataclass
class JointLimits:
    """
    Joint angle limits for the KUKA robot.

    Values are in degrees. These are approximate and should be
    verified against the specific robot model's datasheet.

    Attributes
    ----------
    min_degrees, max_degrees : list of float
        Min/max for each joint [j1, j2, j3, j4, j5, j6].
    """

    min_degrees: tuple = (-185, -120, -120, -350, -125, -350)
    max_degrees: tuple = (185, 120, 120, 350, 125, 350)


def check_workspace_bounds(
    position_world: np.ndarray,
    limits: WorkspaceLimits,
) -> SafetyResult:
    """
    Check whether a world-frame position is within the workspace.

    Parameters
    ----------
    position_world : np.ndarray, shape (3,)
        Target position in meters (world frame).
    limits : WorkspaceLimits
        Workspace boundary definition.

    Returns
    -------
    SafetyResult
    """
    pos = np.asarray(position_world, dtype=np.float64)
    x, y, z = pos[0], pos[1], pos[2]

    if x < limits.x_min or x > limits.x_max:
        return SafetyResult(
            safe=False,
            message=f"X={x:.3f} out of bounds [{limits.x_min}, {limits.x_max}]"
        )
    if y < limits.y_min or y > limits.y_max:
        return SafetyResult(
            safe=False,
            message=f"Y={y:.3f} out of bounds [{limits.y_min}, {limits.y_max}]"
        )
    if z < limits.z_min or z > limits.z_max:
        return SafetyResult(
            safe=False,
            message=f"Z={z:.3f} out of bounds [{limits.z_min}, {limits.z_max}]"
        )

    return SafetyResult(safe=True, message="Position within workspace bounds.")


def check_velocity(
    target_position: np.ndarray,
    current_position: np.ndarray,
    dt: float,
    limits: VelocityLimits,
) -> SafetyResult:
    """
    Check whether the implied velocity to reach a target is safe.

    v_implied = ‖target - current‖ / dt

    Parameters
    ----------
    target_position : np.ndarray, shape (3,)
        Desired position in meters.
    current_position : np.ndarray, shape (3,)
        Current position in meters.
    dt : float
        Time interval in seconds.
    limits : VelocityLimits
        Velocity limit definition.

    Returns
    -------
    SafetyResult
    """
    if dt <= 0.0:
        return SafetyResult(safe=False, message="dt must be positive.")

    target = np.asarray(target_position, dtype=np.float64)
    current = np.asarray(current_position, dtype=np.float64)

    displacement = np.linalg.norm(target - current)
    implied_speed = displacement / dt

    if implied_speed > limits.max_linear_speed:
        return SafetyResult(
            safe=False,
            message=(
                f"Implied speed {implied_speed:.2f} m/s exceeds "
                f"limit {limits.max_linear_speed:.2f} m/s"
            ),
        )

    return SafetyResult(
        safe=True,
        message=f"Velocity {implied_speed:.2f} m/s within limit."
    )


def check_joint_limits(
    joint_angles: list,
    limits: JointLimits,
) -> SafetyResult:
    """
    Check whether joint angles are within hardware limits.

    Parameters
    ----------
    joint_angles : list of float
        Joint angles in degrees [j1..j6].
    limits : JointLimits
        Joint limit definition.

    Returns
    -------
    SafetyResult
    """
    for i, (angle, min_val, max_val) in enumerate(
        zip(joint_angles, limits.min_degrees, limits.max_degrees)
    ):
        if angle < min_val or angle > max_val:
            return SafetyResult(
                safe=False,
                message=(
                    f"Joint {i+1} angle {angle:.1f}° out of bounds "
                    f"[{min_val}, {max_val}]"
                ),
            )

    return SafetyResult(safe=True, message="All joint angles within limits.")


def check_position_jump(
    target_position: np.ndarray,
    current_position: np.ndarray,
    max_jump: float = 1.0,
) -> SafetyResult:
    """
    Guard against large instantaneous position jumps.

    Rejects targets that differ from the current position by more
    than max_jump meters. This catches software bugs, coordinate
    frame errors, or sensor glitches before they reach the robot.

    Parameters
    ----------
    target_position : np.ndarray, shape (3,)
        Desired position.
    current_position : np.ndarray, shape (3,)
        Current position.
    max_jump : float
        Maximum allowed displacement in meters.

    Returns
    -------
    SafetyResult
    """
    target = np.asarray(target_position, dtype=np.float64)
    current = np.asarray(current_position, dtype=np.float64)

    jump = np.linalg.norm(target - current)

    if jump > max_jump:
        return SafetyResult(
            safe=False,
            message=(
                f"Position jump {jump:.3f} m exceeds safety limit "
                f"{max_jump:.3f} m. Command rejected."
            ),
        )

    return SafetyResult(
        safe=True,
        message=f"Position jump {jump:.3f} m within limit."
    )
