"""
Control layer: pose reading, motion commands, safety, and target formatting.

Orchestrates the robot interaction:
  - PoseProvider: reads raw robot data → typed Pose objects
  - TargetSender: converts math-layer targets → robot commands
  - MotionController: coordinates reading, safety, and sending
  - Safety: pure validation functions (workspace, velocity, joints)
"""

from .pose_provider import PoseProvider
from .target_sender import TargetSender
from .motion_controller import MotionController
from .safety import (
    SafetyResult,
    WorkspaceLimits,
    VelocityLimits,
    JointLimits,
    check_workspace_bounds,
    check_velocity,
    check_joint_limits,
    check_position_jump,
)

__all__ = [
    "PoseProvider",
    "TargetSender",
    "MotionController",
    "SafetyResult",
    "WorkspaceLimits",
    "VelocityLimits",
    "JointLimits",
    "check_workspace_bounds",
    "check_velocity",
    "check_joint_limits",
    "check_position_jump",
]
