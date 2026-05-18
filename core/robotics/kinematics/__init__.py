"""
Kinematics: coordinate frames, transforms, and tool calibration.

Provides:
  - Named frame definitions and kinematic chain
  - KUKA ABC ↔ rotation matrix conversions
  - Camera pose computation from robot readings
  - TCP (Tool Center Point) configuration
"""

from .frames import Frame, FrameName, FRAME_WORLD, FRAME_BASE, FRAME_FLANGE, FRAME_TCP, FRAME_CAMERA
from .transforms import (
    euler_abc_to_rotation,
    rotation_to_euler_abc,
    kuka_pose_to_rotation_translation,
    compute_camera_pose,
    kuka_target_from_pose,
    target_3d_to_kuka,
    KUKA_POSITION_SCALE,
)
from .tcp import TCPConfig

__all__ = [
    "Frame",
    "FrameName",
    "FRAME_WORLD",
    "FRAME_BASE",
    "FRAME_FLANGE",
    "FRAME_TCP",
    "FRAME_CAMERA",
    "euler_abc_to_rotation",
    "rotation_to_euler_abc",
    "kuka_pose_to_rotation_translation",
    "compute_camera_pose",
    "kuka_target_from_pose",
    "target_3d_to_kuka",
    "KUKA_POSITION_SCALE",
    "TCPConfig",
]
