"""
Output layer: standardized object pose estimates for downstream consumers.

This module is the final boundary between the perception/tracking pipeline
and external robotics modules. Every consumer receives the same
ObjectPoseEstimate type regardless of tracking mode (online/offline).

Provides:
  - ObjectPoseEstimate: canonical output data type
  - RobotPose6D: 6-DOF pose in robot-native units (mm, degrees)
  - TrackingState: enum for estimate status
  - OutputBuilder: convenience construction from pipeline objects
"""

from .object_pose_output import (
    TrackingState,
    RobotPose6D,
    ObjectPoseEstimate,
    OutputBuilder,
)

__all__ = [
    "TrackingState",
    "RobotPose6D",
    "ObjectPoseEstimate",
    "OutputBuilder",
]
