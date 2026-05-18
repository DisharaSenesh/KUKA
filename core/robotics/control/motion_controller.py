"""
MotionController: primary robotics interface for the trajectory system.

Orchestrates:
  1. Reading camera poses from the robot (via PoseProvider)
  2. Validating motion commands through safety checks
  3. Sending targets to the robot (via TargetSender)

The MotionController is the ONLY entry point that tracking/prediction/
pipeline layers should use to interact with the robot.

High-level data flow:

  Trajectory Prediction → 3D target (world frame)
        ↓
  MotionController
    ├── Safety checks (workspace, velocity, jump guards)
    ├── Coordinate conversion (world → robot-native)
    └── TargetSender → KukaDriver → KUKA Robot
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.types.pose import Pose
from .pose_provider import PoseProvider
from .target_sender import TargetSender
from .safety import (
    SafetyResult,
    WorkspaceLimits,
    VelocityLimits,
    check_workspace_bounds,
    check_velocity,
    check_position_jump,
)


@dataclass
class MotionController:
    """
    High-level motion controller for the robot.

    Combines pose reading, safety validation, and target sending
    into a single coordination point.

    Attributes
    ----------
    pose_provider : PoseProvider
        Reads camera poses from the robot.
    target_sender : TargetSender
        Sends formatted targets to the robot.
    workspace_limits : WorkspaceLimits
        Workspace boundary definition for safety checks.
    velocity_limits : VelocityLimits
        Velocity limit definition for safety checks.
    max_position_jump : float
        Maximum allowed instantaneous position jump (meters).
    safety_enabled : bool
        If False, skip safety checks (use only in simulation/testing).
    """

    pose_provider: PoseProvider = None
    target_sender: TargetSender = None
    workspace_limits: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    velocity_limits: VelocityLimits = field(default_factory=VelocityLimits)
    max_position_jump: float = 0.5
    safety_enabled: bool = True

    # ---- Pose reading (delegates to PoseProvider) ----

    def read_camera_pose(self, timestamp: Optional[float] = None) -> Optional[Pose]:
        """
        Read the current camera pose from the robot.

        Parameters
        ----------
        timestamp : float or None
            Timestamp for the pose.

        Returns
        -------
        Pose or None
        """
        return self.pose_provider.read_camera_pose(timestamp)

    def get_last_pose(self) -> Optional[Pose]:
        """Return the most recently read camera pose."""
        return self.pose_provider.get_last_pose()

    # ---- Motion commands with safety checks ----

    def move_to_position(
        self,
        position_world: np.ndarray,
        dt: float = 0.1,
    ) -> SafetyResult:
        """
        Command the robot to move to a world-frame 3D position.

        Runs safety checks before sending:
          1. Workspace bounds check
          2. Position jump guard
          3. Velocity check

        Orientation is preserved from the last known pose.

        Parameters
        ----------
        position_world : np.ndarray, shape (3,)
            Target position in meters (world frame).
        dt : float
            Expected time to reach the target (for velocity check).

        Returns
        -------
        SafetyResult
            Indicates whether the command was accepted and sent.
        """
        position_world = np.asarray(position_world, dtype=np.float64)

        if self.safety_enabled:
            # 1. Workspace bounds
            result = check_workspace_bounds(position_world, self.workspace_limits)
            if not result.safe:
                return result

            # 2. Position jump guard
            current_pose = self.get_last_pose()
            if current_pose is not None:
                result = check_position_jump(
                    position_world,
                    current_pose.C,
                    self.max_position_jump,
                )
                if not result.safe:
                    return result

                # 3. Velocity check
                result = check_velocity(
                    position_world,
                    current_pose.C,
                    dt,
                    self.velocity_limits,
                )
                if not result.safe:
                    return result

        # Get current pose for orientation
        current_pose = self.get_last_pose()
        if current_pose is None:
            # Try to read it now
            current_pose = self.read_camera_pose()
            if current_pose is None:
                return SafetyResult(
                    safe=False,
                    message="Cannot send target: no current pose available.",
                )

        # Send the position-only target
        success = self.target_sender.send_position_target(
            position_world, current_pose
        )

        if success:
            return SafetyResult(safe=True, message="Position target sent.")
        else:
            return SafetyResult(safe=False, message="Failed to send target.")

    def move_to_pose(self, pose: Pose) -> SafetyResult:
        """
        Command the robot to move to a full 6-DOF pose target.

        Applies workspace and jump checks before sending.

        Parameters
        ----------
        pose : Pose
            Target camera pose in world frame.

        Returns
        -------
        SafetyResult
        """
        if self.safety_enabled:
            result = check_workspace_bounds(pose.C, self.workspace_limits)
            if not result.safe:
                return result

            current_pose = self.get_last_pose()
            if current_pose is not None:
                result = check_position_jump(
                    pose.C, current_pose.C, self.max_position_jump
                )
                if not result.safe:
                    return result

        success = self.target_sender.send_pose_target(pose)

        if success:
            return SafetyResult(safe=True, message="Pose target sent.")
        else:
            return SafetyResult(safe=False, message="Failed to send target.")

    # ---- Utility ----

    def is_connected(self) -> bool:
        """Check if the robot driver is connected."""
        if self.pose_provider is None:
            return False
        return self.pose_provider.driver.is_connected()

    def set_speed(self, speed_percent: int) -> bool:
        """
        Set the robot speed override.

        Parameters
        ----------
        speed_percent : int
            Speed override (0-100).

        Returns
        -------
        bool
            True if set successfully.
        """
        if self.pose_provider is None:
            return False
        return self.pose_provider.driver.set_speed(speed_percent)
