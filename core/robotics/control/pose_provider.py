"""
PoseProvider: bridge between robot driver and the geometry layer.

Reads raw robot TCP poses from the driver, applies kinematic transforms
to produce camera Pose objects ready for the trajectory reconstruction
system.

Pipeline:
  KukaDriver → raw (x,y,z,a,b,c) → kinematics → Pose (R, C, t)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.types.pose import Pose
from ..drivers.kuka_driver import KukaDriver, RobotTCPPose
from ..kinematics.transforms import compute_camera_pose
from ..kinematics.tcp import TCPConfig


@dataclass
class PoseProvider:
    """
    Provides geometry-ready camera Pose objects from the robot.

    Reads the current robot TCP pose, applies the kinematic chain
    (flange → TCP → camera), timestamps the result, and returns
    a typed Pose for use in trajectory reconstruction.

    Attributes
    ----------
    driver : KukaDriver
        The wrapped robot driver for reading raw poses.
    tcp_config : TCPConfig
        Static tool and camera calibration.
    last_pose : Optional[Pose]
        Most recently read camera pose (for quick access).
    """

    driver: KukaDriver
    tcp_config: TCPConfig = None
    last_pose: Optional[Pose] = None

    def __post_init__(self):
        if self.tcp_config is None:
            self.tcp_config = TCPConfig()

    def read_camera_pose(self, timestamp: Optional[float] = None) -> Optional[Pose]:
        """
        Read the current robot pose and convert to a camera Pose.

        Parameters
        ----------
        timestamp : float or None
            Timestamp to attach. If None, uses a default (0.0).
            In production, this should come from the robot clock.

        Returns
        -------
        Pose or None
            Camera pose in world frame, or None if the read failed.
        """
        raw = self.driver.read_tcp_pose()
        if raw is None:
            return None

        if timestamp is None:
            timestamp = 0.0

        # Compute camera pose through the kinematic chain
        camera_pose = compute_camera_pose(
            robot_x_mm=raw.x_mm,
            robot_y_mm=raw.y_mm,
            robot_z_mm=raw.z_mm,
            robot_a_deg=raw.a_deg,
            robot_b_deg=raw.b_deg,
            robot_c_deg=raw.c_deg,
            flange_to_tcp=self.tcp_config.flange_to_tcp,
            tcp_to_camera_R=self.tcp_config.tcp_to_camera_R,
            tcp_to_camera_t=self.tcp_config.tcp_to_camera_t,
            timestamp=timestamp,
        )

        self.last_pose = camera_pose
        return camera_pose

    def get_last_pose(self) -> Optional[Pose]:
        """Return the most recently read pose without re-reading the robot."""
        return self.last_pose

    def read_raw_tcp(self) -> Optional[RobotTCPPose]:
        """
        Read raw TCP data without conversion to camera frame.

        Useful for diagnostics or when the raw robot pose is needed.

        Returns
        -------
        RobotTCPPose or None
        """
        return self.driver.read_tcp_pose()
