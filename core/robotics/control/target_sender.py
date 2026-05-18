"""
TargetSender: converts math-layer geometry into robot commands.

Takes trajectory outputs (Pose objects, 3D positions) from the
prediction/tracking layers and converts them to KUKA-native targets,
then sends them through the driver.

This is the only module that translates between:
  - Math-layer coordinates (meters, radians, Pose objects)
  - Robot-native coordinates (mm, degrees, KUKA ABC)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.types.pose import Pose
from ..drivers.kuka_driver import KukaDriver, RobotTarget
from ..kinematics.transforms import (
    kuka_target_from_pose,
    target_3d_to_kuka,
    KUKA_POSITION_SCALE,
)


@dataclass
class TargetSender:
    """
    Converts pose targets to robot commands and sends them.

    Attributes
    ----------
    driver : KukaDriver
        The robot driver for sending commands.
    last_sent_target : Optional[RobotTarget]
        Most recently sent target (for diagnostics).
    """

    driver: KukaDriver
    last_sent_target: Optional[RobotTarget] = None

    def send_pose_target(self, pose: Pose) -> bool:
        """
        Send a full 6-DOF pose target to the robot.

        Converts the math-layer Pose (R, C, t) to KUKA coordinates
        (x, y, z in mm; A, B, C in degrees) and writes to GTARGET.

        Parameters
        ----------
        pose : Pose
            Target pose in world frame.

        Returns
        -------
        bool
            True if the target was sent successfully.
        """
        x_mm, y_mm, z_mm, a_deg, b_deg, c_deg = kuka_target_from_pose(pose)

        target = RobotTarget(
            x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
            rz_deg=a_deg, ry_deg=b_deg, rx_deg=c_deg,
        )

        success = self.driver.write_target(target)
        if success:
            self.last_sent_target = target

        return success

    def send_position_target(
        self,
        position_world: np.ndarray,
        current_pose: Pose,
    ) -> bool:
        """
        Send a position-only target (3-DOF), holding orientation fixed.

        Useful for tracking where only the position needs updating
        (e.g., the object moves but the tool orientation stays constant).

        Parameters
        ----------
        position_world : np.ndarray, shape (3,)
            Desired position in meters (world frame).
        current_pose : Pose
            Current camera pose (for extracting the orientation).

        Returns
        -------
        bool
            True if sent successfully.
        """
        x_mm, y_mm, z_mm, a_deg, b_deg, c_deg = target_3d_to_kuka(
            position_world, current_pose
        )

        target = RobotTarget(
            x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
            rz_deg=a_deg, ry_deg=b_deg, rx_deg=c_deg,
        )

        success = self.driver.write_target(target)
        if success:
            self.last_sent_target = target

        return success

    def send_raw_target(self, target: RobotTarget) -> bool:
        """
        Send a pre-formatted robot target directly.

        For cases where the caller has already formatted the target
        in robot-native coordinates.

        Parameters
        ----------
        target : RobotTarget
            Target in robot-native format (mm, degrees).

        Returns
        -------
        bool
            True if sent successfully.
        """
        success = self.driver.write_target(target)
        if success:
            self.last_sent_target = target
        return success

    def send_position_mm(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """
        Send a position-only target in millimeters, preserving orientation.

        Parameters
        ----------
        x_mm, y_mm, z_mm : float
            Target position in millimeters.

        Returns
        -------
        bool
            True if sent successfully.
        """
        return self.driver.write_position_only(x_mm, y_mm, z_mm)
