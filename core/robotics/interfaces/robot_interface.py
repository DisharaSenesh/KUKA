"""
Robot interface: abstract protocol for robot communication drivers.

Defines the expected API that any robot driver must implement.
The KukaDriver conforms to this interface. This enables testing
with mock drivers and future support for different robot models.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RobotInterface(ABC):
    """
    Abstract interface for robot communication drivers.

    Any robot driver (KUKA, ABB, Fanuc, simulated) must implement
    this protocol so that the higher layers can remain robot-agnostic.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection and perform handshake."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the robot is connected and ready."""
        ...

    @abstractmethod
    def read_tcp_pose(self) -> Optional:
        """
        Read the current TCP pose.

        Returns
        -------
        RobotTCPPose or None
            Raw TCP pose in robot-native coordinates.
        """
        ...

    @abstractmethod
    def read_joint_angles(self) -> Optional:
        """
        Read current joint angles.

        Returns
        -------
        RobotJointAngles or None
            Joint angles in degrees.
        """
        ...

    @abstractmethod
    def write_target(self, target) -> bool:
        """
        Send a motion target to the robot.

        Parameters
        ----------
        target : RobotTarget
            Target in robot-native format.

        Returns
        -------
        bool
            True if successful.
        """
        ...
