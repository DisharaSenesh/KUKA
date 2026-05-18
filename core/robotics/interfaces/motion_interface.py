"""
Motion interface: abstract protocol for high-level motion control.

Defines the expected API for motion controllers.
The MotionController conforms to this interface.

This is the API that tracking/prediction/pipeline layers should
program against — not against any concrete robot implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from core.types.pose import Pose


class MotionInterface(ABC):
    """
    Abstract interface for high-level robot motion control.

    This is the contract that the trajectory tracking system uses
    to command the robot. Concrete implementations handle the
    robot-specific details internally.
    """

    @abstractmethod
    def read_camera_pose(self, timestamp: Optional[float] = None) -> Optional[Pose]:
        """
        Read the current camera pose in world frame.

        Parameters
        ----------
        timestamp : float or None
            Timestamp to attach.

        Returns
        -------
        Pose or None
            Camera pose ready for geometry/tracking use.
        """
        ...

    @abstractmethod
    def move_to_position(
        self,
        position_world: np.ndarray,
        dt: float = 0.1,
    ) -> object:
        """
        Command the robot to a world-frame 3D position.

        Parameters
        ----------
        position_world : np.ndarray, shape (3,)
            Target position in meters.
        dt : float
            Expected time to target.

        Returns
        -------
        SafetyResult
            Result of the command (accepted/rejected with reason).
        """
        ...

    @abstractmethod
    def move_to_pose(self, pose: Pose) -> object:
        """
        Command the robot to a full 6-DOF pose.

        Parameters
        ----------
        pose : Pose
            Target camera pose.

        Returns
        -------
        SafetyResult
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the robot is operational."""
        ...
