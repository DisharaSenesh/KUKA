"""
Interfaces: abstract protocols for robot communication and motion control.

Defines the expected API contracts. Concrete implementations in
drivers/ and control/ conform to these interfaces.
"""

from .robot_interface import RobotInterface
from .motion_interface import MotionInterface

__all__ = [
    "RobotInterface",
    "MotionInterface",
]
