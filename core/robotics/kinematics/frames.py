"""
Coordinate frame definitions for the KUKA robot system.

Defines the named frames used throughout the robotics layer.
Explicit frame conventions prevent confusion about which coordinate
system a quantity is expressed in.

Frame hierarchy:

    World "world"                          (fixed reference, may be offset from robot base)
        │
        └── Robot Base "base"              (robot mounting point)
                │
                └── Robot Flange "flange"  (end of arm, before tool)
                        │
                        └── TCP "tcp"      (Tool Center Point)
                                │
                                └── Camera "camera"    (monocular camera)

Transform conventions:
  - 'A_from_B' means transform that maps points from frame B to frame A
  - X_A = T_A_from_B @ X_B   (where T is a 4x4 homogeneous matrix or Rotation + translation)
"""

from dataclasses import dataclass
from enum import Enum


class FrameName(Enum):
    """
    Named coordinate frames in the robot system.

    WORLD  — fixed world reference frame (ground truth for reconstruction)
    BASE   — robot base frame (origin at robot mounting point)
    FLANGE — robot end-effector flange (before tool mounting)
    TCP    — Tool Center Point (effective tool tip)
    CAMERA — camera optical frame (Z forward, X right, Y down)
    """
    WORLD = "world"
    BASE = "base"
    FLANGE = "flange"
    TCP = "tcp"
    CAMERA = "camera"


@dataclass(frozen=True)
class Frame:
    """
    A named coordinate frame with a known parent in the kinematic tree.

    Attributes
    ----------
    name : FrameName
        Unique identifier for this frame.
    parent : FrameName or None
        Parent frame in the kinematic chain. None for root frames.
    description : str
        Human-readable description of this frame.
    """
    name: FrameName
    parent: "FrameName | None"
    description: str


# Pre-defined frame hierarchy for the KUKA system
FRAME_WORLD = Frame(
    name=FrameName.WORLD,
    parent=None,
    description="Fixed world reference frame. May be offset from robot base by a static transform."
)

FRAME_BASE = Frame(
    name=FrameName.BASE,
    parent=FrameName.WORLD,
    description="Robot base frame at the mounting point. World frame by default (identity offset)."
)

FRAME_FLANGE = Frame(
    name=FrameName.FLANGE,
    parent=FrameName.BASE,
    description="Robot end-effector flange. Pose is read from $POS_ACT on KUKA."
)

FRAME_TCP = Frame(
    name=FrameName.TCP,
    parent=FrameName.FLANGE,
    description="Tool Center Point. Offset from flange by the mounted tool geometry."
)

FRAME_CAMERA = Frame(
    name=FrameName.CAMERA,
    parent=FrameName.TCP,
    description="Monocular camera optical frame. Z forward along optical axis, X right, Y down."
)

# Ordered kinematic chain from root to leaf
KINEMATIC_CHAIN = [
    FRAME_WORLD,
    FRAME_BASE,
    FRAME_FLANGE,
    FRAME_TCP,
    FRAME_CAMERA,
]
