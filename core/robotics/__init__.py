"""
Robotics layer: clean abstractions above the low-level KUKA driver.

Architecture:

  robotics/
  ├── drivers/         # KukaDriver — wraps existing KUKAControl
  ├── kinematics/      # Frame conventions, ABC transforms, TCP config
  ├── control/         # PoseProvider, MotionController, TargetSender, Safety
  ├── synchronization/ # RobotClock, CommandBuffer
  └── interfaces/      # Abstract protocols (RobotInterface, MotionInterface)

Dependencies:
  robotics → types, geometry, numpy
  robotics does NOT depend on optimization, constraints, tracking, prediction.

Usage:

  from core.robotics import MotionController, KukaDriver, PoseProvider, TCPConfig

  # Create the driver
  driver = KukaDriver(ip="172.31.1.147", port=7000)
  driver.connect()

  # Set up the tool calibration
  tcp = TCPConfig.from_camera_offset(camera_offset_mm=(50, 0, 100))

  # Build the control stack
  pose_provider = PoseProvider(driver=driver, tcp_config=tcp)
  target_sender = TargetSender(driver=driver)
  controller = MotionController(
      pose_provider=pose_provider,
      target_sender=target_sender,
  )

  # Read pose → feed to tracker
  camera_pose = controller.read_camera_pose()

  # Send tracker output → robot
  controller.move_to_position(predicted_position)
"""

from .drivers.kuka_driver import (
    KukaDriver,
    RobotTCPPose,
    RobotJointAngles,
    RobotTarget,
)

from .kinematics import (
    Frame,
    FrameName,
    FRAME_WORLD,
    FRAME_BASE,
    FRAME_FLANGE,
    FRAME_TCP,
    FRAME_CAMERA,
    euler_abc_to_rotation,
    rotation_to_euler_abc,
    compute_camera_pose,
    kuka_target_from_pose,
    target_3d_to_kuka,
    KUKA_POSITION_SCALE,
    TCPConfig,
)

from .control import (
    PoseProvider,
    TargetSender,
    MotionController,
    SafetyResult,
    WorkspaceLimits,
    VelocityLimits,
    JointLimits,
    check_workspace_bounds,
    check_velocity,
    check_joint_limits,
    check_position_jump,
)

from .synchronization import (
    RobotClock,
    wall_time_now,
    TargetCommand,
    CommandBuffer,
)

from .interfaces import (
    RobotInterface,
    MotionInterface,
)

__all__ = [
    # Driver
    "KukaDriver",
    "RobotTCPPose",
    "RobotJointAngles",
    "RobotTarget",
    # Kinematics
    "Frame",
    "FrameName",
    "FRAME_WORLD",
    "FRAME_BASE",
    "FRAME_FLANGE",
    "FRAME_TCP",
    "FRAME_CAMERA",
    "euler_abc_to_rotation",
    "rotation_to_euler_abc",
    "compute_camera_pose",
    "kuka_target_from_pose",
    "target_3d_to_kuka",
    "KUKA_POSITION_SCALE",
    "TCPConfig",
    # Control
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
    # Synchronization
    "RobotClock",
    "wall_time_now",
    "TargetCommand",
    "CommandBuffer",
    # Interfaces
    "RobotInterface",
    "MotionInterface",
]
