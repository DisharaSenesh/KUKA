"""
Processes layer: asynchronous sensor fusion infrastructure.

Provides the multiprocessing pipeline that connects:
  Camera Process  →  Detection queue  →\
                                         Synchronization Thread → sync_queue → tracking / CSV
  Robot Process   →  Pose queue       →/

Architecture:
  - Camera and Robot run as separate multiprocessing.Process instances
  - Synchronization runs as a lightweight threading.Thread
  - All timestamps use time.perf_counter() (monotonic, high-resolution)
  - The canonical SynchronizedMeasurement comes from core.synchronization
  - CSV recording via core.recording.SynchronizedWriter

Usage:

    from core.processes import ProcessManager, ProcessManagerConfig

    config = ProcessManagerConfig(
        frame_interval_s=0.033,
        pose_interval_s=0.1,
        sync_tolerance_s=0.05,
        csv_filepath="recording.csv",  # optional
    )
    manager = ProcessManager(config=config)
    manager.start(target_simulator=sim_fn, pose_simulator=pose_fn)
    time.sleep(2.0)
    manager.drain_to_csv()
    manager.stop()
"""

from .process_types import (
    Detection,
    RawRobotPose,
    SynchronizedMeasurement,
    SyncDiagnostics,
)

from .camera_process import (
    camera_process_loop,
    camera_process_entry,
)

from .robot_process import (
    robot_process_loop,
    robot_process_entry,
)

from .synchronization_thread import (
    synchronization_thread_loop,
    synchronization_thread_entry,
)

from .process_manager import (
    ProcessManagerConfig,
    ProcessManager,
)

__all__ = [
    # Types
    "Detection",
    "RawRobotPose",
    "SynchronizedMeasurement",
    "SyncDiagnostics",
    # Camera
    "camera_process_loop",
    "camera_process_entry",
    # Robot
    "robot_process_loop",
    "robot_process_entry",
    # Sync
    "synchronization_thread_loop",
    "synchronization_thread_entry",
    # Manager
    "ProcessManagerConfig",
    "ProcessManager",
]
