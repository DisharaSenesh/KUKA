"""
ProcessManager: creates, starts, and shuts down the asynchronous pipeline.

Manages:
  - Camera process      (multiprocessing.Process)
  - Robot process       (multiprocessing.Process)
  - Synchronization thread (threading.Thread)
  - Communication queues (multiprocessing.Queue)
  - CSV recording (optional)

Lifecycle:
  1. create manager with configuration
  2. call start() to launch all processes and the sync thread
  3. optionally call drain_to_csv() to record synchronized data
  4. call stop() for graceful shutdown

Design: minimal, explicit, no frameworks.
"""

from __future__ import annotations

import time
import threading
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass, field
from typing import Optional, Callable, List

from .camera_process import camera_process_entry
from .robot_process import robot_process_entry
from .synchronization_thread import synchronization_thread_entry
from .process_types import SynchronizedMeasurement, RawRobotPose
from core.recording.synchronized_writer import SynchronizedWriter


@dataclass
class ProcessManagerConfig:
    """
    Configuration for the asynchronous sensor pipeline.

    Attributes
    ----------
    detection_queue_size : int
        Max size of the detection queue. Exceeding entries are dropped.
    pose_queue_size : int
        Max size of the pose queue.
    sync_queue_size : int
        Max size of the synchronized measurement output queue.
    frame_interval_s : float
        Nominal camera frame interval.
    pose_interval_s : float
        Nominal robot pose read interval.
    sync_tolerance_s : float
        Maximum acceptable sync error.
    use_interpolation : bool
        Enable linear translation interpolation.
    diagnostics_interval_s : float
        How often to print sync diagnostics (seconds). 0 = suppress.
    csv_filepath : Optional[str]
        If provided, synchronized measurements are recorded to this CSV file.
    fx, fy, cx, cy : float
        Camera intrinsics passed to the camera process.
    pixel_noise_std : float
        Noise added to simulated pixel observations.
    """

    detection_queue_size: int = 50
    pose_queue_size: int = 200
    sync_queue_size: int = 100
    frame_interval_s: float = 0.033
    pose_interval_s: float = 0.1
    sync_tolerance_s: float = 0.05
    use_interpolation: bool = False
    diagnostics_interval_s: float = 3.0
    csv_filepath: Optional[str] = None
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    pixel_noise_std: float = 0.0


@dataclass
class ProcessManager:
    """
    Orchestrator for the asynchronous sensor pipeline.

    Creates queues, processes, and the synchronization thread.
    Provides start/stop lifecycle management.

    Optional CSV recording: set csv_filepath in config to record
    all synchronized measurements to disk for offline replay.

    Usage:

        config = ProcessManagerConfig(csv_filepath="recording.csv")
        manager = ProcessManager(config=config)
        manager.start(target_simulator=..., pose_simulator=...)
        time.sleep(2.0)
        manager.drain_to_csv()  # or consume manually from sync_queue
        manager.stop()
    """

    config: ProcessManagerConfig = field(default_factory=ProcessManagerConfig)

    # Queues (created in __post_init__)
    detection_queue: Queue = field(init=False)
    pose_queue: Queue = field(init=False)
    sync_queue: Queue = field(init=False)

    # Process handles
    camera_proc: Optional[Process] = None
    robot_proc: Optional[Process] = None

    # Thread handle
    sync_thread: Optional[threading.Thread] = None

    # CSV writer
    _csv_writer: Optional[SynchronizedWriter] = None

    # Shutdown coordination
    _stop_event: Event = field(init=False)

    def __post_init__(self):
        """Create communication queues and the shared stop event."""
        self.detection_queue = Queue(maxsize=self.config.detection_queue_size)
        self.pose_queue = Queue(maxsize=self.config.pose_queue_size)
        self.sync_queue = Queue(maxsize=self.config.sync_queue_size)
        self._stop_event = Event()

    def start(
        self,
        target_simulator: Optional[Callable] = None,
        pose_simulator: Optional[Callable] = None,
    ) -> None:
        """
        Launch all processes and the synchronization thread.

        Optionally opens a CSV writer for recording.

        Parameters
        ----------
        target_simulator : callable(t) → (u, v) or None
            Simulated target pixel trajectory for the camera process.
            Returns (u, v) pixel coordinates at time t.
        pose_simulator : callable(t) → (X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg) or None
            Simulated robot pose trajectory for the robot process.
            Returns raw robot pose in KUKA-native units at time t.
        """
        if self._stop_event.is_set():
            self._stop_event.clear()

        # Recreate queues in case of restart
        self.detection_queue = Queue(maxsize=self.config.detection_queue_size)
        self.pose_queue = Queue(maxsize=self.config.pose_queue_size)
        self.sync_queue = Queue(maxsize=self.config.sync_queue_size)

        # ---- Optional CSV recording ----
        if self.config.csv_filepath is not None:
            self._csv_writer = SynchronizedWriter(self.config.csv_filepath)
            self._csv_writer.open()

        # ---- Launch camera process ----
        self.camera_proc = Process(
            target=camera_process_entry,
            args=(self.detection_queue, self._stop_event),
            kwargs=dict(
                fx=self.config.fx,
                fy=self.config.fy,
                cx=self.config.cx,
                cy=self.config.cy,
                frame_interval_s=self.config.frame_interval_s,
                target_simulator=target_simulator,
                pixel_noise_std=self.config.pixel_noise_std,
            ),
            name="CameraProcess",
            daemon=True,
        )
        self.camera_proc.start()
        print("[ProcessManager] Camera process started.")

        # ---- Launch robot process ----
        self.robot_proc = Process(
            target=robot_process_entry,
            args=(self.pose_queue, self._stop_event),
            kwargs=dict(
                pose_interval_s=self.config.pose_interval_s,
                pose_simulator=pose_simulator,
            ),
            name="RobotProcess",
            daemon=True,
        )
        self.robot_proc.start()
        print("[ProcessManager] Robot process started.")

        # ---- Launch synchronization thread ----
        self.sync_thread = threading.Thread(
            target=synchronization_thread_entry,
            args=(
                self.detection_queue,
                self.pose_queue,
                self.sync_queue,
                self._stop_event,
            ),
            kwargs=dict(
                sync_tolerance_s=self.config.sync_tolerance_s,
                use_interpolation=self.config.use_interpolation,
                diagnostics_interval_s=self.config.diagnostics_interval_s,
            ),
            name="SyncThread",
            daemon=True,
        )
        self.sync_thread.start()
        print("[ProcessManager] Synchronization thread started.")

    def stop(self, timeout_s: float = 2.0) -> None:
        """
        Gracefully stop all processes and the sync thread.

        1. Signal stop event
        2. Wait for sync thread to finish
        3. Join camera and robot processes
        4. Close CSV writer if open
        5. Close queues

        Parameters
        ----------
        timeout_s : float
            Maximum time to wait for processes to join.
        """
        print("[ProcessManager] Stopping pipeline...")
        self._stop_event.set()

        # Wait for sync thread
        if self.sync_thread is not None and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=timeout_s)
            print("[ProcessManager] Sync thread joined.")

        # Wait for processes
        for proc, name in [
            (self.camera_proc, "Camera"),
            (self.robot_proc, "Robot"),
        ]:
            if proc is not None and proc.is_alive():
                proc.join(timeout=timeout_s)
                if proc.is_alive():
                    proc.terminate()
                print(f"[ProcessManager] {name} process joined.")

        # Close CSV writer
        if self._csv_writer is not None:
            self._csv_writer.close()
            self._csv_writer = None

        # Close queues
        for q in [self.detection_queue, self.pose_queue, self.sync_queue]:
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        print("[ProcessManager] Pipeline stopped.")

    def get_sync_measurement(
        self, timeout_s: float = 0.1
    ) -> Optional[SynchronizedMeasurement]:
        """
        Get the next synchronized measurement from the output queue.

        Non-blocking with a short timeout. Returns None if the queue
        is empty or the pipeline has stopped.

        Parameters
        ----------
        timeout_s : float
            How long to wait for data (seconds). Use 0 for non-blocking.

        Returns
        -------
        SynchronizedMeasurement or None
        """
        if self._stop_event.is_set():
            return None

        try:
            return self.sync_queue.get(timeout=timeout_s)
        except Exception:
            return None

    def drain_to_csv(self) -> int:
        """
        Drain all available synchronized measurements to the CSV file.

        Reads from the sync queue until empty, writing each measurement
        to the CSV writer. The writer must have been opened (configured
        via csv_filepath in config).

        Returns
        -------
        int
            Number of measurements written in this drain.

        Raises
        ------
        RuntimeError
            If no CSV writer was configured.
        """
        if self._csv_writer is None:
            raise RuntimeError(
                "No CSV writer configured. Set csv_filepath in ProcessManagerConfig."
            )

        written = 0
        while True:
            sm = self.get_sync_measurement(timeout_s=0.01)
            if sm is None:
                break
            self._csv_writer.write(sm)
            written += 1

        return written

    def drain_all(self) -> List[SynchronizedMeasurement]:
        """
        Drain all available synchronized measurements into a list.

        Returns
        -------
        list of SynchronizedMeasurement
        """
        measurements = []
        while True:
            sm = self.get_sync_measurement(timeout_s=0.01)
            if sm is None:
                break
            measurements.append(sm)
        return measurements

    def is_running(self) -> bool:
        """Return True if the pipeline is active."""
        camera_alive = self.camera_proc is not None and self.camera_proc.is_alive()
        robot_alive = self.robot_proc is not None and self.robot_proc.is_alive()
        return camera_alive and robot_alive and not self._stop_event.is_set()
