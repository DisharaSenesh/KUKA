"""
Robot process: continuously reads robot poses, timestamps them,
and pushes them into a multiprocessing queue.

Produces RawRobotPose objects (X, Y, Z in mm; A, B, C in degrees; timestamp).
These are the lowest-level representation — no rotation matrices here.

Runs as a separate process via multiprocessing.Process.

Design principle: continuous streaming.
  - Read poses at the robot's natural rate (~10 Hz)
  - Timestamp immediately after read
  - Push to queue (drop if full — freshness over backlog)

Timestamp rule:
  - time.perf_counter() is called immediately after the pose read
  - This is the software timestamp, not hardware motion time
"""

from __future__ import annotations

import time
import traceback
from multiprocessing import Queue, Event

import numpy as np

from .process_types import RawRobotPose


def robot_process_loop(
    pose_queue: Queue,
    stop_event: Event,
    pose_interval_s: float = 0.1,
    pose_simulator=None,
):
    """
    Entry point for the robot process.

    Continuously reads (or simulates) robot TCP poses, timestamps them,
    creates RawRobotPose objects, and pushes them to the pose queue.

    Parameters
    ----------
    pose_queue : Queue
        Multiprocessing queue for pushing RawRobotPose objects downstream.
    stop_event : Event
        Set externally to signal graceful shutdown.
    pose_interval_s : float
        Nominal interval between pose reads (for simulation rate control).
    pose_simulator : callable or None
        If provided, called as pose_simulator(t) → (X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg).
        Used for testing without hardware.
        Each value is in robot-native units: mm for position, degrees for orientation.
    """
    print(f"[RobotProcess] Started. interval={pose_interval_s*1000:.0f}ms")

    pose_count = 0
    last_read_time = time.perf_counter()

    while not stop_event.is_set():
        # ---- Read robot pose (real or simulated) ----
        t_read = time.perf_counter()  # timestamp immediately after read

        if pose_simulator is not None:
            X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg = pose_simulator(t_read)
            raw_pose = RawRobotPose(
                X_mm=float(X_mm),
                Y_mm=float(Y_mm),
                Z_mm=float(Z_mm),
                A_deg=float(A_deg),
                B_deg=float(B_deg),
                C_deg=float(C_deg),
                timestamp=t_read,
            )
        else:
            # Real hardware would read here and create RawRobotPose
            time.sleep(0.1)
            continue

        # ---- Push to queue (non-blocking, drop if full) ----
        try:
            pose_queue.put_nowait(raw_pose)
        except Exception:
            # Queue full — freshness over backlog
            pass

        pose_count += 1

        # ---- Rate control (simulation) ----
        if pose_simulator is not None:
            elapsed = time.perf_counter() - last_read_time
            sleep_time = pose_interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_read_time = time.perf_counter()

    print(f"[RobotProcess] Stopped. {pose_count} poses streamed.")


def robot_process_entry(
    pose_queue: Queue,
    stop_event: Event,
    **kwargs,
):
    """
    Wrapper entry point with exception handling for multiprocessing.Process.

    Parameters
    ----------
    pose_queue : Queue
        Pose output queue.
    stop_event : Event
        Shutdown signal.
    **kwargs
        Passed through to robot_process_loop.
    """
    try:
        robot_process_loop(pose_queue, stop_event, **kwargs)
    except Exception:
        traceback.print_exc()
