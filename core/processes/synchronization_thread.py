"""
Synchronization thread: temporal alignment of asynchronous sensor streams.

Runs as a lightweight thread (threading.Thread), NOT a process.
Consumes from two multiprocessing queues:
  - detection_queue (from camera process)
  - pose_queue (from robot process)

And produces:
  - SynchronizedMeasurement objects (the canonical data type)

Uses the Synchronizer from core.synchronization for the actual
temporal matching logic. The thread handles only queue I/O and
lifecycle management.

All timestamps: time.perf_counter() (monotonic, no wall clock).
"""

from __future__ import annotations

import time
import traceback
from multiprocessing import Queue, Event

from .process_types import Detection, RawRobotPose, SynchronizedMeasurement
from core.synchronization.synchronizer import Synchronizer


def synchronization_thread_loop(
    detection_queue: Queue,
    pose_queue: Queue,
    sync_queue: Queue,
    stop_event: Event,
    sync_tolerance_s: float = 0.05,
    use_interpolation: bool = False,
    diagnostics_interval_s: float = 5.0,
):
    """
    Main loop for the synchronization thread.

    Consumes detections and raw robot poses from their respective queues,
    feeds poses into a Synchronizer, aligns each detection, and produces
    canonical SynchronizedMeasurement objects.

    Parameters
    ----------
    detection_queue : Queue
        Incoming Detection objects from the camera process.
    pose_queue : Queue
        Incoming RawRobotPose objects from the robot process.
    sync_queue : Queue
        Output queue for SynchronizedMeasurement objects.
    stop_event : Event
        Set externally to signal graceful shutdown.
    sync_tolerance_s : float
        Maximum acceptable sync error (seconds).
    use_interpolation : bool
        If True, use linear translation interpolation.
    diagnostics_interval_s : float
        How often to print diagnostics (seconds). Set to 0 to suppress.
    """
    print(f"[SyncThread] Started. tolerance={sync_tolerance_s*1000:.1f}ms "
          f"interp={use_interpolation}")

    # Build the synchronizer
    sync = Synchronizer(
        use_interpolation=use_interpolation,
        sync_tolerance_s=sync_tolerance_s,
    )

    last_diag_time = time.perf_counter()

    while not stop_event.is_set():
        # ---- Drain pose queue first (freshness priority) ----
        pose_drained = 0
        while not pose_queue.empty():
            try:
                raw = pose_queue.get_nowait()
                if isinstance(raw, RawRobotPose):
                    sync.accept_pose(
                        X_mm=raw.X_mm,
                        Y_mm=raw.Y_mm,
                        Z_mm=raw.Z_mm,
                        A_deg=raw.A_deg,
                        B_deg=raw.B_deg,
                        C_deg=raw.C_deg,
                        timestamp=raw.timestamp,
                    )
                pose_drained += 1
            except Exception:
                break

        # ---- Drain detection queue ----
        detection_drained = 0
        while not detection_queue.empty():
            try:
                det = detection_queue.get_nowait()
                if not isinstance(det, Detection):
                    continue
                detection_drained += 1

                # Synchronize the detection
                sm = sync.synchronize(
                    frame_id=det.frame_id,
                    u=det.u, v=det.v,
                    t_frame=det.t,
                    fx=det.fx, fy=det.fy,
                    cx=det.cx, cy=det.cy,
                )

                # Push to output queue (non-blocking, drop if full)
                try:
                    sync_queue.put_nowait(sm)
                except Exception:
                    pass

            except Exception:
                break

        # ---- Print diagnostics periodically ----
        now = time.perf_counter()
        if diagnostics_interval_s > 0 and (now - last_diag_time) >= diagnostics_interval_s:
            diag = sync.diagnostics
            print(
                f"[SyncThread] det={diag.total_detections} "
                f"synced={diag.total_synchronized} "
                f"dropped={diag.total_dropped} "
                f"buf={len(sync.pose_buffer)} "
                f"mean_err={diag.mean_error_s()*1000:.2f}ms "
                f"max_err={diag.max_error_s()*1000:.2f}ms"
            )
            last_diag_time = now

        # ---- Avoid busy-waiting ----
        if pose_drained == 0 and detection_drained == 0:
            time.sleep(0.001)

    # Final drain of remaining items
    while not pose_queue.empty():
        try:
            raw = pose_queue.get_nowait()
            if isinstance(raw, RawRobotPose):
                sync.accept_pose(
                    X_mm=raw.X_mm, Y_mm=raw.Y_mm, Z_mm=raw.Z_mm,
                    A_deg=raw.A_deg, B_deg=raw.B_deg, C_deg=raw.C_deg,
                    timestamp=raw.timestamp,
                )
        except Exception:
            break

    while not detection_queue.empty():
        try:
            det = detection_queue.get_nowait()
            if isinstance(det, Detection):
                sm = sync.synchronize(
                    frame_id=det.frame_id, u=det.u, v=det.v, t_frame=det.t,
                    fx=det.fx, fy=det.fy, cx=det.cx, cy=det.cy,
                )
                try:
                    sync_queue.put_nowait(sm)
                except Exception:
                    pass
        except Exception:
            break

    diag = sync.diagnostics
    print(f"[SyncThread] Stopped. synced={diag.total_synchronized} "
          f"mean_err={diag.mean_error_s()*1000:.2f}ms")


def synchronization_thread_entry(
    detection_queue: Queue,
    pose_queue: Queue,
    sync_queue: Queue,
    stop_event: Event,
    **kwargs,
):
    """
    Wrapper entry point for threading.Thread.

    Parameters
    ----------
    detection_queue, pose_queue, sync_queue : Queue
        Input and output queues.
    stop_event : Event
        Shutdown signal.
    **kwargs
        Passed to synchronization_thread_loop.
    """
    try:
        synchronization_thread_loop(
            detection_queue, pose_queue, sync_queue, stop_event, **kwargs
        )
    except Exception:
        traceback.print_exc()
