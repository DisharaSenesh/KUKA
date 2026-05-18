"""
Camera process: continuously acquires frames, timestamps detections,
and pushes them into a multiprocessing queue.

Runs as a separate process via multiprocessing.Process.

Design principle: freshness over completeness.
  - Always drain the latest frame
  - Drop old detections if the queue is full
  - Never block waiting for downstream consumers

Timestamp rule:
  - time.perf_counter() is called immediately after frame acquisition
  - This timestamp represents when the frame was available to software
"""

from __future__ import annotations

import time
import traceback
from multiprocessing import Queue, Event
from typing import Optional

import numpy as np

from .process_types import Detection


def camera_process_loop(
    detection_queue: Queue,
    stop_event: Event,
    fx: float = 600.0,
    fy: float = 600.0,
    cx: float = 320.0,
    cy: float = 240.0,
    frame_interval_s: float = 0.033,
    target_simulator=None,
    pixel_noise_std: float = 0.0,
):
    """
    Entry point for the camera process.

    Continuously acquires (or simulates) frames, timestamps them,
    creates Detection objects, and pushes them to the detection queue.

    Parameters
    ----------
    detection_queue : Queue
        Multiprocessing queue for pushing Detection objects downstream.
    stop_event : Event
        Set externally to signal graceful shutdown.
    fx, fy, cx, cy : float
        Camera intrinsic parameters.
    frame_interval_s : float
        Nominal time between frames (for simulation rate control).
    target_simulator : callable or None
        If provided, called as target_simulator(t) to produce (u, v) at time t.
        If None, the process waits for frames from a real camera source
        (not implemented in this skeleton — override in subclass or configure).
    pixel_noise_std : float
        Standard deviation of Gaussian noise added to pixel observations.
    """
    print(f"[CameraProcess] Started. interval={frame_interval_s*1000:.1f}ms")

    frame_id = 0
    last_frame_time = time.perf_counter()

    while not stop_event.is_set():
        # ---- Acquire frame (real or simulated) ----
        t_frame = time.perf_counter()  # timestamp immediately

        if target_simulator is not None:
            u, v = target_simulator(t_frame)
            if pixel_noise_std > 0:
                u += np.random.normal(0, pixel_noise_std)
                v += np.random.normal(0, pixel_noise_std)
        else:
            # Real camera: read frame from hardware here
            # For now, just skip cycle
            u, v = 0.0, 0.0

        # ---- Create Detection ----
        detection = Detection(
            u=float(u),
            v=float(v),
            t=t_frame,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            frame_id=frame_id,
        )

        # ---- Push to queue (non-blocking, drop if full) ----
        # Freshness rule: if queue is full, the oldest item is already
        # behind; dropping the new one is equivalent, but we prefer not
        # to block. We use put_nowait and catch the Full exception.
        try:
            detection_queue.put_nowait(detection)
        except Exception:
            # Queue full — drop this detection (freshness over completeness)
            pass

        frame_id += 1

        # ---- Rate control (simulation only) ----
        if target_simulator is not None:
            elapsed = time.perf_counter() - last_frame_time
            sleep_time = frame_interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.perf_counter()

    print(f"[CameraProcess] Stopped. {frame_id} frames acquired.")


def camera_process_entry(
    detection_queue: Queue,
    stop_event: Event,
    **kwargs,
):
    """
    Wrapper entry point with exception handling for multiprocessing.Process.

    Parameters
    ----------
    detection_queue : Queue
        Detection output queue.
    stop_event : Event
        Shutdown signal.
    **kwargs
        Passed through to camera_process_loop.
    """
    try:
        camera_process_loop(detection_queue, stop_event, **kwargs)
    except Exception:
        traceback.print_exc()
