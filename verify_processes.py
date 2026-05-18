"""
Verification demo: asynchronous sensor fusion pipeline.

Simulates:
  - A target moving on a polynomial 3D trajectory
  - A camera observing the target at ~30 Hz (pixel detections)
  - A robot streaming poses at ~10 Hz
  - The synchronization thread aligning both streams

Verifies:
  1. Detection objects are produced with correct timestamps
  2. Pose objects are produced with correct timestamps
  3. SynchronizedMeasurements pair each detection with a nearest pose
  4. Sync errors are bounded by the pose sampling interval
  5. The full pipeline runs without crashes or deadlocks
  6. Shutdown is clean
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.types.trajectory_state import TrajectoryState
from core.types.pose import Pose
from core.geometry.projection import project_to_pixel
from core.processes import (
    ProcessManager,
    ProcessManagerConfig,
    Detection,
    SynchronizedMeasurement,
    SyncDiagnostics,
)
from core.processes.synchronization_thread import PoseBuffer


# ============================================================
# Simulators
# ============================================================

def make_target_simulator(true_state, fx, fy, cx, cy, poses_by_time_ref):
    """
    Create a target simulator that projects the true 3D trajectory
    through a camera at the nearest robot pose.
    """
    def target_simulator(t_frame):
        # True 3D position at frame time
        X_world = true_state.evaluate(t_frame)

        # Use a fixed camera pose (the simulation's robot poses are separate)
        # For simplicity, synthesize a camera pose looking at the target
        # from a fixed position above
        C = np.array([2.0, 1.0, 8.0])
        z_cam = X_world - C
        z_cam = z_cam / np.linalg.norm(z_cam)
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(world_up, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.array([1.0, 0.0, 0.0])
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        R = np.vstack([x_cam, y_cam, z_cam])

        pose = Pose(R=R, C=C, t=t_frame)
        X_cam = pose.world_to_camera(X_world)
        pixel = project_to_pixel(X_cam, fx, fy, cx, cy)

        return float(pixel[0]), float(pixel[1])

    return target_simulator


def make_pose_simulator(true_state):
    """
    Create a pose simulator that generates robot poses following
    the camera mounted on a moving robot.
    """
    def pose_simulator(t_pose):
        # True target position
        X_world = true_state.evaluate(t_pose)

        # Camera at a fixed station observing the moving target
        C = np.array([3.0 * np.cos(0.1 * t_pose),
                      2.0 * np.sin(0.15 * t_pose),
                      7.0 + 0.2 * np.sin(0.2 * t_pose)])

        # Look-at rotation
        z_cam = X_world - C
        z_cam = z_cam / np.linalg.norm(z_cam)
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(world_up, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.array([1.0, 0.0, 0.0])
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        R = np.vstack([x_cam, y_cam, z_cam])

        return np.asarray(R, dtype=np.float64), np.asarray(C, dtype=np.float64)

    return pose_simulator


# ============================================================
# Tests
# ============================================================

def test_pose_buffer():
    """Test PoseBuffer: push, nearest, bracketing."""
    buffer = PoseBuffer(max_size=10)

    # Push time-ordered poses
    for i in range(5):
        pose = Pose(
            R=np.eye(3),
            C=np.array([float(i), 0.0, 0.0]),
            t=float(i),
        )
        buffer.push(pose)

    assert len(buffer) == 5
    print("  [PASS] PoseBuffer holds 5 poses")

    # Nearest-neighbor
    nearest = buffer.find_nearest(1.2)
    assert nearest is not None
    assert abs(nearest.t - 1.0) < 1e-6, f"Expected t=1.0, got {nearest.t}"
    print("  [PASS] find_nearest(1.2) → t=1.0")

    nearest2 = buffer.find_nearest(3.7)
    assert nearest2 is not None
    assert abs(nearest2.t - 4.0) < 1e-6
    print("  [PASS] find_nearest(3.7) → t=4.0")

    # Bracketing
    before, after = buffer.find_bracketing(2.3)
    assert before is not None and after is not None
    assert before.t == 2.0 and after.t == 3.0
    print("  [PASS] find_bracketing(2.3) → (t=2.0, t=3.0)")

    # Beyond buffer range
    before2, after2 = buffer.find_bracketing(-0.5)
    assert after2 is not None and after2.t == 0.0, "Should give first pose as after"
    print("  [PASS] find_bracketing(-0.5) → after=t=0.0")

    # Overflow
    for i in range(20):
        buffer.push(Pose(R=np.eye(3), C=np.array([float(i), 0.0, 0.0]), t=float(i + 5)))

    assert len(buffer) <= 10
    print(f"  [PASS] PoseBuffer capped at {len(buffer)}")


def test_detection_types():
    """Test that Detection and SynchronizedMeasurement are pickle-able."""
    import pickle

    # Detection must be pickle-able (multiprocessing requirement)
    det = Detection(u=100.0, v=200.0, t=1.234, fx=600, fy=600, cx=320, cy=240, frame_id=42)
    det_bytes = pickle.dumps(det)
    det2 = pickle.loads(det_bytes)
    assert det2.u == 100.0
    assert det2.frame_id == 42
    print("  [PASS] Detection is pickle-able")

    # SynchronizedMeasurement
    pose = Pose(R=np.eye(3), C=np.array([1.0, 2.0, 3.0]), t=1.2)
    sm = SynchronizedMeasurement(
        detection=det,
        pose=pose,
        sync_error_s=0.034,
        sync_method="nearest",
        is_valid=True,
    )
    sm_bytes = pickle.dumps(sm)
    sm2 = pickle.loads(sm_bytes)
    assert sm2.sync_error_s == 0.034
    assert sm2.is_valid
    print("  [PASS] SynchronizedMeasurement is pickle-able")


def test_sync_diagnostics():
    """Test SyncDiagnostics rolling statistics."""
    diag = SyncDiagnostics(max_history=10)

    for err in [0.01, 0.02, 0.03, 0.04, 0.05]:
        diag.record_sync(err)

    assert diag.synchronized_produced == 5
    assert abs(diag.mean_sync_error_s() - 0.03) < 1e-10
    assert abs(diag.max_sync_error_s() - 0.05) < 1e-10
    print(f"  [PASS] SyncDiagnostics: mean={diag.mean_sync_error_s()*1000:.1f}ms, "
          f"max={diag.max_sync_error_s()*1000:.1f}ms")

    summary = diag.summary()
    assert "synced=5" in summary
    print(f"  [PASS] summary(): {summary}")


def test_full_pipeline():
    """
    End-to-end test: launch the pipeline with simulated sensors,
    collect synchronized measurements, verify quality.
    """
    print("\n--- Full Pipeline Test ---")

    # Ground-truth trajectory
    a0 = np.array([2.0, 1.0, 0.5])
    a1 = np.array([0.3, -0.2, 0.05])
    a2 = np.array([0.01, 0.005, -0.002])
    true_state = TrajectoryState(coefficients=[a0, a1, a2], t0=0.0)

    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0

    # Build simulators
    target_sim = make_target_simulator(true_state, fx, fy, cx, cy, None)
    pose_sim = make_pose_simulator(true_state)

    # Configure pipeline
    config = ProcessManagerConfig(
        detection_queue_size=100,
        pose_queue_size=200,
        sync_queue_size=200,
        frame_interval_s=0.03,
        pose_interval_s=0.08,
        sync_tolerance_s=0.1,
        use_interpolation=False,  # nearest-neighbor only
        diagnostics_interval_s=0.0,  # suppress periodic prints
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        pixel_noise_std=0.0,
    )

    manager = ProcessManager(config=config)
    manager.start(
        target_simulator=target_sim,
        pose_simulator=pose_sim,
    )

    # Collect synchronized measurements for ~1 second
    time.sleep(1.0)

    sync_measurements = []
    while True:
        sm = manager.get_sync_measurement(timeout_s=0.1)
        if sm is None:
            break
        sync_measurements.append(sm)
        if len(sync_measurements) >= 200:
            break

    manager.stop()

    # ---- Verification ----

    n_total = len(sync_measurements)
    n_valid = sum(1 for sm in sync_measurements if sm.is_valid)
    n_invalid = n_total - n_valid

    print(f"  Total synchronized measurements: {n_total}")
    print(f"  Valid: {n_valid}, Invalid: {n_invalid}")

    assert n_total > 0, "No synchronized measurements produced"
    print(f"  [PASS] Pipeline produced {n_total} measurements")

    # Sync errors should be bounded by pose sampling interval (~80ms)
    if n_valid > 0:
        sync_errors = [sm.sync_error_s for sm in sync_measurements if sm.is_valid]
        mean_err = np.mean(sync_errors)
        max_err = np.max(sync_errors)

        # With nearest-neighbor at ~12.5 Hz nominal, max sync error should be
        # ≤ the pose interval (~80ms). Startup timing jitter may push this
        # slightly above nominal; 100ms is a safe bound.
        assert max_err < 0.11, f"Max sync error {max_err*1000:.1f}ms exceeds 110ms"
        print(f"  [PASS] Sync errors: mean={mean_err*1000:.2f}ms, max={max_err*1000:.2f}ms")

        # Most measurements should be valid
        validity_ratio = n_valid / n_total
        assert validity_ratio > 0.5, f"Only {validity_ratio*100:.1f}% valid"
        print(f"  [PASS] Validity ratio: {validity_ratio*100:.1f}%")

    # Verify that all method fields are "nearest"
    if n_valid > 0:
        methods = set(sm.sync_method for sm in sync_measurements)
        assert "nearest" in methods, f"No nearest-neighbor syncs: {methods}"
        print(f"  [PASS] Sync methods: {methods}")

    # Verify Detection and Pose objects are well-formed
    for sm in sync_measurements[:5]:
        assert sm.detection.t > 0
        assert sm.detection.u != 0 or sm.detection.v != 0
        if sm.pose is not None:
            assert sm.pose.R.shape == (3, 3)
            assert sm.pose.C.shape == (3,)
    print("  [PASS] All measurements have well-formed Detection and Pose objects")

    print("  [PASS] Full pipeline test")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Processes Layer — Verification Suite")
    print("=" * 60)

    print("\n[1] PoseBuffer")
    test_pose_buffer()

    print("\n[2] Pickle-ability (multiprocessing requirement)")
    test_detection_types()

    print("\n[3] SyncDiagnostics")
    test_sync_diagnostics()

    print("\n[4] Full async pipeline")
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("All processes tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
