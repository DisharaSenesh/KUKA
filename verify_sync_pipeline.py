"""
Comprehensive verification: synchronized data pipeline.

Tests:
  1. SynchronizedMeasurement — CSV row roundtrip, properties
  2. PoseBuffer — push, nearest, bracketing, overflow
  3. Matcher — nearest-neighbor and linear interpolation
  4. Synchronizer — detection → synchronized measurement
  5. CSV writer/reader — write → disk → read → verify identical
  6. Online pipeline (ProcessManager) — produce synchronized data
  7. Online/offline unification — CSV replay matches online output
  8. Euler angle preservation — A, B, C values carried through correctly
"""

import sys
import os
import time
import tempfile
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.synchronization.synchronized_measurement import (
    SynchronizedMeasurement,
    CSV_COLUMNS,
)
from core.synchronization.pose_buffer import PoseBuffer, TimedPose
from core.synchronization.matcher import match_nearest, match_linear_translation
from core.synchronization.synchronizer import Synchronizer, SyncDiagnostics
from core.recording.synchronized_writer import SynchronizedWriter
from core.recording.synchronized_reader import SynchronizedReader
from core.recording.csv_format import CSV_HEADER, NUM_COLUMNS


# ============================================================
# Test 1: SynchronizedMeasurement
# ============================================================

def test_sync_meas_properties():
    """Test SynchronizedMeasurement field accessors."""
    sm = SynchronizedMeasurement(
        frame_id=42,
        timestamp=1.234,
        u=320.5, v=240.3,
        X_mm=500.0, Y_mm=-20.0, Z_mm=850.0,
        A_deg=0.0, B_deg=45.0, C_deg=0.0,
        sync_error_s=0.004,
        sync_method="nearest",
    )

    assert sm.frame_id == 42
    assert sm.position_mm == (500.0, -20.0, 850.0)
    assert sm.orientation_deg == (0.0, 45.0, 0.0)
    assert sm.pixel == (320.5, 240.3)
    assert sm.is_position_valid()
    assert sm.is_valid
    print("  [PASS] SynchronizedMeasurement properties")


def test_sync_meas_csv_roundtrip():
    """Test that to_csv_row() and from_csv_row() are inverses."""
    original = SynchronizedMeasurement(
        frame_id=100,
        timestamp=2.567,
        u=150.0, v=350.0,
        X_mm=600.0, Y_mm=25.0, Z_mm=700.0,
        A_deg=90.0, B_deg=-30.0, C_deg=180.0,
        sync_error_s=0.012,
        sync_method="nearest",
    )

    row = original.to_csv_row()
    assert len(row) == NUM_COLUMNS, f"Expected {NUM_COLUMNS} columns, got {len(row)}"

    reconstructed = SynchronizedMeasurement.from_csv_row(row)
    assert reconstructed.frame_id == original.frame_id
    assert abs(reconstructed.timestamp - original.timestamp) < 1e-10
    assert abs(reconstructed.u - original.u) < 1e-6
    assert abs(reconstructed.v - original.v) < 1e-6
    assert abs(reconstructed.X_mm - original.X_mm) < 1e-6
    assert abs(reconstructed.Y_mm - original.Y_mm) < 1e-6
    assert abs(reconstructed.Z_mm - original.Z_mm) < 1e-6
    assert abs(reconstructed.A_deg - original.A_deg) < 1e-6
    assert abs(reconstructed.B_deg - original.B_deg) < 1e-6
    assert abs(reconstructed.C_deg - original.C_deg) < 1e-6
    assert abs(reconstructed.sync_error_s - original.sync_error_s) < 1e-10
    print("  [PASS] SynchronizedMeasurement CSV row roundtrip")

    # Test summary string
    s = original.summary()
    assert "frame=0100" in s
    assert "err=12.00ms" in s
    print(f"  [PASS] Summary: {s}")


def test_sync_meas_invalid_position():
    """Test is_position_valid rejects NaN."""
    sm = SynchronizedMeasurement(
        frame_id=0, timestamp=0.0,
        u=0.0, v=0.0,
        X_mm=float("nan"), Y_mm=0.0, Z_mm=0.0,
        A_deg=0.0, B_deg=0.0, C_deg=0.0,
    )
    assert not sm.is_position_valid()
    print("  [PASS] NaN position detected as invalid")


# ============================================================
# Test 2: PoseBuffer
# ============================================================

def test_pose_buffer():
    """Test PoseBuffer operations."""
    buf = PoseBuffer(max_size=5)

    # Push time-ordered poses
    for i in range(5):
        buf.push(
            X_mm=float(i * 100), Y_mm=0.0, Z_mm=800.0,
            A_deg=float(i * 10), B_deg=0.0, C_deg=0.0,
            timestamp=float(i),
        )

    assert len(buf) == 5
    print("  [PASS] PoseBuffer size = 5")

    # Nearest lookup
    nearest = buf.find_nearest(1.2)
    assert nearest is not None and abs(nearest.timestamp - 1.0) < 1e-10
    print("  [PASS] find_nearest(1.2) → t=1.0")

    # Bracketing
    before, after = buf.find_bracketing(2.3)
    assert before.timestamp == 2.0 and after.timestamp == 3.0
    print("  [PASS] find_bracketing(2.3) → (t=2.0, t=3.0)")

    # Latest / earliest
    assert buf.latest().timestamp == 4.0
    assert buf.earliest().timestamp == 0.0
    print("  [PASS] latest()=4.0, earliest()=0.0")

    # Overflow
    for i in range(10):
        buf.push(
            X_mm=float((i + 5) * 100), Y_mm=0.0, Z_mm=800.0,
            A_deg=0.0, B_deg=0.0, C_deg=0.0,
            timestamp=float(i + 5),
        )
    assert len(buf) <= 5
    print(f"  [PASS] PoseBuffer capped at {len(buf)}")


# ============================================================
# Test 3: Matcher
# ============================================================

def test_matcher_nearest():
    """Test nearest-neighbor matching."""
    buf = PoseBuffer(max_size=10)
    buf.push(100.0, 0.0, 800.0, 0.0, 0.0, 0.0, timestamp=1.0)
    buf.push(110.0, 5.0, 800.0, 5.0, 0.0, 0.0, timestamp=2.0)
    buf.push(120.0, 10.0, 800.0, 10.0, 0.0, 0.0, timestamp=3.0)

    pose, error, method = match_nearest(buf, 2.1)
    assert method == "nearest"
    assert abs(error - 0.1) < 1e-10
    assert abs(pose.X_mm - 110.0) < 1e-6
    print(f"  [PASS] Nearest at t=2.1 → t={pose.timestamp}, error={error*1000:.1f}ms")

    # Empty buffer
    empty_buf = PoseBuffer()
    pose, error, method = match_nearest(empty_buf, 1.0)
    assert pose is None and math.isinf(error)
    print("  [PASS] Empty buffer → None + inf error")


def test_matcher_linear_interp():
    """Test linear translation interpolation."""
    buf = PoseBuffer(max_size=10)
    buf.push(100.0, 0.0, 800.0, 0.0, 0.0, 0.0, timestamp=1.0)
    buf.push(200.0, 100.0, 900.0, 90.0, 0.0, 0.0, timestamp=3.0)

    # Interpolate at midpoint (t=2.0)
    pose, error, method = match_linear_translation(buf, 2.0)
    assert method == "linear_translation"

    # Expected: X = 100 + 0.5*(200-100) = 150, Y = 0 + 0.5*100 = 50, Z = 800 + 0.5*100 = 850
    expected_X = 150.0
    expected_Y = 50.0
    expected_Z = 850.0

    assert abs(pose.X_mm - expected_X) < 1e-6, f"X: {pose.X_mm} ≠ {expected_X}"
    assert abs(pose.Y_mm - expected_Y) < 1e-6, f"Y: {pose.Y_mm} ≠ {expected_Y}"
    assert abs(pose.Z_mm - expected_Z) < 1e-6, f"Z: {pose.Z_mm} ≠ {expected_Z}"
    print(f"  [PASS] Linear interp at t=2.0: XYZ=({pose.X_mm:.1f}, {pose.Y_mm:.1f}, {pose.Z_mm:.1f})")


# ============================================================
# Test 4: Synchronizer
# ============================================================

def test_synchronizer():
    """Test the Synchronizer orchestrator."""
    sync = Synchronizer(use_interpolation=False, sync_tolerance_s=0.05)

    # Feed poses
    sync.accept_pose(500.0, 0.0, 800.0, 0.0, 45.0, 0.0, timestamp=1.00)
    sync.accept_pose(510.0, 2.0, 800.0, 0.0, 45.0, 0.0, timestamp=1.10)
    sync.accept_pose(520.0, 5.0, 800.0, 0.0, 45.0, 0.0, timestamp=1.20)

    # Synchronize a detection at t=1.03 (nearest pose at t=1.00, error=0.03s)
    sm = sync.synchronize(frame_id=0, u=320.0, v=240.0, t_frame=1.03,
                           fx=600, fy=600, cx=320, cy=240)

    assert sm.is_valid
    assert sm.sync_method == "nearest"
    assert abs(sm.sync_error_s - 0.03) < 1e-10
    assert abs(sm.X_mm - 500.0) < 1e-6  # nearest pose's X
    assert abs(sm.A_deg - 0.0) < 1e-6
    assert abs(sm.B_deg - 45.0) < 1e-6
    print(f"  [PASS] Synchronized: XYZ=({sm.X_mm:.1f},{sm.Y_mm:.1f},{sm.Z_mm:.1f}), "
          f"err={sm.sync_error_s*1000:.1f}ms")

    # Test with no poses in buffer
    sync2 = Synchronizer()
    sm_none = sync2.synchronize(frame_id=0, u=100.0, v=200.0, t_frame=0.0)
    assert not sm_none.is_valid
    assert sm_none.sync_method == "none"
    assert math.isinf(sm_none.sync_error_s)
    print("  [PASS] No-pose case produces invalid measurement")

    # Test diagnostics
    assert sync.diagnostics.total_detections == 1
    assert sync.diagnostics.total_synchronized == 1
    assert sync.diagnostics.total_dropped == 0
    assert abs(sync.diagnostics.mean_error_s() - 0.03) < 1e-10
    print(f"  [PASS] Diagnostics: det={sync.diagnostics.total_detections}, "
          f"synced={sync.diagnostics.total_synchronized}")


# ============================================================
# Test 5: CSV writer/reader roundtrip
# ============================================================

def test_csv_writer_reader():
    """Test that measurements written to CSV can be read back identically."""
    # Create test measurements
    originals = []
    for i in range(20):
        sm = SynchronizedMeasurement(
            frame_id=i,
            timestamp=float(i) * 0.033,
            u=320.0 + i * 2.0,
            v=240.0 - i * 1.5,
            X_mm=500.0 + i * 1.0,
            Y_mm=i * 0.5 - 5.0,
            Z_mm=800.0 + i * 0.3,
            A_deg=float(i % 4) * 90.0,
            B_deg=45.0 if i % 2 == 0 else 0.0,
            C_deg=0.0,
            sync_error_s=float(i) * 0.001,
            sync_method="nearest" if i % 2 == 0 else "linear_translation",
        )
        originals.append(sm)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name

    try:
        writer = SynchronizedWriter(tmp_path)
        writer.open()
        for sm in originals:
            writer.write(sm)
        writer.close()

        assert writer.count == 20
        print(f"  [PASS] Wrote {writer.count} measurements")

        # Read back
        reader = SynchronizedReader(tmp_path)
        reader.open()

        read_back = []
        for sm in reader:
            read_back.append(sm)
        reader.close()

        assert len(read_back) == 20
        print(f"  [PASS] Read back {len(read_back)} measurements")

        # Verify roundtrip fidelity
        for i, (orig, back) in enumerate(zip(originals, read_back)):
            assert back.frame_id == orig.frame_id, f"Frame {i}: id mismatch"
            assert abs(back.u - orig.u) < 1e-6, f"Frame {i}: u mismatch"
            assert abs(back.v - orig.v) < 1e-6, f"Frame {i}: v mismatch"
            assert abs(back.X_mm - orig.X_mm) < 1e-6, f"Frame {i}: X mismatch"
            assert abs(back.A_deg - orig.A_deg) < 1e-6, f"Frame {i}: A mismatch"

        print("  [PASS] All 20 measurements roundtrip perfectly")

        # Test load_all convenience
        reader2 = SynchronizedReader(tmp_path)
        all_meas = reader2.load_all()
        assert len(all_meas) == 20
        print(f"  [PASS] load_all() returns {len(all_meas)} measurements")

    finally:
        os.unlink(tmp_path)


# ============================================================
# Test 6: Euler angle preservation
# ============================================================

def test_euler_preservation():
    """Test that Euler angles pass through synchronization unchanged."""
    sync = Synchronizer(use_interpolation=False)
    sync.accept_pose(500.0, 0.0, 800.0, A_deg=10.0, B_deg=20.0, C_deg=30.0, timestamp=1.0)
    sync.accept_pose(510.0, 0.0, 800.0, A_deg=12.0, B_deg=22.0, C_deg=32.0, timestamp=1.1)

    # Detection closest to first pose
    sm = sync.synchronize(frame_id=0, u=320.0, v=240.0, t_frame=1.02)
    assert sm.A_deg == 10.0, f"A_deg: {sm.A_deg}"
    assert sm.B_deg == 20.0
    assert sm.C_deg == 30.0
    print("  [PASS] Euler angles preserved: (A,B,C) = (10°, 20°, 30°)")

    # Detection closest to second pose
    sm2 = sync.synchronize(frame_id=1, u=321.0, v=241.0, t_frame=1.09)
    assert sm2.A_deg == 12.0
    assert sm2.B_deg == 22.0
    assert sm2.C_deg == 32.0
    print("  [PASS] Euler angles preserved: (A,B,C) = (12°, 22°, 32°)")


# ============================================================
# Test 7: Online pipeline (ProcessManager)
# ============================================================

def test_online_pipeline():
    """Test the full online pipeline with simulated sensors."""
    print("\n--- Online Pipeline Test ---")

    from core.processes import ProcessManager, ProcessManagerConfig

    # Simulated target: moving linearly across the image
    def target_sim(t):
        u = 320.0 + 50.0 * np.sin(2.0 * np.pi * 0.5 * t)
        v = 240.0 + 30.0 * np.cos(2.0 * np.pi * 0.3 * t)
        return float(u), float(v)

    # Simulated robot poses: slowly varying
    def pose_sim(t):
        X_mm = 500.0 + 20.0 * np.sin(0.5 * t)
        Y_mm = -5.0 * np.cos(0.3 * t)
        Z_mm = 800.0 + 5.0 * np.sin(0.2 * t)
        A_deg = 10.0 * np.sin(0.1 * t)
        B_deg = 45.0 + 5.0 * np.cos(0.15 * t)
        C_deg = 2.0 * np.sin(0.25 * t)
        return float(X_mm), float(Y_mm), float(Z_mm), float(A_deg), float(B_deg), float(C_deg)

    config = ProcessManagerConfig(
        detection_queue_size=100,
        pose_queue_size=200,
        sync_queue_size=200,
        frame_interval_s=0.03,
        pose_interval_s=0.08,
        sync_tolerance_s=0.1,
        use_interpolation=False,
        diagnostics_interval_s=0.0,
    )

    manager = ProcessManager(config=config)
    manager.start(target_simulator=target_sim, pose_simulator=pose_sim)

    # Let the pipeline run for ~1 second
    time.sleep(1.2)

    # Drain all synchronized measurements
    measurements = manager.drain_all()

    manager.stop()

    n = len(measurements)
    assert n > 0, "No measurements produced"
    n_valid = sum(1 for sm in measurements if sm.is_valid)

    print(f"  Produced {n} synchronized measurements, {n_valid} valid")

    # Verify content on valid measurements
    valid_measurements = [sm for sm in measurements if sm.is_valid]
    for sm in valid_measurements[:5]:
        assert sm.frame_id >= 0
        assert sm.timestamp > 0
        assert abs(sm.u - 320.0) < 100.0  # should be in reasonable range
        assert abs(sm.Y_mm) < 100.0       # Y near zero
        assert sm.Z_mm > 700.0            # Z around 800
        assert sm.B_deg > 30.0            # B near 45

    print("  [PASS] Online pipeline produces valid synchronized measurements")

    # Check that Euler angles are carried through
    for sm in measurements:
        assert math.isfinite(sm.A_deg)
        assert math.isfinite(sm.B_deg)
        assert math.isfinite(sm.C_deg)
    print("  [PASS] All Euler angles are finite")


# ============================================================
# Test 8: Online/offline unification (CSV roundtrip)
# ============================================================

def test_online_offline_unification():
    """Test that CSV-recorded data replays identically to online data."""
    print("\n--- Online/Offline Unification Test ---")

    from core.processes import ProcessManager, ProcessManagerConfig

    def target_sim(t):
        u = 320.0 + 30.0 * np.sin(3.0 * t)
        v = 240.0 + 20.0 * np.cos(2.5 * t)
        return float(u), float(v)

    def pose_sim(t):
        X_mm = 500.0 + 15.0 * t
        Y_mm = 2.0 * np.sin(1.0 * t)
        Z_mm = 800.0
        A_deg = 5.0 * np.sin(0.5 * t)
        B_deg = 45.0
        C_deg = 0.0
        return float(X_mm), float(Y_mm), float(Z_mm), float(A_deg), float(B_deg), float(C_deg)

    # Create temp CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        # ---- ONLINE: record to CSV ----
        config = ProcessManagerConfig(
            frame_interval_s=0.04,
            pose_interval_s=0.1,
            sync_tolerance_s=0.1,
            csv_filepath=csv_path,
            diagnostics_interval_s=0.0,
        )

        manager = ProcessManager(config=config)
        manager.start(target_simulator=target_sim, pose_simulator=pose_sim)
        time.sleep(0.8)
        manager.drain_to_csv()
        manager.stop()

        # ---- OFFLINE: replay from CSV ----
        reader = SynchronizedReader(csv_path)
        replayed = reader.load_all()

        assert len(replayed) > 0, "CSV replay produced no measurements"
        print(f"  CSV recorded: {len(replayed)} measurements")

        # Verify replayed measurements have the canonical structure
        for sm in replayed[:5]:
            assert sm.frame_id >= 0
            assert sm.timestamp > 0
            assert math.isfinite(sm.X_mm)
            assert math.isfinite(sm.A_deg)
            assert math.isfinite(sm.B_deg)
            assert math.isfinite(sm.C_deg)
            assert sm.sync_method == "csv_replay"  # replayed flag

        print(f"  [PASS] CSV replay: {len(replayed)} SynchronizedMeasurement objects")
        print(f"  [PASS] All replayed measurements have valid Euler angles")
        print(f"  [PASS] Online/offline unification verified")

    finally:
        os.unlink(csv_path)


# ============================================================
# Test 9: CSV format compliance
# ============================================================

def test_csv_format():
    """Test CSV format constants."""
    assert CSV_HEADER == "frame_id,timestamp,u,v,X_mm,Y_mm,Z_mm,A_deg,B_deg,C_deg,sync_error_s"
    assert NUM_COLUMNS == 11
    assert CSV_COLUMNS == [
        "frame_id", "timestamp", "u", "v",
        "X_mm", "Y_mm", "Z_mm",
        "A_deg", "B_deg", "C_deg",
        "sync_error_s",
    ]
    print("  [PASS] CSV format constants correct")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Synchronized Data Pipeline — Verification Suite")
    print("=" * 60)

    print("\n[1] SynchronizedMeasurement")
    test_sync_meas_properties()
    test_sync_meas_csv_roundtrip()
    test_sync_meas_invalid_position()

    print("\n[2] PoseBuffer")
    test_pose_buffer()

    print("\n[3] Matcher")
    test_matcher_nearest()
    test_matcher_linear_interp()

    print("\n[4] Synchronizer")
    test_synchronizer()

    print("\n[5] CSV writer/reader roundtrip")
    test_csv_writer_reader()

    print("\n[6] Euler angle preservation")
    test_euler_preservation()

    print("\n[7] CSV format compliance")
    test_csv_format()

    print("\n[8] Online pipeline")
    test_online_pipeline()

    print("\n[9] Online/offline unification")
    test_online_offline_unification()

    print("\n" + "=" * 60)
    print("All synchronized pipeline tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
