"""
Verification suite for the final output layer.

Tests:
  1. TrackingState enum
  2. RobotPose6D construction, validation, serialization
  3. ObjectPoseEstimate construction and field validation
  4. is_valid() gate logic
  5. to_dict() and to_json() roundtrip
  6. pretty() and summary_line() formatting
  7. Factory constructors (lost, initializing, interpolated)
  8. OutputBuilder — build_from_sync
  9. OutputBuilder — build_from_track with TrajectoryEstimate
  10. JSON deserialization compatibility
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.output.object_pose_output import (
    TrackingState,
    RobotPose6D,
    ObjectPoseEstimate,
    OutputBuilder,
)

from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from core.reconstruction.trajectory_estimate import TrajectoryEstimate


# ============================================================
# Test 1: TrackingState
# ============================================================

def test_tracking_state():
    """Test TrackingState enum values and from_string."""
    assert TrackingState.TRACKED.name == "TRACKED"
    assert TrackingState.LOST.name == "LOST"
    assert TrackingState.INTERPOLATED.name == "INTERPOLATED"
    assert TrackingState.INITIALIZING.name == "INITIALIZING"
    assert TrackingState.INVALID.name == "INVALID"
    print("  [PASS] TrackingState enum defined correctly")

    # from_string
    assert TrackingState.from_string("tracked") == TrackingState.TRACKED
    assert TrackingState.from_string("LOST") == TrackingState.LOST
    assert TrackingState.from_string("initializing") == TrackingState.INITIALIZING
    assert TrackingState.from_string("bogus") == TrackingState.INVALID
    print("  [PASS] TrackingState.from_string handles case and invalid")


# ============================================================
# Test 2: RobotPose6D
# ============================================================

def test_robot_pose_6d():
    """Test RobotPose6D construction and serialization."""
    rp = RobotPose6D(x=500.0, y=0.0, z=800.0, rx=0.0, ry=45.0, rz=0.0)

    assert rp.x == 500.0
    assert rp.ry == 45.0
    assert rp.is_valid()
    print("  [PASS] RobotPose6D construction and is_valid")

    tup = rp.to_tuple()
    assert tup == (500.0, 0.0, 800.0, 0.0, 45.0, 0.0)
    print("  [PASS] to_tuple()")

    d = rp.to_dict()
    assert d["x"] == 500.0
    assert d["rz"] == 0.0
    print("  [PASS] to_dict()")

    # Invalid pose
    rp_bad = RobotPose6D(x=float("nan"), y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
    assert not rp_bad.is_valid()
    print("  [PASS] NaN pose detected as invalid")


# ============================================================
# Test 3: ObjectPoseEstimate construction
# ============================================================

def test_estimate_construction():
    """Test basic ObjectPoseEstimate creation."""
    rp = RobotPose6D(500.0, 0.0, 800.0, 0.0, 45.0, 0.0)
    op = RobotPose6D(512.0, -5.0, 825.0, 0.0, 45.0, 0.0)

    est = ObjectPoseEstimate(
        timestamp=1.234,
        frame_id=42,
        object_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.5, v=240.3,
        robot_pose=rp,
        object_pose_base=op,
        confidence=0.92,
        sync_error_s=0.004,
        reprojection_error_px=0.15,
        polynomial_order=1,
        window_size=12,
    )

    assert est.timestamp == 1.234
    assert est.frame_id == 42
    assert est.object_id == 1
    assert est.tracking_state == TrackingState.TRACKED
    assert est.u == 320.5 and est.v == 240.3
    assert est.confidence == 0.92
    assert est.sync_error_s == 0.004
    print("  [PASS] ObjectPoseEstimate construction")


# ============================================================
# Test 4: is_valid() logic
# ============================================================

def test_is_valid():
    """Test ObjectPoseEstimate validity gating."""
    rp = RobotPose6D(500.0, 0.0, 800.0, 0.0, 45.0, 0.0)

    # Valid estimate
    est_valid = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.0, v=240.0,
        robot_pose=rp,
        confidence=0.8,
    )
    assert est_valid.is_valid()
    print("  [PASS] Valid estimate passes is_valid()")

    # LOST state → not valid
    est_lost = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.LOST,
        confidence=0.0,
    )
    assert not est_lost.is_valid()
    print("  [PASS] LOST state → not valid")

    # INVALID state → not valid
    est_invalid = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.INVALID,
        robot_pose=rp,
        confidence=0.8,
    )
    assert not est_invalid.is_valid()
    print("  [PASS] INVALID state → not valid")

    # No robot pose → not valid
    est_no_robot = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.0, v=240.0,
        confidence=0.8,
    )
    assert not est_no_robot.is_valid()
    print("  [PASS] Missing robot pose → not valid")

    # Zero confidence → not valid
    est_zero_conf = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.0, v=240.0,
        robot_pose=rp,
        confidence=0.0,
    )
    assert not est_zero_conf.is_valid()
    print("  [PASS] Zero confidence → not valid")


# ============================================================
# Test 5: to_dict() and to_json()
# ============================================================

def test_serialization():
    """Test dict and JSON serialization."""
    rp = RobotPose6D(500.0, 0.0, 800.0, 0.0, 45.0, 0.0)
    op = RobotPose6D(512.0, -5.0, 825.0, 0.0, 45.0, 0.0)

    est = ObjectPoseEstimate(
        timestamp=1.234, frame_id=42, object_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.5, v=240.3,
        robot_pose=rp,
        object_pose_base=op,
        confidence=0.92,
        sync_error_s=0.004,
        reprojection_error_px=0.15,
        polynomial_order=1,
        window_size=12,
    )

    # to_dict
    d = est.to_dict()
    assert d["frame_id"] == 42
    assert d["tracking_state"] == "tracked"
    assert d["camera_uv"] == [320.5, 240.3]
    assert d["robot_pose"]["x"] == 500.0
    assert d["robot_pose"]["ry"] == 45.0
    assert d["object_pose_base"]["x"] == 512.0
    assert d["confidence"] == 0.92
    assert d["diagnostics"]["polynomial_order"] == 1
    print("  [PASS] to_dict() produces correct structure")

    # to_json
    json_str = est.to_json(indent=2)
    assert isinstance(json_str, str)
    assert '"frame_id": 42' in json_str
    print("  [PASS] to_json() produces valid JSON")

    # Compact JSON
    json_compact = est.to_json(indent=0)
    assert len(json_compact) > 0
    assert "\n" not in json_compact
    print("  [PASS] to_json(indent=0) produces compact output")

    # JSON roundtrip
    parsed = json.loads(json_str)
    assert parsed["frame_id"] == 42
    assert parsed["object_id"] == 1
    assert parsed["tracking_state"] == "tracked"
    assert parsed["camera_uv"] == [320.5, 240.3]
    print("  [PASS] JSON roundtrip preserves all fields")


# ============================================================
# Test 6: pretty() and summary_line()
# ============================================================

def test_formatting():
    """Test human-readable formatting methods."""
    rp = RobotPose6D(500.0, 0.0, 800.0, 0.0, 45.0, 0.0)
    op = RobotPose6D(512.0, -5.0, 825.0, 0.0, 45.0, 0.0)

    est = ObjectPoseEstimate(
        timestamp=1.234, frame_id=42, object_id=1,
        tracking_state=TrackingState.TRACKED,
        u=320.5, v=240.3,
        robot_pose=rp,
        object_pose_base=op,
        confidence=0.92,
        sync_error_s=0.004,
        reprojection_error_px=0.15,
        polynomial_order=1,
        window_size=12,
    )

    # pretty()
    pretty_str = est.pretty()
    assert "frame=0042" in pretty_str
    assert "TRACKED" in pretty_str
    assert "camera_uv" in pretty_str
    assert "robot_pose" in pretty_str
    assert "object_pose" in pretty_str
    assert "0.15 px" in pretty_str
    print("  [PASS] pretty() contains all sections")

    # summary_line()
    summary = est.summary_line()
    assert "[0042]" in summary
    assert "TRAC" in summary  # first 4 chars of TRACKED
    assert "(320,240)" in summary
    assert "c=0.92" in summary
    print(f"  [PASS] summary_line(): {summary}")

    # Estimate without object pose
    est_no_obj = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.INITIALIZING,
        u=100.0, v=200.0,
        robot_pose=rp,
        confidence=0.1,
    )
    pretty2 = est_no_obj.pretty()
    assert "object_pose: None" in pretty2
    summary2 = est_no_obj.summary_line()
    assert "O=None" in summary2
    print("  [PASS] Missing object pose handled gracefully")


# ============================================================
# Test 7: Factory constructors
# ============================================================

def test_factories():
    """Test lost(), initializing(), interpolated() constructors."""
    # lost
    est_lost = ObjectPoseEstimate.lost(frame_id=10, timestamp=0.5)
    assert est_lost.tracking_state == TrackingState.LOST
    assert est_lost.confidence == 0.0
    assert not est_lost.is_valid()
    print("  [PASS] lost() factory")

    # initializing
    est_init = ObjectPoseEstimate.initializing(
        frame_id=11, timestamp=0.6, u=320.0, v=240.0
    )
    assert est_init.tracking_state == TrackingState.INITIALIZING
    assert est_init.u == 320.0
    assert est_init.confidence == 0.1
    print("  [PASS] initializing() factory")

    # interpolated
    rp = RobotPose6D(500.0, 0.0, 800.0, 0.0, 45.0, 0.0)
    est_interp = ObjectPoseEstimate.interpolated(
        frame_id=12, timestamp=0.7, robot_pose=rp
    )
    assert est_interp.tracking_state == TrackingState.INTERPOLATED
    assert est_interp.robot_pose is not None
    assert est_interp.confidence == 0.5
    print("  [PASS] interpolated() factory")


# ============================================================
# Test 8: OutputBuilder — build_from_sync
# ============================================================

def test_builder_from_sync():
    """Test building estimate from a SynchronizedMeasurement."""
    sync = SynchronizedMeasurement(
        frame_id=42,
        timestamp=1.234,
        u=320.5, v=240.3,
        X_mm=500.0, Y_mm=0.0, Z_mm=800.0,
        A_deg=0.0, B_deg=45.0, C_deg=0.0,
        sync_error_s=0.004,
        sync_method="nearest",
        is_valid=True,
    )

    builder = OutputBuilder()
    est = builder.build_from_sync(sync, object_id=1)

    assert est.frame_id == 42
    assert est.object_id == 1
    assert est.tracking_state == TrackingState.INITIALIZING
    assert est.u == 320.5
    assert est.v == 240.3
    assert est.confidence == 0.1
    assert est.robot_pose is not None
    assert abs(est.robot_pose.x - 500.0) < 1e-6
    assert abs(est.robot_pose.ry - 45.0) < 1e-6
    # rx maps from C, ry from B, rz from A
    assert abs(est.robot_pose.rz - 0.0) < 1e-6
    assert abs(est.robot_pose.rx - 0.0) < 1e-6
    assert est.sync_error_s == 0.004
    print("  [PASS] build_from_sync creates correct INITIALIZING estimate")

    # Invalid sync measurement
    sync_invalid = SynchronizedMeasurement(
        frame_id=43,
        timestamp=1.5,
        u=0.0, v=0.0,
        X_mm=0.0, Y_mm=0.0, Z_mm=0.0,
        A_deg=0.0, B_deg=0.0, C_deg=0.0,
        is_valid=False,
    )
    est_inv = builder.build_from_sync(sync_invalid)
    assert est_inv.confidence == 0.0
    print("  [PASS] build_from_sync marks invalid sync with confidence=0")


# ============================================================
# Test 9: OutputBuilder — build_from_track
# ============================================================

def test_builder_from_track():
    """Test building full estimate from sync + trajectory."""
    sync = SynchronizedMeasurement(
        frame_id=42,
        timestamp=1.234,
        u=320.5, v=240.3,
        X_mm=500.0, Y_mm=0.0, Z_mm=800.0,
        A_deg=0.0, B_deg=45.0, C_deg=0.0,
        sync_error_s=0.004,
        is_valid=True,
    )

    # Linear trajectory: X(t) = [1, 2, 0.5] + [0.3, -0.2, 0.05]*(t-0)
    traj = TrajectoryEstimate(
        order=1, t0=0.0,
        coefficients=[
            np.array([1.0, 2.0, 0.5]),
            np.array([0.3, -0.2, 0.05]),
        ],
        reprojection_rms=0.15,
        geometric_rms=0.001,
        window_size=12,
        window_time_span=2.0,
        window_t_start=0.0,
        window_t_end=2.0,
        solvable=True,
    )

    builder = OutputBuilder()
    est = builder.build_from_track(sync, traj, object_id=1)

    assert est.tracking_state == TrackingState.TRACKED
    assert est.frame_id == 42
    assert est.polynomial_order == 1
    assert est.window_size == 12
    assert abs(est.reprojection_error_px - 0.15) < 1e-6

    # Object pose should be computed from trajectory
    assert est.object_pose_base is not None
    # X(1.234) = [1+0.3*1.234, 2-0.2*1.234, 0.5+0.05*1.234]
    # = [1.3702, 1.7532, 0.5617] meters → [1370.2, 1753.2, 561.7] mm
    expected_x = (1.0 + 0.3 * 1.234) / 0.001  # ≈ 1370.2
    assert abs(est.object_pose_base.x - expected_x) < 0.1
    print(f"  [PASS] Object pose: ({est.object_pose_base.x:.1f}, "
          f"{est.object_pose_base.y:.1f}, {est.object_pose_base.z:.1f}) mm")

    # Confidence auto-computed
    assert 0 < est.confidence <= 1.0
    print(f"  [PASS] Auto-confidence: {est.confidence:.3f}")

    # Full roundtrip through JSON
    json_str = est.to_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed["tracking_state"] == "tracked"
    assert parsed["robot_pose"]["x"] == 500.0
    assert parsed["object_pose_base"] is not None
    print("  [PASS] build_from_track → to_json roundtrip")


# ============================================================
# Test 10: has_object_pose()
# ============================================================

def test_has_object_pose():
    """Test has_object_pose() method."""
    est_without = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.INITIALIZING,
    )
    assert not est_without.has_object_pose()
    print("  [PASS] has_object_pose() → False for missing pose")

    op = RobotPose6D(100.0, 200.0, 300.0, 0.0, 0.0, 0.0)
    est_with = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.TRACKED,
        object_pose_base=op,
    )
    assert est_with.has_object_pose()
    print("  [PASS] has_object_pose() → True when present")

    op_bad = RobotPose6D(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0)
    est_bad = ObjectPoseEstimate(
        timestamp=1.0, frame_id=1,
        tracking_state=TrackingState.TRACKED,
        object_pose_base=op_bad,
    )
    assert not est_bad.has_object_pose()
    print("  [PASS] has_object_pose() → False for NaN pose")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Output Layer — Verification Suite")
    print("=" * 60)

    print("\n[1] TrackingState")
    test_tracking_state()

    print("\n[2] RobotPose6D")
    test_robot_pose_6d()

    print("\n[3] ObjectPoseEstimate construction")
    test_estimate_construction()

    print("\n[4] is_valid() logic")
    test_is_valid()

    print("\n[5] Serialization (to_dict, to_json)")
    test_serialization()

    print("\n[6] Formatting (pretty, summary_line)")
    test_formatting()

    print("\n[7] Factory constructors")
    test_factories()

    print("\n[8] OutputBuilder — build_from_sync")
    test_builder_from_sync()

    print("\n[9] OutputBuilder — build_from_track")
    test_builder_from_track()

    print("\n[10] has_object_pose()")
    test_has_object_pose()

    print("\n" + "=" * 60)
    print("All output layer tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
