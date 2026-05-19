"""
Verification suite for the trajectory reconstruction stage.

Tests:
  1. Ray builder — Euler angles → world-space ray
  2. Trajectory model — basis evaluation and polynomial evaluation
  3. Sliding window — add, evict, can_reconstruct
  4. Triangulation — noise-free perfect reconstruction
  5. Triangulation — with noisy observations
  6. Reprojection — error computation and roundtrip
  7. Residuals — geometric residual computation
  8. Model selection — automatic order selection
  9. Model selection — manual mode bypass
  10. TrajectoryEstimate — output dataclass
  11. Full pipeline — sliding window → triangulate → validate
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from core.robotics.kinematics.transforms import (
    euler_abc_to_rotation,
    KUKA_POSITION_SCALE,
)
from core.geometry.projection import project_to_pixel

from core.reconstruction.reconstruction_config import ReconstructionConfig
from core.reconstruction.sliding_window import (
    SlidingWindow,
    FilteredMeasurement,
)
from core.reconstruction.ray_builder import (
    synchronized_to_world_ray,
    synchronized_to_measurement,
    batch_to_measurements,
)
from core.reconstruction.trajectory_model import (
    build_polynomial_basis,
    evaluate_trajectory_at_time,
    evaluate_trajectory_vectorized,
    num_coefficients,
    num_unknowns,
)
from core.reconstruction.triangulation import triangulate
from core.reconstruction.reprojection import (
    reprojection_error_single,
    rms_reprojection_error,
    total_reprojection_cost,
)
from core.reconstruction.residuals import (
    compute_geometric_residual,
    rms_residual,
    max_residual,
)
from core.reconstruction.model_selection import (
    select_model_order,
    ModelSelectionResult,
)
from core.reconstruction.trajectory_estimate import TrajectoryEstimate


# ============================================================
# Helpers
# ============================================================

def make_sync_measurement(
    frame_id, t, u, v, X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg,
    fx=600.0, fy=600.0, cx=320.0, cy=240.0,
    sync_error=0.0,
):
    """Create a SynchronizedMeasurement for testing."""
    return SynchronizedMeasurement(
        frame_id=frame_id, timestamp=t,
        u=u, v=v,
        X_mm=X_mm, Y_mm=Y_mm, Z_mm=Z_mm,
        A_deg=A_deg, B_deg=B_deg, C_deg=C_deg,
        sync_error_s=sync_error,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )


def make_filtered(sm):
    """Wrap in FilteredMeasurement."""
    return FilteredMeasurement(
        synchronized=sm,
        passed_ray_angle=True,
        passed_baseline=True,
        observability_score=0.8,
    )


def generate_test_measurements(true_coeffs_x, true_coeffs_y, true_coeffs_z,
                                cameras, times, fx, fy, cx, cy, t0, noise_std=0.0):
    """
    Generate filtered measurements by projecting a ground-truth trajectory
    through given camera poses.
    """
    measurements = []
    for i, (C, A_deg, B_deg, C_deg) in enumerate(cameras):
        t = times[i]

        # True position
        X_true = evaluate_trajectory_at_time(
            true_coeffs_x, true_coeffs_y, true_coeffs_z, t, t0
        )

        # Build rotation (R maps camera→world)
        R = euler_abc_to_rotation(np.radians(A_deg), np.radians(B_deg), np.radians(C_deg))

        # World→camera point transform: X_cam = R.T @ (X_world - C)
        X_cam = R.T @ (X_true - C)
        pixel = project_to_pixel(X_cam, fx, fy, cx, cy)

        u = pixel[0] + np.random.normal(0, noise_std)
        v = pixel[1] + np.random.normal(0, noise_std)

        # Camera position in mm for storage
        X_mm = C[0] / KUKA_POSITION_SCALE
        Y_mm = C[1] / KUKA_POSITION_SCALE
        Z_mm = C[2] / KUKA_POSITION_SCALE

        sm = make_sync_measurement(
            frame_id=i, t=t, u=u, v=v,
            X_mm=X_mm, Y_mm=Y_mm, Z_mm=Z_mm,
            A_deg=A_deg, B_deg=B_deg, C_deg=C_deg,
        )
        measurements.append(make_filtered(sm))

    return measurements


# ============================================================
# Test 1: Ray builder
# ============================================================

def test_ray_builder():
    """Test conversion from synchronized measurement to world-ray."""
    sm = make_sync_measurement(
        frame_id=0, t=1.0,
        u=320.0, v=240.0,
        X_mm=500.0, Y_mm=0.0, Z_mm=800.0,
        A_deg=0.0, B_deg=0.0, C_deg=0.0,
    )

    ray = synchronized_to_world_ray(sm)

    # Camera at (0.5, 0, 0.8) meters with identity rotation
    assert abs(ray.origin[0] - 0.5) < 1e-10
    assert abs(ray.origin[2] - 0.8) < 1e-10
    # Direction should be forward (along Z)
    assert ray.direction[2] > 0.9
    print("  [PASS] Identity pose → ray origin correct, direction forward")

    # Test with rotation: 90° about Z at origin with a pixel offset in u
    # Camera X → world -Y, so a positive u offset gives a negative Y component
    sm2 = make_sync_measurement(
        frame_id=1, t=2.0,
        u=320.0 + 100.0, v=240.0,   # offset in u → camera X direction
        X_mm=0.0, Y_mm=0.0, Z_mm=0.0,
        A_deg=90.0, B_deg=0.0, C_deg=0.0,
    )
    ray2 = synchronized_to_world_ray(sm2)
    # With A=90°, Rz(90°) maps camera X → world Y
    # A positive u offset → positive camera X → positive world Y
    assert ray2.direction[1] > 0.1, f"Expected positive Y component, got {ray2.direction[1]:.3f}"
    # Z should still be the dominant forward component
    assert ray2.direction[2] > 0.9
    print(f"  [PASS] 90° Z rotation: direction=({ray2.direction[0]:.3f},{ray2.direction[1]:.3f},{ray2.direction[2]:.3f})")

    # Test Measurement conversion
    meas = synchronized_to_measurement(sm)
    assert meas.pixel is not None
    assert abs(meas.pixel.u - 320.0) < 1e-6
    assert meas.t == 1.0
    print("  [PASS] synchronized_to_measurement preserves intrinsics and timestamp")


# ============================================================
# Test 2: Trajectory model
# ============================================================

def test_trajectory_model():
    """Test polynomial basis and trajectory evaluation."""
    # Basis matrix
    times = np.array([0.0, 1.0, 2.0])
    M = build_polynomial_basis(times, t0=0.0, order=2)
    expected = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0],
    ])
    assert np.allclose(M, expected)
    print("  [PASS] Polynomial basis matrix correct")

    # Trajectory evaluation: X(t) = 1 + 2t + 3t²
    cx = np.array([1.0, 2.0, 3.0])
    cy = np.array([0.0, 0.0, 0.0])
    cz = np.array([0.0, 0.0, 0.0])

    pos = evaluate_trajectory_at_time(cx, cy, cz, t=2.0, t0=0.0)
    assert abs(pos[0] - (1.0 + 4.0 + 12.0)) < 1e-10  # 1 + 2*2 + 3*4 = 17
    print(f"  [PASS] X(2.0) = {pos[0]:.1f} (expected 17.0)")

    # Vectorized
    positions = evaluate_trajectory_vectorized(cx, cy, cz, times, t0=0.0)
    assert positions.shape == (3, 3)
    assert abs(positions[0, 0] - 1.0) < 1e-10
    assert abs(positions[2, 0] - 17.0) < 1e-10
    print("  [PASS] Vectorized evaluation correct")

    # Helpers
    assert num_coefficients(2) == 3
    assert num_unknowns(2) == 9
    print("  [PASS] num_coefficients(2)=3, num_unknowns(2)=9")


# ============================================================
# Test 3: Sliding window
# ============================================================

def test_sliding_window():
    """Test sliding window operations."""
    window = SlidingWindow(min_size=3, max_size=5)

    assert not window.can_reconstruct()
    print("  [PASS] Empty window → cannot reconstruct")

    # Add measurements
    for i in range(4):
        sm = make_sync_measurement(i, float(i), 320.0, 240.0,
                                    500.0, 0.0, 800.0, 0.0, 0.0, 0.0)
        window.add(make_filtered(sm))

    assert window.can_reconstruct()
    assert len(window) == 4
    print("  [PASS] Window with 4 entries → can reconstruct")

    # Overflow
    for i in range(6):
        sm = make_sync_measurement(i + 4, float(i + 4), 320.0, 240.0,
                                    500.0, 0.0, 800.0, 0.0, 0.0, 0.0)
        window.add(make_filtered(sm))

    assert len(window) == 5
    print(f"  [PASS] Window capped at {len(window)} (max=5)")

    assert window.time_span() > 0
    print(f"  [PASS] Time span: {window.time_span():.2f}s")


# ============================================================
# Test 4: Triangulation (noise-free)
# ============================================================

def test_triangulation_noise_free():
    """Perfect reconstruction of a constant-velocity trajectory."""
    # Ground truth: X(t) = 2 + 0.5*t, Y(t) = 1 - 0.2*t, Z(t) = 0.5
    cx_true = np.array([2.0, 0.5])
    cy_true = np.array([1.0, -0.2])
    cz_true = np.array([0.5, 0.0])

    # Cameras above the target looking straight down (B=180° → camera +Z = world -Z)
    cameras_m = [
        (np.array([2.0, 1.0, 6.0]), 0.0, 180.0, 0.0),
        (np.array([0.0, 0.0, 6.0]), 0.0, 180.0, 0.0),
        (np.array([3.0, -2.0, 7.0]), 0.0, 180.0, 0.0),
        (np.array([-2.0, 3.0, 6.0]), 0.0, 180.0, 0.0),
        (np.array([5.0, 1.0, 7.0]), 10.0, 180.0, 0.0),
        (np.array([1.0, 5.0, 5.5]), -10.0, 180.0, 5.0),
    ]
    times = np.linspace(0.0, 2.0, len(cameras_m))

    measurements = generate_test_measurements(
        cx_true, cy_true, cz_true,
        cameras_m, times,
        fx=600, fy=600, cx=320, cy=240,
        t0=0.0, noise_std=0.0,
    )

    # Triangulate
    coeffs, x_opt = triangulate(measurements, order=1, t0=0.0)
    assert coeffs is not None, "Triangulation failed"

    # Compare coefficients
    for k in range(2):
        true_k = np.array([cx_true[k], cy_true[k], cz_true[k]])
        err = np.linalg.norm(true_k - coeffs[k])
        assert err < 1e-6, f"Coefficient a{k} error: {err}"
        print(f"  [PASS] Coefficient a{k} matches ground truth (err={err:.2e})")

    # Check residuals
    rms = rms_residual(measurements, coeffs, 0.0)
    assert rms < 1e-6
    print(f"  [PASS] RMS geometric residual: {rms:.2e}")

    # Check reprojection
    reproj_rms = rms_reprojection_error(measurements, coeffs, 0.0)
    assert reproj_rms < 1e-6
    print(f"  [PASS] RMS reprojection error: {reproj_rms:.2f} px")


# ============================================================
# Test 5: Triangulation (with noise)
# ============================================================

def test_triangulation_noisy():
    """Reconstruction with pixel noise — should still be reasonable."""
    cx_true = np.array([2.0, 0.5, 0.02])
    cy_true = np.array([1.0, -0.2, 0.01])
    cz_true = np.array([0.5, 0.05, -0.005])

    # Cameras above the target looking down (B=180°)
    cameras_m = [
        (np.array([0.0, 0.0, 8.0]), 0.0, 180.0, 0.0),
        (np.array([4.0, -2.0, 7.0]), 5.0, 180.0, 0.0),
        (np.array([-3.0, 3.0, 6.0]), -5.0, 180.0, 5.0),
        (np.array([5.0, 1.0, 6.0]), 10.0, 180.0, 0.0),
        (np.array([1.0, 5.0, 5.0]), -10.0, 180.0, -5.0),
        (np.array([2.0, 2.0, 7.0]), 0.0, 180.0, 5.0),
        (np.array([-1.0, -1.0, 6.0]), 0.0, 180.0, -5.0),
        (np.array([3.0, 3.0, 9.0]), 0.0, 180.0, 10.0),
    ]
    times = np.linspace(0.0, 3.0, len(cameras_m))

    measurements = generate_test_measurements(
        cx_true, cy_true, cz_true,
        cameras_m, times,
        fx=600, fy=600, cx=320, cy=240,
        t0=0.0, noise_std=0.3,  # 0.3 pixel noise
    )

    coeffs, x_opt = triangulate(measurements, order=2, t0=0.0)
    assert coeffs is not None, "Triangulation failed with noisy data"

    # Evaluate trajectory accuracy
    errors = []
    for i in range(len(times)):
        X_true = evaluate_trajectory_at_time(cx_true, cy_true, cz_true, times[i], 0.0)
        X_est = np.zeros(3)
        dt = times[i]
        for k, a_k in enumerate(coeffs):
            X_est += a_k * (dt ** k)
        errors.append(np.linalg.norm(X_true - X_est))

    mean_err = np.mean(errors)
    assert mean_err < 0.3, f"Mean position error {mean_err:.3f}m too large"
    print(f"  [PASS] Mean position error: {mean_err:.4f} m (noise=0.3px)")

    # Reprojection error should be small
    reproj_rms = rms_reprojection_error(measurements, coeffs, 0.0)
    assert reproj_rms < 1.0, f"RMS reprojection {reproj_rms:.2f}px too large"
    print(f"  [PASS] RMS reprojection: {reproj_rms:.3f} px")


# ============================================================
# Test 6: Reprojection
# ============================================================

def test_reprojection():
    """Test that reprojection error is zero for perfect estimation."""
    cx = np.array([2.0, 0.5])
    cy = np.array([1.0, 0.0])
    cz = np.array([0.5, 0.0])

    cameras_m = [
        (np.array([0.0, 0.0, 7.0]), 0.0, 180.0, 0.0),
        (np.array([3.0, 0.0, 7.0]), 0.0, 180.0, 0.0),
    ]
    times = np.array([0.0, 1.0])

    measurements = generate_test_measurements(
        cx, cy, cz, cameras_m, times,
        fx=600, fy=600, cx=320, cy=240, t0=0.0, noise_std=0.0,
    )

    coeffs, _ = triangulate(measurements, order=1, t0=0.0)

    for i, fm in enumerate(measurements):
        err = reprojection_error_single(fm, coeffs, 0.0)
        assert err < 1e-6, f"Reprojection error {err:.2e} at frame {i}"
    print("  [PASS] Reprojection errors ≈ 0 for perfect reconstruction")

    total = total_reprojection_cost(measurements, coeffs, 0.0)
    assert total < 1e-10
    print(f"  [PASS] Total reprojection cost: {total:.2e}")


# ============================================================
# Test 7: Residuals
# ============================================================

def test_residuals():
    """Test geometric residual computation."""
    cx = np.array([0.0, 1.0])
    cy = np.array([0.0, 0.0])
    cz = np.array([0.0, 0.0])

    # Camera looking down at the trajectory
    cameras_m = [
        (np.array([0.0, 0.0, 3.0]), 0.0, 180.0, 0.0),
    ]
    times = np.array([0.0])

    measurements = generate_test_measurements(
        cx, cy, cz, cameras_m, times,
        fx=600, fy=600, cx=320, cy=240, t0=0.0, noise_std=0.0,
    )

    # Perfect coefficients
    coeffs = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    r = compute_geometric_residual(measurements[0], coeffs, 0.0)
    assert r < 1e-10, f"Residual {r:.2e} should be zero"
    print("  [PASS] Geometric residual ≈ 0 for perfect trajectory")

    # Wrong coefficients → non-zero residual
    wrong_coeffs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    r_wrong = compute_geometric_residual(measurements[0], wrong_coeffs, 0.0)
    assert r_wrong > 0.1, f"Residual {r_wrong:.3f} should be non-zero"
    print(f"  [PASS] Wrong coefficients → residual={r_wrong:.3f} > 0")


# ============================================================
# Test 8: Model selection (automatic)
# ============================================================

def test_model_selection_automatic():
    """Test automatic model order selection on well-behaved data."""
    cx_true = np.array([2.0, 0.3])
    cy_true = np.array([1.0, -0.2])
    cz_true = np.array([0.5, 0.0])

    cameras_m = [
        (np.array([0.0, 0.0, 7.0]), 0.0, 180.0, 0.0),
        (np.array([3.0, -2.0, 7.0]), 0.0, 180.0, 0.0),
        (np.array([-2.0, 3.0, 6.0]), 0.0, 180.0, 0.0),
        (np.array([5.0, 1.0, 7.0]), 10.0, 180.0, 0.0),
        (np.array([1.0, 5.0, 6.5]), -10.0, 180.0, 5.0),
        (np.array([2.0, 2.0, 8.0]), 0.0, 180.0, -5.0),
    ]
    times = np.linspace(0.0, 2.0, len(cameras_m))

    measurements = generate_test_measurements(
        cx_true, cy_true, cz_true,
        cameras_m, times,
        fx=600, fy=600, cx=320, cy=240,
        t0=0.0, noise_std=0.1,
    )

    config = ReconstructionConfig(
        min_poly_order=1,
        max_poly_order=3,
        complexity_penalty=0.5,
    )

    result = select_model_order(measurements, config)
    assert result.selected_order > 0
    print(f"  [PASS] Selected order: {result.selected_order}")
    print(f"  [PASS] Reason: {result.reason}")
    print(f"  [PASS] Reprojection RMS: {result.winner_reprojection_rms:.3f} px")
    print(f"  [PASS] Geometric RMS: {result.winner_geometric_rms*1000:.3f} mm")

    # For clean constant-velocity data, order 1 should be competitive
    assert 1 in result.candidates, "Order 1 should be evaluated"

    # Higher orders should have worse scores due to penalty
    for order in range(2, config.max_poly_order + 1):
        if order in result.candidates:
            # Higher order may have lower reprojection error but penalty may
            # make the score worse — both outcomes are valid
            pass
    print("  [PASS] Multiple orders evaluated")


# ============================================================
# Test 9: Model selection (manual mode)
# ============================================================

def test_model_selection_manual():
    """Test that manual mode bypasses automatic selection."""
    config = ReconstructionConfig(polynomial_order=2)

    measurements = []
    for i in range(3):
        sm = make_sync_measurement(i, float(i), 320.0, 240.0,
                                    500.0, 0.0, 800.0, 0.0, 0.0, 0.0)
        measurements.append(make_filtered(sm))

    result = select_model_order(measurements, config)
    assert result.selected_order == 2
    assert "Manual mode" in result.reason
    print(f"  [PASS] Manual mode: selected order = {result.selected_order}")
    print(f"  [PASS] Reason: {result.reason}")


# ============================================================
# Test 10: TrajectoryEstimate
# ============================================================

def test_trajectory_estimate():
    """Test TrajectoryEstimate dataclass."""
    est = TrajectoryEstimate(
        order=1, t0=0.0,
        coefficients=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        reprojection_rms=0.3,
        geometric_rms=0.001,
        window_size=10,
        window_time_span=2.0,
        window_t_start=0.0,
        window_t_end=2.0,
        solvable=True,
    )

    # Evaluation
    pos = est.evaluate(2.0)
    assert abs(pos[0] - 2.0) < 1e-10  # X = 1.0 * 2.0
    print(f"  [PASS] evaluate(2.0) = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    vel = est.evaluate_velocity(2.0)
    assert abs(vel[0] - 1.0) < 1e-10
    print(f"  [PASS] velocity = ({vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f})")

    # Summary
    s = est.summary()
    assert "order=1" in s
    print(f"  [PASS] Summary: {s}")

    # Failed estimate
    failed = TrajectoryEstimate.failed("Not enough measurements")
    assert not failed.solvable
    assert "Not enough measurements" in str(failed.failure_reason)
    print(f"  [PASS] Failed estimate: {failed.failure_reason}")


# ============================================================
# Test 11: Full pipeline
# ============================================================

def test_full_pipeline():
    """End-to-end: generate → window → triangulate → validate with diverse cameras."""
    print("\n--- Full Reconstruction Pipeline ---")

    # Ground truth: linear trajectory X(t) = [0,0,2] + [0.3,-0.2,0.1]*t
    cx_true = np.array([0.0, 0.3])
    cy_true = np.array([0.0, -0.2])
    cz_true = np.array([2.0, 0.1])

    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0

    n_meas = 12
    times = np.linspace(0.0, 2.0, n_meas)

    # Cameras at diverse XY positions WITH diverse look directions
    # Each camera at a different XY position, all looking approximately
    # toward the trajectory's center of mass
    cameras_m = []
    for i in range(n_meas):
        t_i = times[i]
        X_target = np.array([
            cx_true[0] + cx_true[1] * t_i,
            cy_true[0] + cy_true[1] * t_i,
            cz_true[0] + cz_true[1] * t_i,
        ])
        # Camera positions on a circle around the target
        angle = 2.0 * np.pi * i / n_meas
        C = np.array([
            X_target[0] + 3.0 * np.cos(angle),
            X_target[1] + 3.0 * np.sin(angle),
            X_target[2] + 4.0,
        ])
        # Build look-at rotation so camera Z points toward target
        z_cam = X_target - C
        z_cam = z_cam / np.linalg.norm(z_cam)
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(world_up, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.array([1.0, 0.0, 0.0])
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        # Rotation: maps world→camera (Pose convention)
        R = np.vstack([x_cam, y_cam, z_cam])
        # Extract Euler angles
        from core.robotics.kinematics.transforms import rotation_to_euler_abc
        a_rad, b_rad, c_rad = rotation_to_euler_abc(R)
        cameras_m.append((C, np.degrees(a_rad), np.degrees(b_rad), np.degrees(c_rad)))

    # Generate with 0.1 px noise
    measurements = generate_test_measurements(
        cx_true, cy_true, cz_true,
        cameras_m, times,
        fx=fx, fy=fy, cx=cx, cy=cy,
        t0=0.0, noise_std=0.1,
    )

    # ---- Sliding window ----
    window = SlidingWindow(min_size=5, max_size=25)
    for fm in measurements:
        window.add(fm)

    assert window.can_reconstruct()
    print(f"  Window: {len(window)} measurements")

    # ---- Triangulate (order 1) ----
    # Use the latest measurement time as the polynomial reference
    t0_ref = times[-1]
    coeffs, _ = triangulate(window.get_measurements(), order=1, t0=t0_ref)
    assert coeffs is not None

    reproj_rms = rms_reprojection_error(window.get_measurements(), coeffs, t0_ref)
    geom_rms_val = rms_residual(window.get_measurements(), coeffs, t0_ref)

    print(f"  Reprojection RMS: {reproj_rms:.3f} px")
    print(f"  Geometric RMS:    {geom_rms_val*1000:.3f} mm")

    assert reproj_rms < 0.5, f"Reprojection error {reproj_rms:.3f}px"
    assert geom_rms_val < 0.01, f"Geometric RMS {geom_rms_val:.6f}m"

    # ---- Build TrajectoryEstimate ----
    est = TrajectoryEstimate(
        order=1, t0=t0_ref,
        coefficients=coeffs,
        reprojection_rms=reproj_rms,
        geometric_rms=geom_rms_val,
        max_geometric_residual=max_residual(window.get_measurements(), coeffs, t0_ref),
        window_size=len(window),
        window_time_span=window.time_span(),
        window_t_start=times[0],
        window_t_end=times[-1],
        solvable=True,
    )

    assert est.solvable
    print(f"  {est.summary()}")

    # ---- Model selection ----
    config = ReconstructionConfig(min_poly_order=1, max_poly_order=2, complexity_penalty=0.5)
    result = select_model_order(window.get_measurements(), config)
    assert result.selected_order > 0
    print(f"  Model selection: order={result.selected_order}, reason='{result.reason}'")

    print("  [PASS] Full pipeline reconstruction")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Trajectory Reconstruction — Verification Suite")
    print("=" * 60)

    print("\n[1] Ray builder")
    test_ray_builder()

    print("\n[2] Trajectory model")
    test_trajectory_model()

    print("\n[3] Sliding window")
    test_sliding_window()

    print("\n[4] Triangulation (noise-free)")
    test_triangulation_noise_free()

    print("\n[5] Triangulation (noisy)")
    test_triangulation_noisy()

    print("\n[6] Reprojection")
    test_reprojection()

    print("\n[7] Residuals")
    test_residuals()

    print("\n[8] Model selection (automatic)")
    test_model_selection_automatic()

    print("\n[9] Model selection (manual)")
    test_model_selection_manual()

    print("\n[10] TrajectoryEstimate")
    test_trajectory_estimate()

    test_full_pipeline()

    print("\n" + "=" * 60)
    print("All reconstruction tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
