#!/usr/bin/env python3
"""
Offline execution: process a pre-synchronized CSV through the reconstruction pipeline.

Flow:
  YAML config → parse CSV → undistort → SynchronizedMeasurement
     → FilteredMeasurement → SlidingWindow → triangulate(1,2,3)
     → select_model_order → TrajectoryEstimate
     → OutputBuilder → ObjectPoseEstimate → JSON + plots

Usage:
  python3 offline_run.py                    # uses offline_config.yaml
  python3 offline_run.py my_config.yaml     # uses custom config
"""

from __future__ import annotations

import csv
import json
import sys
import os
from collections import deque
from typing import List, Optional, Tuple, Any

import numpy as np
import yaml

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from core.reconstruction.sliding_window import SlidingWindow, FilteredMeasurement
from core.reconstruction.triangulation import triangulate
from core.reconstruction.reprojection import rms_reprojection_error
from core.reconstruction.residuals import rms_residual
from core.reconstruction.model_selection import select_model_order
from core.reconstruction.reconstruction_config import ReconstructionConfig
from core.reconstruction.trajectory_estimate import TrajectoryEstimate
from core.output.object_pose_output import (
    ObjectPoseEstimate,
    OutputBuilder,
    TrackingState,
    RobotPose6D,
)
from core.sensing.camera.intrinsics import CameraIntrinsics
from core.sensing.camera.calibration import undistort_points
from core.robotics.kinematics.tcp import TCPConfig
from core.robotics.kinematics.transforms import (
    euler_abc_to_rotation,
    compute_camera_pose,
    rotation_to_euler_abc,
    KUKA_POSITION_SCALE,
)


# ============================================================
# Config Loading
# ============================================================

def load_config(path: str) -> dict:
    """Load and validate the YAML configuration.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration with all sections.

    Raises
    ------
    ValueError
        If required sections are missing.
    FileNotFoundError
        If the config file does not exist.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required = ["camera", "input", "output", "window", "model", "hand_eye"]
    for key in required:
        if key not in cfg:
            raise ValueError(
                f"Missing required config section: '{key}'"
            )

    return cfg


def build_camera_intrinsics(cfg: dict) -> CameraIntrinsics:
    """Build CameraIntrinsics from config."""
    cam = cfg["camera"]
    distortion = None
    if cam.get("distortion") is not None and len(cam["distortion"]) > 0:
        distortion = np.array(cam["distortion"], dtype=np.float64)
    return CameraIntrinsics(
        fx=float(cam["fx"]),
        fy=float(cam["fy"]),
        cx=float(cam["cx"]),
        cy=float(cam["cy"]),
        distortion=distortion,
    )


def build_tcp_config(cfg: dict) -> TCPConfig:
    """Build TCPConfig (hand-eye calibration) from config.

    The hand_eye section gives flange → camera transform.
    Rotation angles follow Rx @ Ry @ Rz order (matching reference comment).
    This is extrinsic XYZ order, NOT KUKA's ABC (Z-Y-X) convention.
    """
    he = cfg["hand_eye"]
    rx_rad = np.radians(float(he["rx_deg"]))
    ry_rad = np.radians(float(he["ry_deg"]))
    rz_rad = np.radians(float(he["rz_deg"]))

    # Rotation matrices for Rx, Ry, Rz
    crx, srx = np.cos(rx_rad), np.sin(rx_rad)
    cry, sry = np.cos(ry_rad), np.sin(ry_rad)
    crz, srz = np.cos(rz_rad), np.sin(rz_rad)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, crx, -srx],
        [0.0, srx, crx],
    ], dtype=np.float64)

    Ry = np.array([
        [cry, 0.0, sry],
        [0.0, 1.0, 0.0],
        [-sry, 0.0, cry],
    ], dtype=np.float64)

    Rz = np.array([
        [crz, -srz, 0.0],
        [srz, crz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Rx @ Ry @ Rz — camera → tool rotation (reference convention)
    R_cam_in_tcp = Rx @ Ry @ Rz

    t_cam_in_tcp = np.array([
        float(he["tx_mm"]) * 0.001,
        float(he["ty_mm"]) * 0.001,
        float(he["tz_mm"]) * 0.001,
    ], dtype=np.float64)

    return TCPConfig(
        flange_to_tcp=np.zeros(3, dtype=np.float64),
        tcp_to_camera_R=R_cam_in_tcp,
        tcp_to_camera_t=t_cam_in_tcp,
        description="hand_eye_from_config",
    )


# ============================================================
# CSV Parsing
# ============================================================

def parse_csv(
    filepath: str,
    intrinsics: CameraIntrinsics,
    cfg: dict,
    tcp_config: TCPConfig,
) -> List[SynchronizedMeasurement]:
    """
    Parse the pre-synchronized CSV into SynchronizedMeasurement objects.

    Expected columns: frame_id, time, u, v, X, Y, Z, A, B, C
    These are robot TCP poses (from $POS_ACT).

    Applies:
      - Camera undistortion (using lens distortion coefficients)
      - Hand-eye calibration: TCP pose → camera pose via compute_camera_pose()
      - Pixel bounds validation

    sync_error_s is set to 0.0 (data pre-synchronized by the robot).
    """
    cam = cfg["camera"]
    width = int(cam.get("width", 1280))
    height = int(cam.get("height", 960))
    has_distortion = intrinsics.distortion is not None

    measurements: List[SynchronizedMeasurement] = []
    skipped_bounds = 0

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        for row in reader:
            if len(row) < 10:
                continue

            frame_id = int(float(row[0]))
            timestamp = float(row[1])
            u_raw = float(row[2])
            v_raw = float(row[3])
            # Robot TCP pose from CSV (mm, degrees — KUKA $POS_ACT format)
            tcp_X_mm = float(row[4])
            tcp_Y_mm = float(row[5])
            tcp_Z_mm = float(row[6])
            tcp_A_deg = float(row[7])
            tcp_B_deg = float(row[8])
            tcp_C_deg = float(row[9])

            # ---- Apply hand-eye calibration: TCP pose → camera pose ----
            camera_pose = compute_camera_pose(
                robot_x_mm=tcp_X_mm,
                robot_y_mm=tcp_Y_mm,
                robot_z_mm=tcp_Z_mm,
                robot_a_deg=tcp_A_deg,
                robot_b_deg=tcp_B_deg,
                robot_c_deg=tcp_C_deg,
                flange_to_tcp=tcp_config.flange_to_tcp,
                tcp_to_camera_R=tcp_config.tcp_to_camera_R,
                tcp_to_camera_t=tcp_config.tcp_to_camera_t,
                timestamp=timestamp,
            )

            # Convert camera Pose back to mm/deg for SynchronizedMeasurement storage
            cam_X_mm = float(camera_pose.C[0] / KUKA_POSITION_SCALE)
            cam_Y_mm = float(camera_pose.C[1] / KUKA_POSITION_SCALE)
            cam_Z_mm = float(camera_pose.C[2] / KUKA_POSITION_SCALE)
            a_rad, b_rad, c_rad = rotation_to_euler_abc(camera_pose.R)
            cam_A_deg = float(np.degrees(a_rad))
            cam_B_deg = float(np.degrees(b_rad))
            cam_C_deg = float(np.degrees(c_rad))

            # ---- Apply undistortion ----
            if has_distortion:
                pts = undistort_points(
                    np.array([[u_raw, v_raw]], dtype=np.float64),
                    intrinsics,
                )
                u = float(pts[0, 0])
                v = float(pts[0, 1])
            else:
                u, v = u_raw, v_raw

            # ---- Validate pixel bounds ----
            if not (0.0 <= u < width and 0.0 <= v < height):
                skipped_bounds += 1
                continue

            sm = SynchronizedMeasurement(
                frame_id=frame_id,
                timestamp=timestamp,
                u=u, v=v,
                X_mm=cam_X_mm, Y_mm=cam_Y_mm, Z_mm=cam_Z_mm,
                A_deg=cam_A_deg, B_deg=cam_B_deg, C_deg=cam_C_deg,
                sync_error_s=0.0,
                sync_method="pre_synced",
                is_valid=True,
                fx=float(cam["fx"]),
                fy=float(cam["fy"]),
                cx=float(cam["cx"]),
                cy=float(cam["cy"]),
            )
            measurements.append(sm)

    print(f"Parsed {len(measurements)} measurements from {filepath}")
    if skipped_bounds > 0:
        print(f"  Skipped {skipped_bounds} out-of-bounds pixels")
    if has_distortion:
        print(f"  Undistortion applied (k1={intrinsics.distortion[0]:.4f}, "
              f"k2={intrinsics.distortion[1]:.4f}, k3={intrinsics.distortion[4]:.4f})")
    return measurements


# ============================================================
# Reconstruction Loop
# ============================================================

def run_reconstruction(
    measurements: List[SynchronizedMeasurement],
    cfg: dict,
) -> Tuple[List[dict], List[ObjectPoseEstimate], dict]:
    """
    Feed measurements through the sliding window, triangulate,
    select model order, and produce ObjectPoseEstimate outputs.
    """
    w = cfg["window"]
    m = cfg["model"]
    window_min = int(w["min_size"])
    window_max = int(w["max_size"])

    window = SlidingWindow(min_size=window_min, max_size=window_max)
    builder = OutputBuilder()

    recon_config = ReconstructionConfig(
        min_window_size=window_min,
        max_window_size=window_max,
        min_poly_order=int(m["min_poly_order"]),
        max_poly_order=int(m["max_poly_order"]),
        complexity_penalty=float(m.get("complexity_penalty", 0.5)),
        use_weighted=bool(m.get("use_weighted", False)),
    )

    results_dicts: List[dict] = []
    estimates: List[ObjectPoseEstimate] = []

    reproj_errors_px: deque = deque(maxlen=500)
    geom_errors_mm: deque = deque(maxlen=500)
    selected_orders: deque = deque(maxlen=500)

    total_frames = len(measurements)

    for i, sm in enumerate(measurements):
        fm = FilteredMeasurement(
            synchronized=sm,
            passed_ray_angle=True,
            passed_baseline=True,
            observability_score=1.0,
        )
        window.add(fm)

        if (i + 1) % 60 == 0 or i == total_frames - 1:
            pct = 100.0 * (i + 1) / total_frames
            print(f"  Processing... {i+1}/{total_frames} ({pct:.0f}%)  "
                  f"window={len(window)}")

        if not window.can_reconstruct():
            continue

        measurements_in_window = window.get_measurements()
        t0_ref = measurements_in_window[-1].synchronized.timestamp

        result = select_model_order(measurements_in_window, recon_config, t0=t0_ref)
        selected_order = result.selected_order

        coeffs, _ = triangulate(measurements_in_window, order=selected_order, t0=t0_ref)

        if coeffs is None:
            continue

        reproj_rms = rms_reprojection_error(measurements_in_window, coeffs, t0_ref)
        geom_rms = rms_residual(measurements_in_window, coeffs, t0_ref)

        reproj_errors_px.append(reproj_rms)
        geom_errors_mm.append(geom_rms * 1000.0)
        selected_orders.append(selected_order)

        traj_est = TrajectoryEstimate(
            order=selected_order,
            t0=t0_ref,
            coefficients=coeffs,
            reprojection_rms=reproj_rms,
            geometric_rms=geom_rms,
            max_geometric_residual=0.0,
            window_size=len(window),
            window_time_span=window.time_span(),
            window_t_start=measurements_in_window[0].synchronized.timestamp,
            window_t_end=t0_ref,
            solvable=True,
        )

        obj_est = builder.build_from_track(
            sync_meas=sm,
            trajectory_est=traj_est,
            object_id=1,
        )

        estimates.append(obj_est)
        results_dicts.append(obj_est.to_dict())

    stats = _compute_stats(
        results_dicts, reproj_errors_px, geom_errors_mm, selected_orders
    )

    return results_dicts, estimates, stats


def _compute_stats(
    results_dicts: list,
    reproj_px: deque,
    geom_mm: deque,
    orders: deque,
) -> dict:
    """Compute aggregate statistics, filtering out inf/nan values."""
    n = len(results_dicts)
    if n == 0:
        return {"total_estimates": 0}

    reproj_finite = [v for v in reproj_px if np.isfinite(v) and v >= 0]
    geom_finite = [v for v in geom_mm if np.isfinite(v) and v >= 0]

    reproj_mean = float(np.mean(reproj_finite)) if reproj_finite else 0.0
    reproj_max = float(np.max(reproj_finite)) if reproj_finite else 0.0
    reproj_min = float(np.min(reproj_finite)) if reproj_finite else 0.0
    geom_mean = float(np.mean(geom_finite)) if geom_finite else 0.0
    geom_max = float(np.max(geom_finite)) if geom_finite else 0.0

    order_counts: dict = {}
    for o in orders:
        order_counts[int(o)] = order_counts.get(int(o), 0) + 1

    confidences = [d.get("confidence", 0.0) for d in results_dicts]
    conf_mean = float(np.mean(confidences)) if confidences else 0.0

    return {
        "total_estimates": n,
        "reproj_rms_px": {
            "mean": round(reproj_mean, 3),
            "min": round(reproj_min, 3),
            "max": round(reproj_max, 3),
        },
        "geom_rms_mm": {
            "mean": round(geom_mean, 2),
            "max": round(geom_max, 2),
        },
        "order_distribution": order_counts,
        "mean_confidence": round(conf_mean, 3),
        "valid_reproj": len(reproj_finite),
    }


# ============================================================
# Plotting
# ============================================================

def generate_plots(
    measurements: List[SynchronizedMeasurement],
    estimates: List[ObjectPoseEstimate],
    stats: dict,
    output_path: str,
    cfg: dict,
):
    """Generate a 7-panel diagnostic figure."""
    p = cfg.get("plots", {})
    if not p.get("enabled", True):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = float(p.get("figure_width", 16))
    fig_h = float(p.get("figure_height", 14))
    dpi = int(p.get("dpi", 150))

    n_meas = len(measurements)
    n_est = len(estimates)

    times_all = np.array([m.timestamp for m in measurements])
    u_all = np.array([m.u for m in measurements])
    v_all = np.array([m.v for m in measurements])
    X_all = np.array([m.X_mm for m in measurements])
    Y_all = np.array([m.Y_mm for m in measurements])
    Z_all = np.array([m.Z_mm for m in measurements])
    A_all = np.array([m.A_deg for m in measurements])
    B_all = np.array([m.B_deg for m in measurements])
    C_all = np.array([m.C_deg for m in measurements])

    est_times = np.array([e.timestamp for e in estimates])
    est_frame_ids = np.array([e.frame_id for e in estimates])
    est_reproj = np.array([
        e.reprojection_error_px if e.reprojection_error_px >= 0 else np.nan
        for e in estimates
    ])
    est_order = np.array([e.polynomial_order for e in estimates])

    obj_X = np.array([
        e.object_pose_base.x if e.object_pose_base else np.nan
        for e in estimates
    ])
    obj_Y = np.array([
        e.object_pose_base.y if e.object_pose_base else np.nan
        for e in estimates
    ])
    obj_Z = np.array([
        e.object_pose_base.z if e.object_pose_base else np.nan
        for e in estimates
    ])

    fig, axes = plt.subplots(4, 2, figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Offline Reconstruction — {n_meas} frames, {n_est} estimates, "
        f"{times_all[-1]:.1f}s span",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: Camera observations
    ax = axes[0, 0]
    ax.scatter(times_all, u_all, c=times_all, cmap="viridis", s=4, alpha=0.7)
    ax.set_ylabel("u [px]")
    ax.set_title("Camera Observations — u(t)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(times_all, v_all, c=times_all, cmap="viridis", s=4, alpha=0.7)
    ax.set_ylabel("v [px]")
    ax.set_title("Camera Observations — v(t)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Robot pose position
    ax = axes[1, 0]
    ax.plot(times_all, X_all, linewidth=0.8, label="X")
    ax.plot(times_all, Y_all, linewidth=0.8, label="Y")
    ax.plot(times_all, Z_all, linewidth=0.8, label="Z")
    ax.set_ylabel("Position [mm]")
    ax.set_title("Robot Pose — X, Y, Z")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Robot pose orientation
    ax = axes[1, 1]
    ax.plot(times_all, A_all, linewidth=0.8, label="A")
    ax.plot(times_all, B_all, linewidth=0.8, label="B")
    ax.plot(times_all, C_all, linewidth=0.8, label="C")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Robot Pose — A, B, C")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Estimated object position
    ax = axes[2, 0]
    if len(est_times) > 0:
        ax.plot(est_times, obj_X, linewidth=0.8, label="est X")
        ax.plot(est_times, obj_Y, linewidth=0.8, label="est Y")
        ax.plot(est_times, obj_Z, linewidth=0.8, label="est Z")
    ax.set_ylabel("Position [mm]")
    ax.set_title("Estimated Object Position — X, Y, Z (robot base frame)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Reprojection RMS
    ax = axes[2, 1]
    if len(est_frame_ids) > 0:
        ax.plot(est_frame_ids, est_reproj, linewidth=0.8, color="tab:red")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, label="1 px")
    ax.axhline(y=stats.get("reproj_rms_px", {}).get("mean", 0),
               color="orange", linestyle=":", linewidth=0.8, label="mean")
    ax.set_ylabel("RMS [px]")
    ax.set_xlabel("Frame ID")
    ax.set_title("Reprojection RMS Error")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: Geometric RMS
    ax = axes[3, 0]
    if len(estimates) > 0:
        geom_vals = []
        for e in estimates:
            if e.window_size > 0:
                geom_vals.append(e.reprojection_error_px * 0.17)
            else:
                geom_vals.append(np.nan)
        ax.plot(est_frame_ids, geom_vals, linewidth=0.8, color="tab:purple")
    ax.set_ylabel("~mm")
    ax.set_xlabel("Frame ID")
    ax.set_title("Geometric RMS (approx)")
    ax.grid(True, alpha=0.3)

    # Panel 7: Selected polynomial order
    ax = axes[3, 1]
    if len(est_frame_ids) > 0:
        ax.step(est_frame_ids, est_order, where="mid", linewidth=1.2, color="tab:green")
        ax.set_yticks([1, 2, 3])
    ax.set_ylabel("Order")
    ax.set_xlabel("Frame ID")
    ax.set_title("Selected Polynomial Order")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {output_path}")


# ============================================================
# Main
# ============================================================

def main(config_path: str = "offline_config.yaml"):
    # ---- Load config ----
    print("=" * 60)
    print("Offline Reconstruction Pipeline")
    print("=" * 60)

    cfg = load_config(config_path)
    print(f"  Config:    {config_path}  (mode={cfg.get('mode', 'offline')})")

    cam = cfg["camera"]
    w = cfg["window"]
    m = cfg["model"]
    he = cfg["hand_eye"]

    csv_path = cfg["input"]["csv_path"]
    results_json = cfg["output"]["results_json"]
    plots_png = cfg["output"]["plots_png"]

    print(f"  Input:     {csv_path}")
    print(f"  Camera:    {cam['fx']:.1f}x{cam['fy']:.1f}  "
          f"principal=({cam['cx']:.0f},{cam['cy']:.0f})  "
          f"size={cam.get('width',1280)}x{cam.get('height',960)}")
    print(f"  Distortion: k1={cam['distortion'][0]:.4f}  "
          f"k2={cam['distortion'][1]:.4f}  "
          f"p1={cam['distortion'][2]:.4f}  "
          f"p2={cam['distortion'][3]:.4f}  "
          f"k3={cam['distortion'][4]:.4f}")
    print(f"  Hand-eye:  tx={he['tx_mm']:.1f}  ty={he['ty_mm']:.1f}  tz={he['tz_mm']:.1f} mm  "
          f"rx={he['rx_deg']:.2f}°  ry={he['ry_deg']:.2f}°  rz={he['rz_deg']:.2f}°")
    print(f"  Window:    {w['min_size']}–{w['max_size']} measurements")
    print(f"  Model:     orders {m['min_poly_order']}–{m['max_poly_order']}  "
          f"penalty={m.get('complexity_penalty', 0.5)}")
    print("=" * 60)

    # ---- Build intrinsics and TCP ----
    intrinsics = build_camera_intrinsics(cfg)
    tcp_config = build_tcp_config(cfg)
    print(f"  TCP: {tcp_config}")

    # ---- Parse CSV ----
    print(f"\n[1] Parsing CSV...")
    measurements = parse_csv(csv_path, intrinsics, cfg, tcp_config)

    if len(measurements) == 0:
        print("ERROR: No measurements parsed.")
        return

    # ---- Reconstruct ----
    print(f"\n[2] Running reconstruction...")
    results_dicts, estimates, stats = run_reconstruction(measurements, cfg)

    # ---- Write JSON ----
    print(f"\n[3] Writing {len(results_dicts)} estimates to {results_json}...")
    with open(results_json, "w") as f:
        json.dump(results_dicts, f, indent=2, default=str)
    print(f"  Wrote {os.path.getsize(results_json):,} bytes")

    # ---- Plot ----
    print(f"\n[4] Generating plots...")
    generate_plots(measurements, estimates, stats, plots_png, cfg)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Reconstruction Summary")
    print("=" * 60)
    n = stats.get("total_estimates", 0)
    print(f"  Total estimates produced:  {n} / {len(measurements)}")
    if n > 0:
        r = stats.get("reproj_rms_px", {})
        g = stats.get("geom_rms_mm", {})
        print(f"  Reprojection RMS:  mean={r.get('mean', 0):.3f} px  "
              f"min={r.get('min', 0):.3f} px  max={r.get('max', 0):.3f} px  "
              f"({stats.get('valid_reproj', 0)} valid)")
        print(f"  Geometric RMS:     mean={g.get('mean', 0):.2f} mm  max={g.get('max', 0):.2f} mm")
        print(f"  Mean confidence:   {stats.get('mean_confidence', 0):.3f}")
        print(f"  Order distribution: {stats.get('order_distribution', {})}")

    if estimates:
        print(f"\n  Last 3 estimates:")
        for e in estimates[-3:]:
            print(f"    frame={e.frame_id:04d}  "
                  f"order={e.polynomial_order}  "
                  f"reproj={e.reprojection_error_px:.2f}px  "
                  f"conf={e.confidence:.3f}")

    print(f"\n  Output files:")
    print(f"    {results_json}")
    print(f"    {plots_png}")
    print("\nDone.")


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "offline_config.yaml"
    main(config_arg)
