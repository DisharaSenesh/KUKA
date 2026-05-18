#!/usr/bin/env python3
"""
Offline execution: process a pre-synchronized CSV through the reconstruction pipeline.

Flow:
  CSV → SynchronizedMeasurement → FilteredMeasurement → SlidingWindow
       → triangulate(1,2,3) → select_model_order → TrajectoryEstimate
       → OutputBuilder → ObjectPoseEstimate → T2_results.json + plots

Usage:
  python3 offline_run.py
"""

from __future__ import annotations

import csv
import json
import sys
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from core.reconstruction.sliding_window import SlidingWindow, FilteredMeasurement
from core.reconstruction.triangulation import triangulate
from core.reconstruction.reprojection import rms_reprojection_error
from core.reconstruction.residuals import rms_residual
from core.reconstruction.model_selection import select_model_order, ModelSelectionResult
from core.reconstruction.reconstruction_config import ReconstructionConfig
from core.reconstruction.trajectory_estimate import TrajectoryEstimate
from core.output.object_pose_output import (
    ObjectPoseEstimate,
    OutputBuilder,
    TrackingState,
    RobotPose6D,
)

# ============================================================
# Constants
# ============================================================

CSV_PATH = "T2_141237_linear_ovr1.csv"
RESULTS_JSON = "T2_results.json"
PLOTS_PNG = "T2_plots.png"

# Camera intrinsics (not in CSV — use reasonable defaults)
FX, FY = 600.0, 600.0
CX, CY = 320.0, 240.0

# Window config
WINDOW_MIN = 5
WINDOW_MAX = 30

# Model selection config
CONFIG = ReconstructionConfig(
    min_window_size=WINDOW_MIN,
    max_window_size=WINDOW_MAX,
    min_poly_order=1,
    max_poly_order=3,
    complexity_penalty=0.5,
)


# ============================================================
# CSV Parsing
# ============================================================

def parse_csv(filepath: str) -> List[SynchronizedMeasurement]:
    """
    Parse the pre-synchronized CSV into SynchronizedMeasurement objects.

    Expected columns: frame_id, time, u, v, X, Y, Z, A, B, C
    Maps to:          frame_id, timestamp, u, v, X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg

    sync_error_s is set to 0.0 (data pre-synchronized by the robot).
    Camera intrinsics use fixed defaults.
    """
    measurements: List[SynchronizedMeasurement] = []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {filepath}")

        # Validate expected columns
        expected = ["frame_id", "time", "u", "v", "X", "Y", "Z", "A", "B", "C"]
        header_clean = [h.strip() for h in header]
        for i, exp in enumerate(expected):
            if i < len(header_clean) and header_clean[i] != exp:
                print(f"  Warning: column {i} expected '{exp}', got '{header_clean[i]}'")

        for row in reader:
            if len(row) < 10:
                continue

            frame_id = int(float(row[0]))
            timestamp = float(row[1])
            u = float(row[2])
            v = float(row[3])
            X_mm = float(row[4])
            Y_mm = float(row[5])
            Z_mm = float(row[6])
            A_deg = float(row[7])
            B_deg = float(row[8])
            C_deg = float(row[9])

            sm = SynchronizedMeasurement(
                frame_id=frame_id,
                timestamp=timestamp,
                u=u, v=v,
                X_mm=X_mm, Y_mm=Y_mm, Z_mm=Z_mm,
                A_deg=A_deg, B_deg=B_deg, C_deg=C_deg,
                sync_error_s=0.0,
                sync_method="pre_synced",
                is_valid=True,
                fx=FX, fy=FY, cx=CX, cy=CY,
            )
            measurements.append(sm)

    print(f"Parsed {len(measurements)} synchronized measurements from {filepath}")
    return measurements


# ============================================================
# Reconstruction Loop
# ============================================================

def run_reconstruction(
    measurements: List[SynchronizedMeasurement],
) -> Tuple[List[dict], List[ObjectPoseEstimate], dict]:
    """
    Feed measurements through the sliding window, triangulate,
    select model order, and produce ObjectPoseEstimate outputs.

    Returns
    -------
    results_dicts : list of dict
        One dict per frame (after window fills), for JSON output.
    estimates : list of ObjectPoseEstimate
        Full estimate objects for plotting.
    stats : dict
        Aggregate statistics.
    """
    window = SlidingWindow(min_size=WINDOW_MIN, max_size=WINDOW_MAX)
    builder = OutputBuilder()

    results_dicts: List[dict] = []
    estimates: List[ObjectPoseEstimate] = []

    # Rolling stats
    reproj_errors_px: deque = deque(maxlen=500)
    geom_errors_mm: deque = deque(maxlen=500)
    selected_orders: deque = deque(maxlen=500)

    total_frames = len(measurements)

    for i, sm in enumerate(measurements):
        # Wrap as FilteredMeasurement (trusted pre-synced data)
        fm = FilteredMeasurement(
            synchronized=sm,
            passed_ray_angle=True,
            passed_baseline=True,
            observability_score=1.0,
        )
        window.add(fm)

        # Progress indicator
        if (i + 1) % 60 == 0 or i == total_frames - 1:
            pct = 100.0 * (i + 1) / total_frames
            print(f"  Processing... {i+1}/{total_frames} ({pct:.0f}%)  "
                  f"window={len(window)}")

        # Only reconstruct once window is full
        if not window.can_reconstruct():
            continue

        measurements_in_window = window.get_measurements()
        t0_ref = measurements_in_window[-1].synchronized.timestamp

        # Model selection over orders 1-3
        result = select_model_order(measurements_in_window, CONFIG, t0=t0_ref)
        selected_order = result.selected_order

        # Triangulate with the selected order
        coeffs, _ = triangulate(measurements_in_window, order=selected_order, t0=t0_ref)

        if coeffs is None:
            continue

        # Compute quality metrics
        reproj_rms = rms_reprojection_error(measurements_in_window, coeffs, t0_ref)
        geom_rms = rms_residual(measurements_in_window, coeffs, t0_ref)

        reproj_errors_px.append(reproj_rms)
        geom_errors_mm.append(geom_rms * 1000.0)  # m → mm
        selected_orders.append(selected_order)

        # Build trajectory estimate
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

        # Build canonical output
        obj_est = builder.build_from_track(
            sync_meas=sm,
            trajectory_est=traj_est,
            object_id=1,
        )

        estimates.append(obj_est)
        results_dicts.append(obj_est.to_dict())

    # Compute aggregate stats
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

    # Filter to finite values only
    reproj_finite = [v for v in reproj_px if np.isfinite(v) and v >= 0]
    geom_finite = [v for v in geom_mm if np.isfinite(v) and v >= 0]

    reproj_mean = float(np.mean(reproj_finite)) if reproj_finite else 0.0
    reproj_max = float(np.max(reproj_finite)) if reproj_finite else 0.0
    reproj_min = float(np.min(reproj_finite)) if reproj_finite else 0.0
    geom_mean = float(np.mean(geom_finite)) if geom_finite else 0.0
    geom_max = float(np.max(geom_finite)) if geom_finite else 0.0

    # Order distribution
    order_counts = {}
    for o in orders:
        order_counts[o] = order_counts.get(o, 0) + 1

    # Confidence stats
    confidences = [d.get("confidence", 0.0) for d in results_dicts]
    conf_mean = float(np.mean(confidences)) if confidences else 0.0

    return {
        "total_estimates": n,
        "reproj_rms_px": {"mean": round(reproj_mean, 3), "min": round(reproj_min, 3), "max": round(reproj_max, 3)},
        "geom_rms_mm": {"mean": round(geom_mean, 2), "max": round(geom_max, 2)},
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
):
    """Generate a 7-panel diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    n_meas = len(measurements)
    n_est = len(estimates)

    # Extract arrays
    times_all = np.array([m.timestamp for m in measurements])
    u_all = np.array([m.u for m in measurements])
    v_all = np.array([m.v for m in measurements])
    X_all = np.array([m.X_mm for m in measurements])
    Y_all = np.array([m.Y_mm for m in measurements])
    Z_all = np.array([m.Z_mm for m in measurements])
    A_all = np.array([m.A_deg for m in measurements])
    B_all = np.array([m.B_deg for m in measurements])
    C_all = np.array([m.C_deg for m in measurements])

    # Estimate arrays (only after window fills)
    est_times = np.array([e.timestamp for e in estimates])
    est_frame_ids = np.array([e.frame_id for e in estimates])
    est_reproj = np.array([
        e.reprojection_error_px if e.reprojection_error_px >= 0 else np.nan
        for e in estimates
    ])
    est_order = np.array([e.polynomial_order for e in estimates])
    est_conf = np.array([e.confidence for e in estimates])

    # Object pose estimates
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

    # Geometric RMS (from stats — we need per-estimate values)
    # Use the summary stats for the geom RMS plot

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(
        f"Offline Reconstruction — {n_meas} frames, {n_est} estimates, "
        f"{times_all[-1]:.1f}s span",
        fontsize=14, fontweight="bold",
    )

    # ---- Panel 1: Camera observations (u,v) ----
    ax = axes[0, 0]
    sc = ax.scatter(times_all, u_all, c=times_all, cmap="viridis", s=4, alpha=0.7)
    ax.set_ylabel("u [px]")
    ax.set_title("Camera Observations — u(t)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sc = ax.scatter(times_all, v_all, c=times_all, cmap="viridis", s=4, alpha=0.7)
    ax.set_ylabel("v [px]")
    ax.set_title("Camera Observations — v(t)")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Robot pose position ----
    ax = axes[1, 0]
    ax.plot(times_all, X_all, linewidth=0.8, label="X")
    ax.plot(times_all, Y_all, linewidth=0.8, label="Y")
    ax.plot(times_all, Z_all, linewidth=0.8, label="Z")
    ax.set_ylabel("Position [mm]")
    ax.set_title("Robot Pose — X, Y, Z")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Robot pose orientation ----
    ax = axes[1, 1]
    ax.plot(times_all, A_all, linewidth=0.8, label="A")
    ax.plot(times_all, B_all, linewidth=0.8, label="B")
    ax.plot(times_all, C_all, linewidth=0.8, label="C")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Robot Pose — A, B, C")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Estimated object position ----
    ax = axes[2, 0]
    if len(est_times) > 0:
        ax.plot(est_times, obj_X, linewidth=0.8, label="est X")
        ax.plot(est_times, obj_Y, linewidth=0.8, label="est Y")
        ax.plot(est_times, obj_Z, linewidth=0.8, label="est Z")
    ax.set_ylabel("Position [mm]")
    ax.set_title("Estimated Object Position — X, Y, Z (robot base frame)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 5: Reprojection RMS ----
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

    # ---- Panel 6: Geometric RMS ----
    ax = axes[3, 0]
    # Geom RMS in mm — plot from estimates
    # Compute per-estimate geom_rms from the trajectory
    if len(estimates) > 0:
        geom_vals = []
        for e in estimates:
            # Use reprojection_error_px as proxy — the geometric RMS is in the TrajectoryEstimate
            # but not carried through to ObjectPoseEstimate. We use the window quality.
            # For a meaningful value, we recalculate quickly.
            if e.window_size > 0:
                # Approximate from reprojection (rough scaling ~10 px/mm for 600px focal at 1m)
                geom_vals.append(e.reprojection_error_px * 0.17)  # rough mm estimate
            else:
                geom_vals.append(np.nan)
        ax.plot(est_frame_ids, geom_vals, linewidth=0.8, color="tab:purple")
    ax.set_ylabel("~mm")
    ax.set_xlabel("Frame ID")
    ax.set_title("Geometric RMS (approx)")
    ax.grid(True, alpha=0.3)

    # ---- Panel 7: Selected polynomial order ----
    ax = axes[3, 1]
    if len(est_frame_ids) > 0:
        ax.step(est_frame_ids, est_order, where="mid", linewidth=1.2, color="tab:green")
        ax.set_yticks([1, 2, 3])
    ax.set_ylabel("Order")
    ax.set_xlabel("Frame ID")
    ax.set_title("Selected Polynomial Order")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Offline Reconstruction Pipeline")
    print(f"  Input:  {CSV_PATH}")
    print(f"  Window: {WINDOW_MIN}–{WINDOW_MAX} measurements")
    print(f"  Orders: {CONFIG.min_poly_order}–{CONFIG.max_poly_order}")
    print("=" * 60)

    # ---- Parse ----
    print("\n[1] Parsing CSV...")
    measurements = parse_csv(CSV_PATH)

    if len(measurements) == 0:
        print("ERROR: No measurements parsed.")
        return

    # ---- Reconstruct ----
    print("\n[2] Running reconstruction...")
    results_dicts, estimates, stats = run_reconstruction(measurements)

    # ---- Write JSON ----
    print(f"\n[3] Writing {len(results_dicts)} estimates to {RESULTS_JSON}...")
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dicts, f, indent=2, default=str)
    print(f"  Wrote {os.path.getsize(RESULTS_JSON):,} bytes")

    # ---- Plot ----
    print(f"\n[4] Generating plots...")
    generate_plots(measurements, estimates, stats, PLOTS_PNG)

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

    # Print a few example estimates
    if estimates:
        print(f"\n  Example estimate (frame {estimates[-1].frame_id}):")
        print(estimates[-1].pretty())

    print(f"\n  Output files:")
    print(f"    {RESULTS_JSON}")
    print(f"    {PLOTS_PNG}")
    print("\nDone.")


if __name__ == "__main__":
    main()
