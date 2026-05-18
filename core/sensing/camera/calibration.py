"""Checkerboard camera calibration.

Estimates the intrinsic matrix K and distortion coefficients
from a set of images of a known planar checkerboard target.

Uses OpenCV's Zhang calibration method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import cv2

from .intrinsics import CameraIntrinsics


@dataclass
class CalibrationResult:
    """Output of a checkerboard calibration run.

    Attributes
    ----------
    intrinsics : CameraIntrinsics
        Estimated camera intrinsics.
    rms_error : float
        Root-mean-square reprojection error in pixels.
    image_size : tuple of int
        (width, height) of the calibration images.
    n_images_used : int
        Number of calibration images that contributed.
    per_view_errors : np.ndarray or None
        Reprojection error per view, shape (N,).  None if
        fewer than 2 views were used.
    """

    intrinsics: CameraIntrinsics
    rms_error: float
    image_size: Tuple[int, int]
    n_images_used: int
    per_view_errors: Optional[np.ndarray] = None


def calibrate_checkerboard(
    images: List[np.ndarray],
    board_size: Tuple[int, int],
    square_size_mm: float = 1.0,
) -> CalibrationResult:
    """Estimate camera intrinsics from checkerboard images.

    Parameters
    ----------
    images : list of np.ndarray
        Calibration images.  All must have the same dimensions.
    board_size : (int, int)
        Number of *inner* corners: (columns, rows).
        For a board with 10×7 squares, use (9, 6).
    square_size_mm : float
        Physical size of one square, used only for extrinsics
        (not needed for intrinsics). Default 1.0.

    Returns
    -------
    CalibrationResult
        Estimated intrinsics and quality metrics.

    Raises
    ------
    ValueError
        If fewer than 3 usable views are found.
    """
    if len(images) < 3:
        raise ValueError(
            f"need at least 3 calibration images, got {len(images)}"
        )

    cols, rows = board_size
    h, w = images[0].shape[:2]

    # 3D object points in board-local coordinates (z = 0)
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    )

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []

    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows), None
        )
        if not found:
            continue

        # Refine to sub-pixel accuracy
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners_sub)

    if len(obj_points) < 3:
        raise ValueError(
            f"only {len(obj_points)} usable views found; need at least 3"
        )

    # Run Zhang calibration
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    # Per-view reprojection errors
    per_view: List[float] = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], K, dist
        )
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        per_view.append(err)

    intrinsics = CameraIntrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        distortion=dist.ravel() if dist is not None else None,
    )

    return CalibrationResult(
        intrinsics=intrinsics,
        rms_error=float(rms),
        image_size=(w, h),
        n_images_used=len(obj_points),
        per_view_errors=np.array(per_view, dtype=np.float64),
    )


def undistort_points(
    points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Compensate lens distortion for a set of pixel coordinates.

    Parameters
    ----------
    points : np.ndarray
        Distorted pixel coordinates, shape (N, 2) or (N, 1, 2).
    intrinsics : CameraIntrinsics
        Camera model including distortion coefficients.

    Returns
    -------
    np.ndarray
        Undistorted pixel coordinates, same shape as input.
    """
    if intrinsics.distortion is None:
        return points.copy()

    pts = np.asarray(points, dtype=np.float64)
    K = intrinsics.camera_matrix
    dist = intrinsics.distortion

    # OpenCV undistortPoints expects (N, 1, 2)
    if pts.ndim == 2:
        pts = pts[:, np.newaxis, :]
        squeezed = True
    else:
        squeezed = False

    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return undistorted[:, 0, :] if squeezed else undistorted
