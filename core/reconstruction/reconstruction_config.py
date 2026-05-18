"""
Reconstruction configuration: parameters controlling trajectory estimation.

All numeric values use the robot's native units (mm for position,
degrees for Euler angles) at the I/O boundary, and converted to
meters/radians internally in the geometry layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReconstructionConfig:
    """
    Configuration for the trajectory reconstruction stage.

    Attributes
    ----------
    min_window_size : int
        Minimum measurements required in the sliding window before
        attempting reconstruction. Below this, geometry is too weak.
    max_window_size : int
        Maximum measurements retained. Oldest are evicted.
    polynomial_order : Optional[int]
        If set, use this fixed order (1=linear, 2=quadratic, 3=cubic).
        If None, enable automatic model selection.
    min_poly_order : int
        Minimum polynomial order considered during automatic selection.
    max_poly_order : int
        Maximum polynomial order considered during automatic selection.
    complexity_penalty : float
        Penalty weight λ per additional polynomial order during model
        selection. Higher values favor simpler trajectories.
    sync_tolerance_s : float
        Maximum sync error to consider a measurement valid for reconstruction.
    min_ray_angle_rad : float
        Minimum angular separation between ray directions within the window.
        Below this threshold, reconstruction is skipped (degenerate).
    use_weighted : bool
        If True, weight measurements by inverse sync error.
    """

    min_window_size: int = 5
    max_window_size: int = 50
    polynomial_order: Optional[int] = None
    min_poly_order: int = 1
    max_poly_order: int = 3
    complexity_penalty: float = 0.5
    sync_tolerance_s: float = 0.1
    min_ray_angle_rad: float = 0.017
    use_weighted: bool = False

    def is_manual_mode(self) -> bool:
        """True if a fixed polynomial order is enforced."""
        return self.polynomial_order is not None
