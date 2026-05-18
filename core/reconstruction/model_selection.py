"""
Adaptive model selection: automatically choose the best polynomial order.

Evaluates candidate polynomial orders (1, 2, 3, ...) on the current
window of filtered measurements and selects the one that best explains
the data while penalizing unnecessary complexity.

Selection criterion:
  Score(order) = RMS_reprojection_error + λ * order

where λ (complexity_penalty) penalizes higher model complexity.
Lower score is better.

Higher-order polynomials always fit the data better (lower reprojection
error), but they can overfit noisy monocular geometry. The penalty term
encourages choosing the simplest model that adequately explains the
observations.

If manual mode is active (config.polynomial_order is set), model selection
is bypassed and the fixed order is used directly.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .reconstruction_config import ReconstructionConfig
from .sliding_window import FilteredMeasurement
from .triangulation import triangulate
from .reprojection import rms_reprojection_error
from .residuals import rms_residual


@dataclass
class ModelSelectionResult:
    """
    Result of automatic model order evaluation.

    Attributes
    ----------
    selected_order : int
        The selected polynomial order (1 = linear, 2 = quadratic, etc.).
    candidates : dict
        Maps order → score for each evaluated candidate.
    winner_reprojection_rms : float
        RMS reprojection error (pixels) of the selected model.
    winner_geometric_rms : float
        RMS geometric residual (meters) of the selected model.
    reason : str
        Explanation of why this order was chosen.
    """

    selected_order: int
    candidates: dict
    winner_reprojection_rms: float
    winner_geometric_rms: float
    reason: str


def select_model_order(
    filtered_measurements: List[FilteredMeasurement],
    config: ReconstructionConfig,
    t0: Optional[float] = None,
) -> ModelSelectionResult:
    """
    Select the best polynomial order for the current measurement set.

    If config.polynomial_order is set (manual mode), returns that order
    directly without evaluating alternatives.

    In automatic mode, evaluates orders from config.min_poly_order to
    config.max_poly_order and selects the one with the lowest score:
      Score = RMS_reprojection + complexity_penalty * order

    Parameters
    ----------
    filtered_measurements : list of FilteredMeasurement
        Valid measurements in the sliding window.
    config : ReconstructionConfig
        Configuration with model selection parameters.
    t0 : float or None
        Reference time. Uses latest measurement time if None.

    Returns
    -------
    ModelSelectionResult
        Selected order, scores, and diagnostics.
    """
    # Manual mode: fixed order
    if config.is_manual_mode():
        order = config.polynomial_order
        coeffs, _ = triangulate(filtered_measurements, order, t0)
        if t0 is None and len(filtered_measurements) > 0:
            t0 = filtered_measurements[-1].synchronized.timestamp

        reproj_rms = float("inf")
        geom_rms = float("inf")
        if coeffs is not None:
            reproj_rms = rms_reprojection_error(filtered_measurements, coeffs, t0)
            geom_rms = rms_residual(filtered_measurements, coeffs, t0)

        return ModelSelectionResult(
            selected_order=order,
            candidates={order: np.inf},
            winner_reprojection_rms=reproj_rms,
            winner_geometric_rms=geom_rms,
            reason=f"Manual mode: fixed order = {order}",
        )

    # Automatic mode: evaluate candidates
    if t0 is None and len(filtered_measurements) > 0:
        t0 = filtered_measurements[-1].synchronized.timestamp

    candidates = {}
    best_order = None
    best_score = float("inf")
    best_reproj = float("inf")
    best_geom = float("inf")

    for order in range(config.min_poly_order, config.max_poly_order + 1):
        min_required = order + 1
        if len(filtered_measurements) < min_required:
            continue

        coeffs, _ = triangulate(filtered_measurements, order, t0)
        if coeffs is None:
            continue

        # Compute reprojection cost
        reproj_rms = rms_reprojection_error(filtered_measurements, coeffs, t0)

        # Score: reprojection quality + complexity penalty
        # Higher-order models can overfit noise in monocular geometry
        score = reproj_rms + config.complexity_penalty * float(order)

        candidates[order] = score

        if score < best_score:
            best_score = score
            best_order = order
            best_reproj = reproj_rms
            best_geom = rms_residual(filtered_measurements, coeffs, t0)

    if best_order is None:
        # Fallback: use minimum order
        best_order = config.min_poly_order
        return ModelSelectionResult(
            selected_order=best_order,
            candidates=candidates,
            winner_reprojection_rms=float("inf"),
            winner_geometric_rms=float("inf"),
            reason="No candidates evaluated successfully — using minimum order.",
        )

    # Determine why this order was chosen
    if len(candidates) == 1:
        reason = f"Only one viable order ({best_order})"
    else:
        # Find runner-up to explain
        others = {o: s for o, s in candidates.items() if o != best_order}
        if others:
            runner_up = min(others, key=others.get)
            reason = (
                f"Order {best_order} scored {best_score:.3f} "
                f"(vs order {runner_up} at {others[runner_up]:.3f}). "
                f"Penalty={config.complexity_penalty}"
            )
        else:
            reason = f"Order {best_order} selected."

    return ModelSelectionResult(
        selected_order=best_order,
        candidates=candidates,
        winner_reprojection_rms=best_reproj,
        winner_geometric_rms=best_geom,
        reason=reason,
    )
