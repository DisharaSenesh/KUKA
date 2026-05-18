"""
Runtime pipeline runners for trajectory estimation.

OfflinePipeline: batch process a pre-recorded sequence.
OnlinePipeline: incremental, frame-by-frame processing.

These classes handle ONLY data flow. They delegate all computation to:
  - geometry: pixel → ray conversion
  - tracking: measurement buffering, system building, solving
  - prediction: future trajectory extrapolation
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.types.pose import Pose
from core.types.measurement import Measurement, PixelObservation
from core.types.trajectory_state import TrajectoryState
from core.geometry.backprojection import pixel_to_world_ray
from core.tracking.sliding_window import SlidingWindowTracker, TrackerConfig
from core.prediction.extrapolator import predict_position, predict_trajectory


@dataclass
class PipelineResult:
    """
    Result from a pipeline run.

    Attributes
    ----------
    state : Optional[TrajectoryState]
        Final trajectory estimate.
    position_history : List[np.ndarray]
        Estimated 3D positions at each observation time.
    time_history : List[float]
        Timestamps corresponding to position estimates.
    prediction : Optional[np.ndarray]
        Predicted future trajectory samples, shape (n_samples, 3).
    prediction_times : Optional[np.ndarray]
        Times corresponding to prediction samples.
    """
    state: Optional[TrajectoryState] = None
    position_history: List[np.ndarray] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)
    prediction: Optional[np.ndarray] = None
    prediction_times: Optional[np.ndarray] = None


@dataclass
class OfflinePipeline:
    """
    Batch pipeline: process an entire sequence of observations at once.

    Suitable for post-processing recorded data.

    Attributes
    ----------
    tracker : SlidingWindowTracker
        The sliding window tracker instance.
    """

    tracker: SlidingWindowTracker = field(default_factory=SlidingWindowTracker)

    def process_sequence(
        self,
        pixel_observations: List[PixelObservation],
        poses: List[Pose],
        times: List[float],
        predict_ahead: float = 0.0,
        prediction_samples: int = 50
    ) -> PipelineResult:
        """
        Process a complete sequence of observations.

        Parameters
        ----------
        pixel_observations : list of PixelObservation
            Raw pixel observations with intrinsics.
        poses : list of Pose
            Camera poses for each observation.
        times : list of float
            Timestamps for each observation.
        predict_ahead : float
            If > 0, predict trajectory this many seconds into the future.
        prediction_samples : int
            Number of samples for the predicted trajectory.

        Returns
        -------
        PipelineResult
            Final trajectory estimate, position history, and prediction.
        """
        result = PipelineResult()

        for pixel, pose, t in zip(pixel_observations, poses, times):
            # Convert pixel to world-frame ray
            ray = pixel_to_world_ray(
                u=pixel.u, v=pixel.v,
                fx=pixel.fx, fy=pixel.fy,
                cx=pixel.cx, cy=pixel.cy,
                pose=pose
            )

            # Create measurement
            measurement = Measurement(ray=ray, t=t, pixel=pixel)

            # Add to tracker
            self.tracker.add_measurement(measurement)

            # Solve if possible
            if self.tracker.can_solve():
                state = self.tracker.solve()
                position = state.evaluate(t)
            else:
                # Not enough data yet; use last known position or zero
                if self.tracker.state is not None:
                    position = self.tracker.state.evaluate(t)
                else:
                    position = np.zeros(3, dtype=np.float64)

            result.position_history.append(position.copy())
            result.time_history.append(t)

        # Store final state
        result.state = self.tracker.state

        # Generate future prediction if requested
        if predict_ahead > 0 and result.state is not None:
            t_last = times[-1]
            result.prediction_times = np.linspace(
                t_last, t_last + predict_ahead, prediction_samples
            )
            result.prediction = predict_trajectory(
                state=result.state,
                t_start=t_last,
                t_end=t_last + predict_ahead,
                n_samples=prediction_samples
            )

        return result

    def reset(self) -> None:
        """Reset the pipeline for a new sequence."""
        self.tracker.reset()


@dataclass
class OnlinePipeline:
    """
    Online pipeline: process observations one frame at a time.

    Suitable for real-time streaming data.

    Attributes
    ----------
    tracker : SlidingWindowTracker
        The sliding window tracker.
    """

    tracker: SlidingWindowTracker = field(default_factory=SlidingWindowTracker)

    def add_observation(
        self,
        pixel: PixelObservation,
        pose: Pose,
        t: float
    ) -> Optional[np.ndarray]:
        """
        Process a single observation frame.

        Parameters
        ----------
        pixel : PixelObservation
            Raw pixel observation with intrinsics.
        pose : Pose
            Camera pose at observation time.
        t : float
            Timestamp.

        Returns
        -------
        np.ndarray or None
            Estimated 3D position at time t, or None if not yet solvable.
        """
        # Convert pixel to world-frame ray
        ray = pixel_to_world_ray(
            u=pixel.u, v=pixel.v,
            fx=pixel.fx, fy=pixel.fy,
            cx=pixel.cx, cy=pixel.cy,
            pose=pose
        )

        # Create and buffer measurement
        measurement = Measurement(ray=ray, t=t, pixel=pixel)
        self.tracker.add_measurement(measurement)

        # Solve if we have enough measurements
        if self.tracker.can_solve():
            self.tracker.solve()

        # Return current position estimate
        if self.tracker.state is not None:
            return self.tracker.state.evaluate(t)

        return None

    def get_prediction(
        self,
        t_future: float
    ) -> Optional[np.ndarray]:
        """
        Predict position at a future time.

        Parameters
        ----------
        t_future : float
            Future time for prediction.

        Returns
        -------
        np.ndarray or None
            Predicted 3D position, or None if no estimate exists.
        """
        if self.tracker.state is None:
            return None
        return predict_position(self.tracker.state, t_future)

    def reset(self) -> None:
        """Reset the pipeline."""
        self.tracker.reset()
