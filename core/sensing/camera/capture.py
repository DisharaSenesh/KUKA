"""Camera acquisition — opens a video source and produces Frames.

CameraCapture is a small stateful wrapper around OpenCV's
VideoCapture.  It assigns monotonic frame IDs and timestamps
to every grabbed frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from .frame import Frame


@dataclass
class CameraCapture:
    """Stateful camera / video-file acquisition.

    Attributes
    ----------
    source : int or str
        Camera index (0, 1, …) for a USB / built-in camera,
        or a file path for a pre-recorded video.
    _cap : cv2.VideoCapture or None
        Underlying OpenCV capture object.  Created in ``open``.
    _frame_id : int
        Monotonically increasing frame counter.
    _start_time : float
        Acquisition epoch (monotonic seconds).
    """

    source: int | str = 0
    _cap: Optional[cv2.VideoCapture] = field(default=None, init=False, repr=False)
    _frame_id: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the video source.

        Returns
        -------
        bool
            True if the source was opened successfully.
        """
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._frame_id = 0
        self._start_time = time.monotonic()
        return True

    def release(self) -> None:
        """Release the underlying capture object."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Return whether the video source is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> Optional[Frame]:
        """Grab and decode the next frame.

        Returns
        -------
        Frame or None
            The next frame, or None if the stream ended.
        """
        if self._cap is None:
            return None

        ok, image = self._cap.read()
        if not ok:
            return None

        # Timestamp captured immediately after frame acquisition
        ts = time.monotonic() - self._start_time

        frame = Frame(
            image=image,
            timestamp=ts,
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        return frame

    def read_undistorted(
        self,
        intrinsics: "CameraIntrinsics",
    ) -> Optional[Frame]:
        """Read a frame and remove lens distortion inline.

        Parameters
        ----------
        intrinsics : CameraIntrinsics
            Camera model with distortion coefficients.

        Returns
        -------
        Frame or None
            Undistorted frame, or None if the stream ended.
        """
        frame = self.read()
        if frame is None:
            return None

        if intrinsics.distortion is not None:
            K = intrinsics.camera_matrix
            dist = intrinsics.distortion
            h, w = frame.image.shape[:2]
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            undistorted = cv2.undistort(frame.image, K, dist, None, new_K)
        else:
            undistorted = frame.image

        return Frame(
            image=undistorted,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        """Frame width in pixels (property of the opened stream)."""
        if self._cap is None:
            return 0
        return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self) -> float:
        """Frame height in pixels (property of the opened stream)."""
        if self._cap is None:
            return 0
        return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self) -> float:
        """Nominal frame rate reported by the source."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Total number of frames (estimated for video files)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        if not self.is_opened():
            self.open()
        return self

    def __exit__(self, *args) -> None:
        self.release()
