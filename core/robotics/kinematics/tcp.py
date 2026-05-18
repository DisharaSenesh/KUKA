"""
Tool Center Point (TCP) configuration.

Defines the static transform from the robot flange to the tool/camera
mounted on the end-effector. This calibration is performed once and
remains constant for a given physical setup.

Transform chain:
  Flange → TCP → Camera

The TCP is the effective tool operating point. The camera is mounted
at a known offset from the TCP.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TCPConfig:
    """
    Static tool and camera mounting calibration.

    Attributes
    ----------
    flange_to_tcp : np.ndarray, shape (3,)
        Translation from flange origin to TCP in meters (flange frame).
    tcp_to_camera_R : np.ndarray, shape (3, 3)
        Rotation from camera frame to TCP frame.
        Default is identity (camera axes aligned with TCP axes).
    tcp_to_camera_t : np.ndarray, shape (3,)
        Translation from camera origin to TCP in meters (TCP frame).
        Default is zero (camera at TCP).
    description : str
        Human-readable label for this tool configuration.
    """

    flange_to_tcp: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    tcp_to_camera_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    tcp_to_camera_t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    description: str = "default"

    def __post_init__(self):
        self.flange_to_tcp = np.asarray(self.flange_to_tcp, dtype=np.float64)
        self.tcp_to_camera_R = np.asarray(self.tcp_to_camera_R, dtype=np.float64)
        self.tcp_to_camera_t = np.asarray(self.tcp_to_camera_t, dtype=np.float64)

    @classmethod
    def from_camera_offset(
        cls,
        camera_offset_mm: tuple,
        camera_rotation_deg: tuple = (0.0, 0.0, 0.0),
        description: str = "camera_tool",
    ) -> "TCPConfig":
        """
        Build a TCP configuration from a camera offset relative to the flange.

        This is the common case: the camera is rigidly mounted to the robot
        flange (or a tool bracket), and the TCP is at the camera optical center.

        Parameters
        ----------
        camera_offset_mm : tuple of (dx, dy, dz)
            Camera offset in millimeters from flange to camera center
            (expressed in flange frame).
        camera_rotation_deg : tuple of (rx, ry, rz)
            Camera rotation relative to flange, in degrees, following
            KUKA ABC convention (intrinsic Z-Y-X). Default is no rotation.
        description : str
            Label for this configuration.

        Returns
        -------
        TCPConfig
            Configured tool with camera at specified offset.
        """
        from .transforms import euler_abc_to_rotation

        # Camera position in flange frame (convert mm → m)
        flange_to_camera = np.array(camera_offset_mm, dtype=np.float64) * 0.001

        # Camera-to-flange rotation
        rx_rad = np.radians(camera_rotation_deg[0])
        ry_rad = np.radians(camera_rotation_deg[1])
        rz_rad = np.radians(camera_rotation_deg[2])

        # Note: the ABC angles define the camera orientation in flange frame
        R_camera_in_flange = euler_abc_to_rotation(rz_rad, ry_rad, rx_rad)

        # We store: TCP = camera center (so flange_to_tcp = flange_to_camera)
        # and TCP-to-camera is identity (they coincide)
        return cls(
            flange_to_tcp=flange_to_camera.copy(),
            tcp_to_camera_R=R_camera_in_flange.T.copy(),
            tcp_to_camera_t=np.zeros(3, dtype=np.float64),
            description=description,
        )

    def __repr__(self) -> str:
        offset_mm = self.flange_to_tcp * 1000.0
        return (
            f"TCPConfig(description='{self.description}', "
            f"offset_mm=({offset_mm[0]:.1f}, {offset_mm[1]:.1f}, {offset_mm[2]:.1f}))"
        )
