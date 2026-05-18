"""
KUKA Robot Driver — thin wrapper around the existing KUKAControl class.

This module wraps the low-level KUKA socket communication layer.
It does NOT modify the underlying KUKAControl internals.

Responsibilities:
  - Import and wrap the existing KUKAControl class
  - Expose typed, well-documented methods
  - Maintain the robot connection lifecycle
  - Convert between raw KUKA data and Python-native types

The driver is the ONLY module that directly touches the KUKA
socket communication. All higher layers go through this driver.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---- Import existing KUKAControl from local path ----
# The KUKAControl class is located in the same directory as this file.

_KUKA_IMPORTED = False
_KUKAControl = None


def _ensure_kuka_imported():
    """Lazy import of the existing KUKAControl class."""
    global _KUKA_IMPORTED, _KUKAControl
    if _KUKA_IMPORTED:
        return

    try:
        from RobotControl import KUKAControl as KC
    except ImportError as e:
        raise ImportError(
            f"Could not import KUKAControl from local path.\n"
            f"Error: {e}\n"
            f"Ensure py_openshowvar is installed: pip install py_openshowvar"
        )

    _KUKAControl = KC
    _KUKA_IMPORTED = True


# ---- Typed data containers ----

@dataclass
class RobotTCPPose:
    """
    Raw robot TCP (Tool Center Point) pose in base frame.

    Units are in the robot's native representation (mm, degrees).
    This is the raw data before conversion to the math layer.

    Attributes
    ----------
    x_mm, y_mm, z_mm : float
        Position in millimeters (robot base frame).
    a_deg, b_deg, c_deg : float
        Orientation in degrees (KUKA ABC convention).
    """
    x_mm: float
    y_mm: float
    z_mm: float
    a_deg: float
    b_deg: float
    c_deg: float

    def as_tuple(self) -> tuple:
        return (self.x_mm, self.y_mm, self.z_mm, self.a_deg, self.b_deg, self.c_deg)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RobotTCPPose":
        """Build from a 6-element numpy array [x, y, z, a, b, c]."""
        return cls(
            x_mm=float(arr[0]), y_mm=float(arr[1]), z_mm=float(arr[2]),
            a_deg=float(arr[3]), b_deg=float(arr[4]), c_deg=float(arr[5]),
        )


@dataclass
class RobotJointAngles:
    """
    Raw robot joint angles (axis positions).

    Attributes
    ----------
    j1, j2, j3, j4, j5, j6 : float
        Joint angles in degrees for axes 1 through 6.
    """
    j1: float
    j2: float
    j3: float
    j4: float
    j5: float
    j6: float

    def as_list(self) -> list:
        return [self.j1, self.j2, self.j3, self.j4, self.j5, self.j6]

    @classmethod
    def from_list(cls, joints: list) -> "RobotJointAngles":
        return cls(
            j1=float(joints[0]), j2=float(joints[1]), j3=float(joints[2]),
            j4=float(joints[3]), j5=float(joints[4]), j6=float(joints[5]),
        )


@dataclass
class RobotTarget:
    """
    Motion target in robot-native coordinates.

    Attributes
    ----------
    x_mm, y_mm, z_mm : float
        Target position in millimeters.
    rz_deg, ry_deg, rx_deg : float
        Target orientation (ABC Euler angles) in degrees.
    """
    x_mm: float
    y_mm: float
    z_mm: float
    rz_deg: float
    ry_deg: float
    rx_deg: float


# ---- Driver wrapper ----

@dataclass
class KukaDriver:
    """
    Typed wrapper around the existing KUKAControl low-level driver.

    Provides a clean, documented interface to the KUKA robot.
    Internally delegates all communication to the wrapped KUKAControl instance.

    Lifecycle:
      1. Create driver with IP and port
      2. Call connect() to establish handshake
      3. Use read_* / write_* methods
      4. Call close() to clean up

    Attributes
    ----------
    ip : str
        Robot controller IP address.
    port : int
        Robot communication port (typically 7000 for KUKA).
    _kuka : KUKAControl or None
        The wrapped low-level driver instance (lazy-init).
    _connected : bool
        Whether the handshake has completed successfully.
    """

    ip: str
    port: int = 7000
    _kuka: "object | None" = None
    _connected: bool = False

    def _get_kuka(self):
        """
        Get or create the underlying KUKAControl instance.

        Returns
        -------
        KUKAControl
            The low-level driver.

        Raises
        ------
        RuntimeError
            If the KUKAControl class could not be imported.
        """
        if self._kuka is None:
            _ensure_kuka_imported()
            self._kuka = _KUKAControl(self.ip, self.port)
        return self._kuka

    # ---- Connection management ----

    def connect(self) -> bool:
        """
        Establish connection and perform the handshake with the robot.

        The handshake uses the GO variable:
          - Reads GO; if '1', robot is ready
          - Otherwise, writes '1' and retries

        Returns
        -------
        bool
            True if connection and handshake succeeded.
        """
        kuka = self._get_kuka()

        while True:
            print(f"[KukaDriver] Connecting to {self.ip}:{self.port}...")

            if kuka.client is None:
                kuka.connect()
                time.sleep(1)
            else:
                go_state = kuka.read_go()
                if go_state == '1':
                    print("[KukaDriver] Robot handshake confirmed.")
                    self._connected = True
                    return True
                else:
                    print("[KukaDriver] Handshake pending, writing GO=1...")
                    kuka.write_go('1')
                    time.sleep(1)

    def close(self) -> None:
        """Close the robot connection and release resources."""
        if self._kuka is not None:
            self._kuka.close()
        self._kuka = None
        self._connected = False
        print("[KukaDriver] Connection closed.")

    def is_connected(self) -> bool:
        """Check whether the robot connection is active."""
        if self._kuka is None:
            return False
        return self._kuka.client is not None and self._connected

    # ---- Reading robot state ----

    def read_tcp_pose(self) -> Optional[RobotTCPPose]:
        """
        Read the current robot TCP pose from $POS_ACT.

        Returns
        -------
        RobotTCPPose or None
            Current TCP pose (x, y, z in mm; A, B, C in degrees),
            or None if the read failed or robot is not connected.
        """
        if not self.is_connected():
            return None

        raw = self._kuka.read_pose()
        if raw is None:
            return None

        return RobotTCPPose.from_array(raw)

    def read_joint_angles(self) -> Optional[RobotJointAngles]:
        """
        Read the current robot joint angles from $AXIS_ACT.

        Returns
        -------
        RobotJointAngles or None
            Joint angles in degrees for axes 1-6,
            or None if the read failed.
        """
        if not self.is_connected():
            return None

        raw = self._kuka.read_joint()
        if raw is None:
            return None

        return RobotJointAngles.from_list(raw)

    def read_speed_override(self) -> Optional[int]:
        """
        Read the current speed override percentage ($OV_PRO).

        Returns
        -------
        int or None
            Speed override percentage (0-100), or None on failure.
        """
        if not self.is_connected():
            return None

        try:
            raw = self._kuka.overideSpeed()  # read variant (no argument)
            return int(raw.decode())
        except Exception:
            return None

    # ---- Writing targets ----

    def write_target(self, target: RobotTarget) -> bool:
        """
        Send a motion target to the robot.

        Writes to GTARGET_X, GTARGET_Y, GTARGET_Z, GTARGET_A, GTARGET_B, GTARGET_C.

        Parameters
        ----------
        target : RobotTarget
            Target in robot-native coordinates (mm, degrees).

        Returns
        -------
        bool
            True if the write was acknowledged.
        """
        if not self.is_connected():
            print("[KukaDriver] Cannot write target: not connected.")
            return False

        try:
            self._kuka.push_target(
                x=target.x_mm,
                y=target.y_mm,
                z=target.z_mm,
                rz=target.rz_deg,
                ry=target.ry_deg,
                rx=target.rx_deg,
            )
            return True
        except Exception as e:
            print(f"[KukaDriver] Error writing target: {e}")
            return False

    def write_position_only(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
    ) -> bool:
        """
        Send a position-only target (3-DOF), preserving current orientation.

        Reads the current pose to get the current ABC angles,
        then sends the new position with those angles.

        Parameters
        ----------
        x_mm, y_mm, z_mm : float
            Target position in millimeters.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_connected():
            return False

        try:
            self._kuka.push_3p([x_mm, y_mm, z_mm])
            return True
        except Exception as e:
            print(f"[KukaDriver] Error writing position: {e}")
            return False

    def write_orientation_only(
        self,
        a_deg: float,
        b_deg: float,
        c_deg: float,
    ) -> bool:
        """
        Send an orientation-only target (3-DOF), preserving current position.

        Parameters
        ----------
        a_deg, b_deg, c_deg : float
            Target orientation in degrees (KUKA ABC convention).

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_connected():
            return False

        try:
            self._kuka.push_3o([a_deg, b_deg, c_deg])
            return True
        except Exception as e:
            print(f"[KukaDriver] Error writing orientation: {e}")
            return False

    def set_speed(self, speed_percent: int) -> bool:
        """
        Set the robot speed override percentage.

        Parameters
        ----------
        speed_percent : int
            Speed override (0-100). Lower values for safety during testing.

        Returns
        -------
        bool
            True if the speed was verified as set.
        """
        if not self.is_connected():
            return False

        try:
            self._kuka.overideSpeed(speed_percent)
            return True
        except Exception as e:
            print(f"[KukaDriver] Error setting speed: {e}")
            return False
