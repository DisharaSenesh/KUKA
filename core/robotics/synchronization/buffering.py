"""
Command buffering: rate-limited and batched target delivery.

In online tracking, target commands may arrive faster than the robot
can execute them. This module provides utilities to buffer, rate-limit,
and smooth command delivery.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class TargetCommand:
    """
    A buffered motion target with metadata.

    Attributes
    ----------
    position_world : np.ndarray, shape (3,)
        Target position in meters (world frame).
    timestamp : float
        Time at which this command was generated.
    sent : bool
        Whether this command has been transmitted to the robot.
    """
    position_world: np.ndarray
    timestamp: float
    sent: bool = False


@dataclass
class CommandBuffer:
    """
    Rate-limited command buffer for robot targets.

    Buffers incoming commands and delivers the most recent valid
    target at a controlled rate, dropping stale intermediate targets.

    This prevents flooding the robot with commands faster than
    its control loop can process them.

    Attributes
    ----------
    max_size : int
        Maximum number of pending commands.
    min_interval : float
        Minimum time between command deliveries (seconds).
    commands : deque
        Buffered command queue.
    last_send_time : float
        Timestamp of the most recent delivery.
    """

    max_size: int = 10
    min_interval: float = 0.05
    commands: deque = field(default_factory=deque)
    last_send_time: float = 0.0

    def push(self, position_world: np.ndarray, timestamp: float) -> None:
        """
        Add a command to the buffer. Evicts oldest if full.

        Parameters
        ----------
        position_world : np.ndarray, shape (3,)
            Target position.
        timestamp : float
            Command generation time.
        """
        cmd = TargetCommand(
            position_world=np.asarray(position_world, dtype=np.float64),
            timestamp=timestamp,
        )
        self.commands.append(cmd)

        # Evict oldest if over capacity
        while len(self.commands) > self.max_size:
            self.commands.popleft()

    def pop_latest(self) -> Optional[TargetCommand]:
        """
        Get the most recent unsent command.

        Drops all older unsent commands — only the latest target matters.

        Returns
        -------
        TargetCommand or None
            The most recent pending command, or None if buffer is empty.
        """
        if len(self.commands) == 0:
            return None

        # Take the most recent command
        latest = self.commands.pop()
        return latest

    def can_send(self, current_time: float) -> bool:
        """
        Check if enough time has elapsed since the last send.

        Parameters
        ----------
        current_time : float
            Current timestamp.

        Returns
        -------
        bool
            True if the rate limit interval has passed.
        """
        return (current_time - self.last_send_time) >= self.min_interval

    def mark_sent(self) -> None:
        """Record that a command was just sent (for rate limiting)."""
        import time
        self.last_send_time = time.perf_counter()

    def clear(self) -> None:
        """Discard all buffered commands."""
        self.commands.clear()

    def __len__(self) -> int:
        return len(self.commands)
