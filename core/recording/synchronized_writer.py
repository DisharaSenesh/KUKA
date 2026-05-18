"""
Synchronized CSV writer: records SynchronizedMeasurement objects to disk.

Writes the standard CSV format (frame_id, timestamp, u, v, X, Y, Z, A, B, C, sync_error_s).

Supports:
  - Opening a file with header
  - Appending individual measurements
  - Automatic flushing (configurable interval)
  - Clean close
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, TextIO

from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from .csv_format import CSV_HEADER


@dataclass
class SynchronizedWriter:
    """
    Writes SynchronizedMeasurement objects to a CSV file.

    Usage:

        writer = SynchronizedWriter("recording.csv")
        writer.open()
        for sm in measurements:
            writer.write(sm)
        writer.close()

    Attributes
    ----------
    filepath : str
        Path to the output CSV file.
    _file : TextIO or None
        Open file handle.
    _count : int
        Number of measurements written.
    """

    filepath: str
    _file: Optional[TextIO] = None
    _count: int = 0

    def open(self) -> None:
        """
        Open the CSV file and write the header row.

        Creates the output directory if it does not exist.
        """
        # Ensure directory exists
        dirname = os.path.dirname(self.filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        self._file = open(self.filepath, "w")
        self._file.write(CSV_HEADER + "\n")
        self._file.flush()
        self._count = 0

    def write(self, measurement: SynchronizedMeasurement) -> None:
        """
        Append a single synchronized measurement as a CSV row.

        Parameters
        ----------
        measurement : SynchronizedMeasurement
            The measurement to write.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Call open() first.")

        row = measurement.to_csv_row()
        line = ",".join(str(v) for v in row)
        self._file.write(line + "\n")
        self._count += 1

    def write_batch(self, measurements: list) -> None:
        """
        Append multiple measurements at once.

        Parameters
        ----------
        measurements : list of SynchronizedMeasurement
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Call open() first.")

        lines = []
        for sm in measurements:
            row = sm.to_csv_row()
            lines.append(",".join(str(v) for v in row))

        self._file.write("\n".join(lines) + "\n")
        self._count += len(measurements)

    def flush(self) -> None:
        """Force-flush buffered writes to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file. Safe to call multiple times."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            print(f"[SynchronizedWriter] Wrote {self._count} measurements to {self.filepath}")

    @property
    def count(self) -> int:
        """Number of measurements written."""
        return self._count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
