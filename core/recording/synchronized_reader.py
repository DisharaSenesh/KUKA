"""
Synchronized CSV reader: offline replay of recorded measurements.

Reads the standard CSV format and reconstructs SynchronizedMeasurement
objects. This enables offline algorithm development, debugging, and
validation using recorded data — with identical input to the online path.

The reader produces the same SynchronizedMeasurement objects that the
online pipeline would, achieving the online/offline unification goal.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional, List

from core.synchronization.synchronized_measurement import SynchronizedMeasurement
from .csv_format import CSV_HEADER, NUM_COLUMNS


@dataclass
class SynchronizedReader:
    """
    Reads a synchronized measurement CSV file and reconstructs objects.

    Usage:

        reader = SynchronizedReader("recording.csv")
        reader.open()
        for sm in reader:
            process(sm)
        reader.close()

    Or load all into memory:

        reader = SynchronizedReader("recording.csv")
        measurements = reader.load_all()

    Attributes
    ----------
    filepath : str
        Path to the input CSV file.
    _file : file or None
        Open file handle.
    _reader : csv.reader or None
        CSV row iterator.
    _count : int
        Number of measurements read.
    """

    filepath: str
    _file: Optional[object] = None
    _reader: Optional[object] = None
    _count: int = 0

    def open(self) -> None:
        """
        Open the CSV file for reading.

        Skips the header row and validates the column count.
        """
        self._file = open(self.filepath, "r")
        self._reader = csv.reader(self._file)

        # Read and validate header
        header = next(self._reader, None)
        if header is None:
            raise ValueError(f"Empty file: {self.filepath}")

        if len(header) < NUM_COLUMNS:
            raise ValueError(
                f"Expected at least {NUM_COLUMNS} columns, got {len(header)}. "
                f"Header: {header}"
            )

        self._count = 0

    def read_next(self) -> Optional[SynchronizedMeasurement]:
        """
        Read the next measurement from the CSV file.

        Returns
        -------
        SynchronizedMeasurement or None
            None when the file is exhausted.
        """
        if self._reader is None:
            raise RuntimeError("Reader not opened. Call open() first.")

        try:
            row = next(self._reader)
        except StopIteration:
            return None

        # Parse numeric columns
        values = [float(v) for v in row]

        sm = SynchronizedMeasurement(
            frame_id=int(values[0]),
            timestamp=float(values[1]),
            u=float(values[2]),
            v=float(values[3]),
            X_mm=float(values[4]),
            Y_mm=float(values[5]),
            Z_mm=float(values[6]),
            A_deg=float(values[7]),
            B_deg=float(values[8]),
            C_deg=float(values[9]),
            sync_error_s=float(values[10]) if len(values) > 10 else 0.0,
            sync_method="csv_replay",
            is_valid=True,
        )

        self._count += 1
        return sm

    def load_all(self) -> List[SynchronizedMeasurement]:
        """
        Load all measurements from the CSV file into memory.

        Returns
        -------
        list of SynchronizedMeasurement
        """
        with open(self.filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            measurements = []
            for row in reader:
                values = [float(v) for v in row]
                sm = SynchronizedMeasurement(
                    frame_id=int(values[0]),
                    timestamp=float(values[1]),
                    u=float(values[2]),
                    v=float(values[3]),
                    X_mm=float(values[4]),
                    Y_mm=float(values[5]),
                    Z_mm=float(values[6]),
                    A_deg=float(values[7]),
                    B_deg=float(values[8]),
                    C_deg=float(values[9]),
                    sync_error_s=float(values[10]) if len(values) > 10 else 0.0,
                    sync_method="csv_replay",
                    is_valid=True,
                )
                measurements.append(sm)

        print(f"[SynchronizedReader] Loaded {len(measurements)} measurements from {self.filepath}")
        return measurements

    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._reader = None

    @property
    def count(self) -> int:
        return self._count

    def __iter__(self):
        return self

    def __next__(self) -> SynchronizedMeasurement:
        sm = self.read_next()
        if sm is None:
            raise StopIteration
        return sm

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
