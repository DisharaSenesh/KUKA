"""
Recording module: CSV I/O for synchronized measurement data.

Enables online/offline unification:
  - Online: SynchronizedWriter records live synchronized measurements to CSV
  - Offline: SynchronizedReader replays CSV measurements as SynchronizedMeasurement objects

Both paths produce identical input for the downstream tracking/optimization pipeline.
"""

from .csv_format import CSV_HEADER, NUM_COLUMNS, COL_FRAME_ID, COL_TIMESTAMP
from .synchronized_writer import SynchronizedWriter
from .synchronized_reader import SynchronizedReader

__all__ = [
    "CSV_HEADER",
    "NUM_COLUMNS",
    "COL_FRAME_ID",
    "COL_TIMESTAMP",
    "SynchronizedWriter",
    "SynchronizedReader",
]
