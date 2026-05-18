"""
CSV format definition for synchronized measurement files.

Standard column order:
  frame_id, timestamp, u, v, X_mm, Y_mm, Z_mm, A_deg, B_deg, C_deg, sync_error_s

This format is the canonical on-disk representation of synchronized
monocular observations. It enables:
  - Offline replay (read CSV → reconstruct SynchronizedMeasurement objects)
  - Debugging (human-readable columns)
  - Algorithm validation (deterministic input from recorded data)
  - Data exchange between pipeline stages
"""

# Standard CSV header
CSV_HEADER = "frame_id,timestamp,u,v,X_mm,Y_mm,Z_mm,A_deg,B_deg,C_deg,sync_error_s"

# Column indices for programmatic access
COL_FRAME_ID = 0
COL_TIMESTAMP = 1
COL_U = 2
COL_V = 3
COL_X_MM = 4
COL_Y_MM = 5
COL_Z_MM = 6
COL_A_DEG = 7
COL_B_DEG = 8
COL_C_DEG = 9
COL_SYNC_ERROR = 10

# Number of columns in the CSV format
NUM_COLUMNS = 11
