"""
Pipeline module: runtime orchestration for trajectory estimation.

This module handles data flow and execution modes only.
No mathematics, geometry, or optimization logic belongs here.

Provides:
  - offline pipeline: batch process a sequence of observations
  - online pipeline: frame-by-frame incremental processing
"""

from .runner import OfflinePipeline, OnlinePipeline

__all__ = [
    "OfflinePipeline",
    "OnlinePipeline",
]
