"""
Masking strategies for the unified trainer.
"""

from .time_step_masking import TimeStepMasking
from .progressive_masking import ProgressiveMasking

__all__ = [
    "TimeStepMasking",
    "ProgressiveMasking"
]
