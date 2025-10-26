"""
Masking strategies for the unified trainer.
"""

import warnings

# Import TimeStepMasking with experimental warning
warnings.warn(
    "TimeStepMasking is HIGHLY EXPERIMENTAL and in active development. "
    "It may have bugs and should be used with caution in production environments.",
    UserWarning,
    stacklevel=2
)
from .time_step_masking import TimeStepMasking

from .progressive_masking import ProgressiveMasking

__all__ = [
    "TimeStepMasking",
    "ProgressiveMasking"
]
