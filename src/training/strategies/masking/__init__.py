"""
Masking strategies for the unified trainer.

⚠️  WARNING: This module is currently BROKEN and UNUSED.
The masking classes have config attribute mismatches and are not functional.
The ProgressiveTrainer that depends on these classes is also broken.
"""

import warnings

# Issue deprecation warning for the entire module
warnings.warn(
    "The masking module is currently broken and unused. "
    "The masking classes have config attribute mismatches and are not functional. "
    "The ProgressiveTrainer that depends on these classes is also broken.",
    DeprecationWarning,
    stacklevel=2
)

from .time_step_masking import TimeStepMasking
from .progressive_masking import ProgressiveMasking

__all__ = [
    "TimeStepMasking",
    "ProgressiveMasking"
]
