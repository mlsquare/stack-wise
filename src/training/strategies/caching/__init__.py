"""
Caching strategies for the unified trainer.

⚠️  WARNING: This module is currently UNUSED and potentially DEPRECATED.
The actual caching functionality is implemented in ProgressiveDataLoader
using simple dictionary-based caching.
"""

import warnings

# Issue deprecation warning for the entire module
warnings.warn(
    "The caching module is currently unused and potentially deprecated. "
    "Actual caching is implemented in ProgressiveDataLoader using simple dictionary-based caching.",
    DeprecationWarning,
    stacklevel=2
)

from .time_step_cache import TimeStepCache
from .activation_cache import ActivationCache

__all__ = [
    "TimeStepCache",
    "ActivationCache"
]
