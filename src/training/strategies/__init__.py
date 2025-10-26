"""
Training strategies for the unified trainer.
"""

import warnings

from .masking import TimeStepMasking, ProgressiveMasking

# Import quantization classes with deprecation warning
warnings.warn(
    "QLoRAManager and QuantizationManager are currently unused and marked for deprecation. "
    "The quantization classes have config attribute mismatches and are not functional. "
    "The ProgressiveRackBuilder implements its own quantization logic instead of using these classes.",
    DeprecationWarning,
    stacklevel=2
)
from .quantization import QLoRAManager, QuantizationManager

# Import caching classes with deprecation warning
warnings.warn(
    "TimeStepCache and ActivationCache are currently unused and potentially deprecated. "
    "Actual caching is implemented in ProgressiveDataLoader using simple dictionary-based caching.",
    DeprecationWarning,
    stacklevel=2
)
from .caching import TimeStepCache, ActivationCache

__all__ = [
    "TimeStepMasking",
    "ProgressiveMasking", 
    "QLoRAManager",
    "QuantizationManager",
    "TimeStepCache",
    "ActivationCache"
]
