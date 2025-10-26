"""
Training strategies for the unified trainer.
"""

from .masking import TimeStepMasking, ProgressiveMasking
from .quantization import QLoRAManager, QuantizationManager

# Import caching classes with deprecation warning
import warnings
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
