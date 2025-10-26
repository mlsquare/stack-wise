"""
Training strategies for the unified trainer.
"""

# Import masking classes with deprecation warning
import warnings
warnings.warn(
    "TimeStepMasking and ProgressiveMasking are currently broken and unused. "
    "The masking classes have config attribute mismatches and are not functional. "
    "The ProgressiveTrainer that depends on these classes is also broken.",
    DeprecationWarning,
    stacklevel=2
)
from .masking import TimeStepMasking, ProgressiveMasking

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
