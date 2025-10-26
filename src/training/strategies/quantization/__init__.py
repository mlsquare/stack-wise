"""
Quantization strategies for the unified trainer.

⚠️  WARNING: This module is currently UNUSED and marked for DEPRECATION.
The quantization classes have config attribute mismatches and are not functional.
The ProgressiveRackBuilder implements its own quantization logic instead of using these classes.
"""

import warnings

# Issue deprecation warning for the entire module
warnings.warn(
    "The quantization module is currently unused and marked for deprecation. "
    "The quantization classes have config attribute mismatches and are not functional. "
    "The ProgressiveRackBuilder implements its own quantization logic instead of using these classes.",
    DeprecationWarning,
    stacklevel=2
)

from .qlora_manager import QLoRAManager
from .quantization_manager import QuantizationManager

__all__ = [
    "QLoRAManager",
    "QuantizationManager"
]
