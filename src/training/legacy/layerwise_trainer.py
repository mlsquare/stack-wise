"""
Legacy layerwise trainer (deprecated).

This module contains the original layerwise trainer implementation.
It is kept for backward compatibility but should not be used in new code.
Use the new UnifiedTrainer instead.
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "LayerwiseTrainer is deprecated. Use UnifiedTrainer from src.training.core instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the original implementation
try:
    from ...training.layerwise_trainer import LayerwiseTrainer
except ImportError:
    # If the original file doesn't exist, create a placeholder
    class LayerwiseTrainer:
        """Deprecated LayerwiseTrainer - use UnifiedTrainer instead."""
        
        def __init__(self, *args, **kwargs):
            raise DeprecationWarning(
                "LayerwiseTrainer is deprecated. Use UnifiedTrainer from src.training.core instead."
            )
