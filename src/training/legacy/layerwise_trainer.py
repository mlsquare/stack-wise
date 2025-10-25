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

# LayerwiseTrainer has been removed - use UnifiedTrainer instead
class LayerwiseTrainer:
    """Deprecated LayerwiseTrainer - use UnifiedTrainer instead."""
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "LayerwiseTrainer has been removed. Use UnifiedTrainer from src.training.core instead."
        )
