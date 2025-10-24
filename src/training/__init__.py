"""
Unified training module for stack-wise transformer training.

This module provides a comprehensive training framework supporting:
- Layer-wise training (block_size=1)
- Block-wise training (block_size=4)
- Fused training with quantization and QLoRA adapters
- Time-step-based masking strategies
- Memory-efficient caching
"""

from .core import UnifiedTrainer, BlockTrainer, FusionTrainer
from .strategies import (
    TimeStepMasking, ProgressiveMasking,
    QLoRAManager, QuantizationManager,
    TimeStepCache, ActivationCache
)
from .utils import ConfigValidator, CheckpointManager, TrainingMetrics

# Legacy imports (deprecated)
from .legacy import LayerwiseTrainer

__all__ = [
    # Core trainers
    "UnifiedTrainer",
    "BlockTrainer", 
    "FusionTrainer",
    
    # Strategies
    "TimeStepMasking",
    "ProgressiveMasking",
    "QLoRAManager",
    "QuantizationManager", 
    "TimeStepCache",
    "ActivationCache",
    
    # Utilities
    "ConfigValidator",
    "CheckpointManager",
    "TrainingMetrics",
    
    # Legacy (deprecated)
    "LayerwiseTrainer"
]

# Version information
__version__ = "2.0.0"
__author__ = "Stack-Wise Team"
__description__ = "Unified training framework for stack-wise transformer training"
