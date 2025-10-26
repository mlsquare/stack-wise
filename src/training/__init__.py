"""
Unified training module for stack-wise transformer training.

This module provides a comprehensive training framework supporting:
- Layer-wise training (block_size=1)
- Block-wise training (block_size=4)
- Fused training with quantization and QLoRA adapters
- Time-step-based masking strategies
- Memory-efficient caching
"""

# Strategies
from .strategies import (
    TimeStepMasking, ProgressiveMasking,
    QLoRAManager, QuantizationManager,
    TimeStepCache, ActivationCache
)
from .utils import ConfigValidator, CheckpointManager, TrainingMetrics

# Trainer (hierarchical structure)
from .trainer import (
    BlockTrainer,
    StackTrainer,
    RackTrainer,
    Trainer
)

# Progressive training components
from .progressive_trainer import ProgressiveTrainer
from .progressive_rack_builder import ProgressiveRackBuilder, PrecisionManager
from .progressive_dataloader import ProgressiveDataLoader, CachedDataLoader

# Legacy imports (deprecated) are lazily imported to avoid import-time
# deprecation warnings during test collection.

__all__ = [
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
    
    # Trainers (hierarchical structure)
    "BlockTrainer",
    "StackTrainer",
    "RackTrainer",
    "Trainer",
    
    # Progressive training components
    "ProgressiveTrainer",
    "ProgressiveRackBuilder",
    "PrecisionManager",
    "ProgressiveDataLoader",
    "CachedDataLoader",
    
    # Legacy (deprecated)
    "LayerwiseTrainer"
]


def __getattr__(name: str):
    """Lazily import deprecated legacy symbols on attribute access."""
    if name == "LayerwiseTrainer":
        from . import legacy
        return getattr(legacy, "LayerwiseTrainer")
    raise AttributeError(name)

# Version information
__version__ = "2.0.0"
__author__ = "Stack-Wise Team"
__description__ = "Unified training framework for stack-wise transformer training"
