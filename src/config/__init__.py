"""
Configuration module for StackWise.
Provides hierarchical configuration with validation and defaults.
"""

from .base import (
    BaseConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    StackWiseConfig,
    AttentionType,
    AttentionMode,
    FineTuneMode,
    KernelType
)

__all__ = [
    "BaseConfig",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "StackWiseConfig",
    "AttentionType",
    "AttentionMode", 
    "FineTuneMode",
    "KernelType"
]
