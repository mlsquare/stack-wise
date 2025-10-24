"""
StackWise Model Package

This package contains all model components for the StackWise layer-wise trainable Transformer.
"""

# Core attention components
from .attention.attention import CoreAttention
from .attention.builder import AttentionBuilder
from .attention.presets import AttentionPresets

# Layer components (consolidated)
from .layers import (
    LexicalKernelManager,
    SwiGLUFFN, 
    MLGKALayer,
    ModelFamily,
    SUPPORTED_FAMILIES
)

# Training components
from training.layerwise_trainer import (
    LayerwiseTrainer,
    MaskScheduler,
    MaskDiffusionObjective,
    HashBasedActivationCache,
    FixedMaskAssigner
)

__all__ = [
    # Attention components
    "CoreAttention",
    "AttentionBuilder", 
    "AttentionPresets",
    
    # Layer components
    "LexicalKernelManager",
    "SwiGLUFFN",
    "MLGKALayer", 
    "ModelFamily",
    "SUPPORTED_FAMILIES",
    
    # Training components
    "LayerwiseTrainer",
    "MaskScheduler",
    "MaskDiffusionObjective", 
    "HashBasedActivationCache",
    "FixedMaskAssigner"
]
