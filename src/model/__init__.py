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

# Architecture components (new hierarchical structure)
from .architecture import (
    Block,
    Stack,
    Rack,
    create_block_spec,
    create_stack_from_spec,
    create_rack_from_specs,
    create_rack_from_config,
    create_simple_rack
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
    
    # Architecture components (new hierarchical structure)
    "Block",
    "Stack", 
    "Rack",
    "create_block_spec",
    "create_stack_from_spec",
    "create_rack_from_specs",
    "create_rack_from_config",
    "create_simple_rack",
    
    # Training components
    "LayerwiseTrainer",
    "MaskScheduler",
    "MaskDiffusionObjective", 
    "HashBasedActivationCache",
    "FixedMaskAssigner"
]
