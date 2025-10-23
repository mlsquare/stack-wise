"""
Attention module for StackWise.
Provides all attention mechanisms with unified interface.
"""

from .attention import CoreAttention, AttentionMode
from .kernels import KernelType
from .builder import get_attention_info, AttentionBuilder
from .presets import AttentionPresets, AttentionFactory, create_attention_factory, create_layer_attention

__all__ = [
    "CoreAttention",
    "AttentionMode", 
    "KernelType",
    "get_attention_info",
    "AttentionBuilder",
    "AttentionPresets",
    "AttentionFactory",
    "create_attention_factory",
    "create_layer_attention"
]
