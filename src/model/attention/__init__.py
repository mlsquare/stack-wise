"""
Attention module for StackWise.
Provides all attention mechanisms with unified interface.
"""

from .base import BaseAttention, AttentionMode
from .mha import MHA
from .gqa import GQA
from .mla import MLA
from .kernel import KernelAttention, KernelType

__all__ = [
    "BaseAttention",
    "AttentionMode", 
    "MHA",
    "GQA",
    "MLA",
    "KernelAttention",
    "KernelType"
]
