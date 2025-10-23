"""
Attention factory for creating attention mechanisms.
"""

from typing import Union, Literal
from .base import BaseAttention, AttentionMode
from .mha import MHA
from .gqa import GQA
from .mla import MLA
from .kernel import KernelAttention, KernelType

AttentionType = Literal["mha", "gqa", "mla", "kernel"]


def create_attention(
    attention_type: AttentionType,
    d_model: int,
    n_heads: int,
    n_kv_heads: int = None,
    r_q: int = None,
    r_kv: int = None,
    kernel_dim: int = 64,
    kernel_type: KernelType = "gaussian",
    dropout: float = 0.0,
    attention_mode: AttentionMode = "bidirectional"
) -> BaseAttention:
    """
    Create attention mechanism based on type.
    
    Args:
        attention_type: Type of attention mechanism
        d_model: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (for GQA/MLA)
        r_q: Query rank for MLA
        r_kv: Key-value rank for MLA
        kernel_dim: Kernel dimension for kernel attention
        kernel_type: Kernel type for kernel attention
        dropout: Dropout probability
        attention_mode: Attention mode (bidirectional/causal)
        
    Returns:
        Attention mechanism instance
    """
    if attention_type == "mha":
        return MHA(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_mode=attention_mode
        )
    
    elif attention_type == "gqa":
        if n_kv_heads is None:
            raise ValueError("n_kv_heads required for GQA")
        return GQA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            attention_mode=attention_mode
        )
    
    elif attention_type == "mla":
        if n_kv_heads is None or r_q is None or r_kv is None:
            raise ValueError("n_kv_heads, r_q, and r_kv required for MLA")
        return MLA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            r_q=r_q,
            r_kv=r_kv,
            dropout=dropout,
            attention_mode=attention_mode
        )
    
    elif attention_type == "kernel":
        return KernelAttention(
            d_model=d_model,
            n_heads=n_heads,
            kernel_dim=kernel_dim,
            kernel_type=kernel_type,
            dropout=dropout,
            attention_mode=attention_mode
        )
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def get_attention_info(attention: BaseAttention) -> dict:
    """
    Get information about an attention mechanism.
    
    Args:
        attention: Attention mechanism instance
        
    Returns:
        Dictionary with attention information
    """
    info = {
        "type": attention.__class__.__name__,
        "d_model": attention.d_model,
        "attention_mode": attention.attention_mode,
        "dropout": attention.dropout
    }
    
    # Add type-specific information
    if isinstance(attention, MHA):
        info.update({
            "n_heads": attention.n_heads,
            "d_k": attention.d_k
        })
    
    elif isinstance(attention, GQA):
        info.update({
            "n_heads": attention.n_heads,
            "n_kv_heads": attention.n_kv_heads,
            "group_size": attention.group_size,
            "d_k": attention.d_k
        })
    
    elif isinstance(attention, MLA):
        info.update({
            "n_heads": attention.n_heads,
            "n_kv_heads": attention.n_kv_heads,
            "r_q": attention.r_q,
            "r_kv": attention.r_kv,
            "group_size": attention.group_size,
            "d_k": attention.d_k
        })
        # Add parameter count if available
        if hasattr(attention, 'get_parameter_count'):
            info["parameter_count"] = attention.get_parameter_count()
    
    elif isinstance(attention, KernelAttention):
        info.update({
            "n_heads": attention.n_heads,
            "kernel_dim": attention.kernel_dim,
            "kernel_type": attention.kernel_type,
            "d_k": attention.d_k
        })
        # Add kernel info if available
        if hasattr(attention, 'get_kernel_info'):
            info["kernel_info"] = attention.get_kernel_info()
    
    return info
