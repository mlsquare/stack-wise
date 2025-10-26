"""
Attention factory for creating attention mechanisms.
"""

from typing import Literal, Optional, Union
from .attention import CoreAttention, AttentionMode
from .kernels import KernelType


def get_attention_info(attention: CoreAttention) -> dict:
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
    
    # All attention mechanisms are now CoreAttention
    if isinstance(attention, CoreAttention):
        info.update({
            "n_heads": attention.n_heads,
            "n_kv_heads": attention.n_kv_heads,
            "group_size": attention.group_size,
            "d_k": attention.d_k,
            "kernel_type": attention.kernel_type,
            "kernel_dim": attention.kernel_dim,
            "use_low_rank": attention.use_low_rank,
            "r_q": attention.r_q,
            "r_kv": attention.r_kv
        })
        
        # Add kernel info if available
        if hasattr(attention, 'get_kernel_info'):
            info["kernel_info"] = attention.get_kernel_info()
    
    return info


class AttentionBuilder:
    """
    Builder pattern for creating attention mechanisms.
    
    Default configuration: MHA (Multi-Head Attention)
    - No GQA (n_kv_heads = n_heads)
    - No MLA (r_q = None, r_kv = None)  
    - Scaled dot-product attention (kernel_type = "linear")
    
    Each method is disjoint and can be combined as needed.
    
    Note: GQA is determined by n_kv_heads < n_heads, not by attention_type.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, attention_mode: AttentionMode = "bidirectional"):
        """
        Initialize attention builder with default MHA configuration.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            attention_mode: Attention mode (bidirectional/causal)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_mode = attention_mode
        
        # Default MHA configuration
        self.n_kv_heads = n_heads  # No GQA by default
        self.r_q = None            # No MLA by default
        self.r_kv = None           # No MLA by default
        self.kernel_type = "linear"  # Scaled dot-product by default
        self.kernel_dim = 64       # Default kernel dimension
    
    def with_gqa(self, n_kv_heads: int) -> 'AttentionBuilder':
        """
        Enable Grouped Query Attention (GQA).
        
        Args:
            n_kv_heads: Number of key-value heads (must be < n_heads)
            
        Returns:
            Self for method chaining
        """
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive for GQA")
        if n_kv_heads >= self.n_heads:
            raise ValueError(f"n_kv_heads ({n_kv_heads}) must be less than n_heads ({self.n_heads}) for GQA")
        if self.n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({n_kv_heads}) for grouped attention"
            )
        self.n_kv_heads = n_kv_heads
        return self
    
    def with_mla(self, r_q: int, r_kv: int) -> 'AttentionBuilder':
        """
        Enable Multi-Latent Attention (MLA).
        
        Args:
            r_q: Query rank for low-rank factorization
            r_kv: Key-value rank for low-rank factorization
            
        Returns:
            Self for method chaining
        """
        if r_q <= 0 or r_kv <= 0:
            raise ValueError("r_q and r_kv must be positive for MLA")
        self.r_q = r_q
        self.r_kv = r_kv
        return self
    
    def with_kernel(self, kernel_type: KernelType, kernel_dim: int = 64) -> 'AttentionBuilder':
        """
        Set kernel type for attention computation.
        
        Args:
            kernel_type: Type of kernel ("linear", "gaussian", "laplacian", "uniform")
            kernel_dim: Kernel dimension (ignored for linear kernel)
            
        Returns:
            Self for method chaining
        """
        if kernel_dim <= 0:
            raise ValueError("kernel_dim must be positive")
        self.kernel_type = kernel_type
        self.kernel_dim = kernel_dim
        return self
    
    def build(self) -> CoreAttention:
        """
        Build the attention mechanism with current configuration.
        
        Returns:
            Configured attention mechanism
        """
        return CoreAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            r_q=self.r_q,
            r_kv=self.r_kv,
            kernel_dim=self.kernel_dim,
            kernel_type=self.kernel_type,
            dropout=self.dropout,
            attention_mode=self.attention_mode
        )
    
    @classmethod
    def from_config(cls, config: Union[dict, object]) -> 'AttentionBuilder':
        """
        Create AttentionBuilder from configuration object or dictionary.
        
        Args:
            config: Configuration object (with attributes) or dictionary
            
        Returns:
            Configured AttentionBuilder instance
        """
        # Handle both dict and object configs
        if isinstance(config, dict):
            d_model = config['d_model']
            n_heads = config['n_heads']
            dropout = config.get('dropout', 0.0)
            attention_mode = config.get('attention_mode', 'bidirectional')
        else:
            d_model = config.d_model
            n_heads = config.n_heads
            dropout = getattr(config, 'dropout', 0.0)
            attention_mode = getattr(config, 'attention_mode', 'bidirectional')
        
        builder = cls(d_model, n_heads, dropout, attention_mode)
        
        # Apply configuration
        if isinstance(config, dict):
            if 'n_kv_heads' in config:
                builder.n_kv_heads = config['n_kv_heads']
            if 'r_q' in config and 'r_kv' in config:
                builder.r_q = config['r_q']
                builder.r_kv = config['r_kv']
            if 'kernel_type' in config:
                builder.kernel_type = config['kernel_type']
            if 'kernel_dim' in config:
                builder.kernel_dim = config['kernel_dim']
        else:
            if hasattr(config, 'n_kv_heads'):
                builder.n_kv_heads = config.n_kv_heads
            if hasattr(config, 'r_q') and hasattr(config, 'r_kv'):
                builder.r_q = config.r_q
                builder.r_kv = config.r_kv
            if hasattr(config, 'kernel_type'):
                builder.kernel_type = config.kernel_type
            if hasattr(config, 'kernel_dim'):
                builder.kernel_dim = config.kernel_dim
        
        return builder
    
    def get_config(self) -> dict:
        """
        Get current configuration for inspection.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "r_q": self.r_q,
            "r_kv": self.r_kv,
            "kernel_type": self.kernel_type,
            "kernel_dim": self.kernel_dim,
            "dropout": self.dropout,
            "attention_mode": self.attention_mode,
            "is_gqa": self.n_kv_heads < self.n_heads,
            "is_mla": self.r_q is not None and self.r_kv is not None,
            "is_kernel": self.kernel_type != "linear"
        }
