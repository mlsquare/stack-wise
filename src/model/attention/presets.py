"""
Attention presets and configuration utilities for StackWise.
Provides common attention configurations for easy reuse across layers.
"""

from typing import Dict, Any
from .attention import CoreAttention
from .builder import AttentionBuilder


class AttentionPresets:
    """
    Common attention configurations for StackWise layers.
    """
    
    @staticmethod
    def gpt_style(d_model: int, n_heads: int, dropout: float = 0.0) -> Dict[str, Any]:
        """GPT-style attention: MHA with causal attention."""
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_heads,  # No GQA
            "r_q": None,           # No MLA
            "r_kv": None,          # No MLA
            "kernel_type": "linear",
            "kernel_dim": d_model // n_heads,
            "dropout": dropout,
            "attention_mode": "causal"
        }
    
    @staticmethod
    def bert_style(d_model: int, n_heads: int, dropout: float = 0.0) -> Dict[str, Any]:
        """BERT-style attention: MHA with bidirectional attention."""
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_heads,  # No GQA
            "r_q": None,           # No MLA
            "r_kv": None,          # No MLA
            "kernel_type": "linear",
            "kernel_dim": d_model // n_heads,
            "dropout": dropout,
            "attention_mode": "bidirectional"
        }
    
    @staticmethod
    def efficient_gqa(d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0) -> Dict[str, Any]:
        """Efficient GQA configuration for large models."""
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "r_q": None,           # No MLA
            "r_kv": None,          # No MLA
            "kernel_type": "linear",
            "kernel_dim": d_model // n_heads,
            "dropout": dropout,
            "attention_mode": "bidirectional"
        }
    
    @staticmethod
    def kernel_attention(d_model: int, n_heads: int, kernel_type: str, kernel_dim: int, dropout: float = 0.0) -> Dict[str, Any]:
        """Kernel-based attention configuration."""
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_heads,  # No GQA
            "r_q": None,           # No MLA
            "r_kv": None,          # No MLA
            "kernel_type": kernel_type,
            "kernel_dim": kernel_dim,
            "dropout": dropout,
            "attention_mode": "bidirectional"
        }
    
    @staticmethod
    def full_featured(d_model: int, n_heads: int, n_kv_heads: int, r_q: int, r_kv: int, 
                     kernel_type: str, kernel_dim: int, dropout: float = 0.0) -> Dict[str, Any]:
        """Full-featured attention with GQA, MLA, and kernel."""
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "r_q": r_q,
            "r_kv": r_kv,
            "kernel_type": kernel_type,
            "kernel_dim": kernel_dim,
            "dropout": dropout,
            "attention_mode": "bidirectional"
        }
    
    @staticmethod
    def mlgka(d_model: int, n_heads: int, n_kv_heads: int, r_q: int, r_kv: int, 
              kernel_dim: int, dropout: float = 0.0) -> Dict[str, Any]:
        """
        MLGKA: Multi-Latent + GQA + Kernel Attention.
        
        Combines three efficiency techniques:
        - ML: Multi-Latent Attention (low-rank factorization)
        - GQA: Grouped Query Attention (shared K/V heads)
        - KA: Kernel Attention with Laplacian kernel (Random Kitchen Sinks)
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads (for GQA)
            r_q: Query rank for MLA
            r_kv: Key-value rank for MLA
            kernel_dim: Kernel dimension for Laplacian kernel
            dropout: Dropout probability
            
        Returns:
            Configuration dictionary for MLGKA attention
        """
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "r_q": r_q,
            "r_kv": r_kv,
            "kernel_type": "laplacian",
            "kernel_dim": kernel_dim,
            "dropout": dropout,
            "attention_mode": "bidirectional"
        }


class AttentionFactory:
    """
    Factory for creating and managing attention mechanisms across layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize factory with attention configuration.
        
        Args:
            config: Attention configuration dictionary
        """
        self.config = config
        self._attention = None
    
    def get_attention(self) -> CoreAttention:
        """
        Get or create attention mechanism.
        Creates once and reuses for efficiency.
        
        Returns:
            Configured attention mechanism
        """
        if self._attention is None:
            self._attention = CoreAttention(**self.config)
        return self._attention
    
    def create_attention(self) -> CoreAttention:
        """
        Create a new attention mechanism instance.
        Useful when you need multiple instances.
        
        Returns:
            New attention mechanism instance
        """
        return CoreAttention(**self.config)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration and reset cached attention.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        self._attention = None  # Reset cached attention


def create_attention_factory(preset: str, **kwargs) -> AttentionFactory:
    """
    Create attention factory from preset.
    
    Args:
        preset: Preset name ("gpt_style", "bert_style", "efficient_gqa", "kernel_attention", "full_featured")
        **kwargs: Additional parameters to override preset defaults
        
    Returns:
        Configured attention factory
    """
    presets = {
        "gpt_style": AttentionPresets.gpt_style,
        "bert_style": AttentionPresets.bert_style,
        "efficient_gqa": AttentionPresets.efficient_gqa,
        "kernel_attention": AttentionPresets.kernel_attention,
        "full_featured": AttentionPresets.full_featured
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset](**kwargs)
    return AttentionFactory(config)


def create_layer_attention(config: Dict[str, Any]) -> CoreAttention:
    """
    Create attention mechanism for a single layer.
    
    Args:
        config: Attention configuration
        
    Returns:
        Configured attention mechanism
    """
    return CoreAttention(**config)
