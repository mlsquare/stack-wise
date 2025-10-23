"""
Multi-Latent Attention (MLA) implementation.
Uses low-rank factorization of Q/K/V projections to reduce parameters and computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseAttention


class MLA(BaseAttention):
    """
    Multi-Latent Attention.
    
    Uses low-rank factorization of Q/K/V projections to reduce parameters and computation.
    Similar to DeepSeek-V2/V3 architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        r_q: int,
        r_kv: int,
        dropout: float = 0.0,
        attention_mode: str = "bidirectional"
    ):
        super().__init__(d_model, dropout, attention_mode)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert r_q > 0, "r_q must be positive"
        assert r_kv > 0, "r_kv must be positive"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.r_q = r_q
        self.r_kv = r_kv
        
        # Low-rank projections for queries
        self.W_q1 = nn.Linear(d_model, r_q, bias=False)
        self.W_q2 = nn.Linear(r_q, n_heads * self.d_k, bias=False)
        
        # Low-rank projections for keys
        self.W_k1 = nn.Linear(d_model, r_kv, bias=False)
        self.W_k2 = nn.Linear(r_kv, n_kv_heads * self.d_k, bias=False)
        
        # Low-rank projections for values
        self.W_v1 = nn.Linear(d_model, r_kv, bias=False)
        self.W_v2 = nn.Linear(r_kv, n_kv_heads * self.d_k, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-latent attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create causal mask if needed
        if self.attention_mode == "causal" and attn_mask is None:
            attn_mask = self.create_attention_mask(seq_len, x.device)
        
        # Low-rank projections for queries
        q_latent = self.W_q1(x)  # (batch_size, seq_len, r_q)
        q = self.W_q2(q_latent)  # (batch_size, seq_len, n_heads * d_k)
        
        # Low-rank projections for keys
        k_latent = self.W_k1(x)  # (batch_size, seq_len, r_kv)
        k = self.W_k2(k_latent)  # (batch_size, seq_len, n_kv_heads * d_k)
        
        # Low-rank projections for values
        v_latent = self.W_v1(x)  # (batch_size, seq_len, r_kv)
        v = self.W_v2(v_latent)  # (batch_size, seq_len, n_kv_heads * d_k)
        
        # Reshape queries
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Reshape keys and values
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, n_kv_heads, seq_len, d_k)
        
        # Repeat keys and values for each group
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Apply attention
        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask)
        # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention weights for analysis.
        
        Args:
            x: Input tensor
            attn_mask: Optional attention mask
            
        Returns:
            Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create causal mask if needed
        if self.attention_mode == "causal" and attn_mask is None:
            attn_mask = self.create_attention_mask(seq_len, x.device)
        
        # Low-rank projections
        q_latent = self.W_q1(x)
        q = self.W_q2(q_latent)
        k_latent = self.W_k1(x)
        k = self.W_k2(k_latent)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat keys for each group
        k = k.repeat_interleave(self.group_size, dim=1)
        
        # Get attention pattern
        attention_weights = self.get_attention_pattern(q, k, attn_mask)
        
        return attention_weights
    
    def get_parameter_count(self) -> dict:
        """
        Get parameter count breakdown for analysis.
        
        Returns:
            Dictionary with parameter counts for each component
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        q_params = sum(p.weight.numel() for p in [self.W_q1, self.W_q2])
        k_params = sum(p.weight.numel() for p in [self.W_k1, self.W_k2])
        v_params = sum(p.weight.numel() for p in [self.W_v1, self.W_v2])
        o_params = self.W_o.weight.numel()
        
        return {
            "total": total_params,
            "queries": q_params,
            "keys": k_params,
            "values": v_params,
            "output": o_params,
            "compression_ratio": total_params / (self.d_model * self.d_model * 4)  # vs standard attention
        }
    
    def get_kv_cache_size(self, seq_len: int) -> int:
        """
        Get the size of KV cache for this attention mechanism.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Number of parameters in KV cache
        """
        return 2 * self.n_kv_heads * seq_len * self.d_k
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_kv_heads={self.n_kv_heads}, r_q={self.r_q}, r_kv={self.r_kv}, "
                f"mode={self.attention_mode}")
