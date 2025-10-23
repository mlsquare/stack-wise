"""
Grouped Query Attention (GQA) implementation.
Reduces memory usage by sharing key-value projections across query head groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseAttention


class GQA(BaseAttention):
    """
    Grouped Query Attention.
    
    Shares key-value projections across groups of query heads to reduce memory usage.
    Useful for inference with KV-cache optimization.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        attention_mode: str = "bidirectional"
    ):
        super().__init__(d_model, dropout, attention_mode)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of grouped query attention.
        
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
        
        # Linear projections
        q = self.W_q(x)  # (batch_size, seq_len, d_model)
        k = self.W_k(x)  # (batch_size, seq_len, n_kv_heads * d_k)
        v = self.W_v(x)  # (batch_size, seq_len, n_kv_heads * d_k)
        
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
        
        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        
        # Reshape queries
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Reshape keys
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat keys for each group
        k = k.repeat_interleave(self.group_size, dim=1)
        
        # Get attention pattern
        attention_weights = self.get_attention_pattern(q, k, attn_mask)
        
        return attention_weights
    
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
                f"n_kv_heads={self.n_kv_heads}, group_size={self.group_size}, "
                f"mode={self.attention_mode}")
