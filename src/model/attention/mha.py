"""
Standard Multi-Head Attention implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseAttention


class MHA(BaseAttention):
    """
    Standard Multi-Head Attention.
    
    Implements the original attention mechanism from "Attention Is All You Need".
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attention_mode: str = "bidirectional"
    ):
        super().__init__(d_model, dropout, attention_mode)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of standard multi-head attention.
        
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
        k = self.W_k(x)  # (batch_size, seq_len, d_model)
        v = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
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
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Get attention pattern
        attention_weights = self.get_attention_pattern(q, k, attn_mask)
        
        return attention_weights
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"d_model={self.d_model}, n_heads={self.n_heads}, d_k={self.d_k}, mode={self.attention_mode}"
