"""
Base attention class for StackWise.
Provides common interface and utilities for all attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
from abc import ABC, abstractmethod

AttentionMode = Literal["bidirectional", "causal"]


class BaseAttention(nn.Module, ABC):
    """
    Base class for all attention mechanisms.
    Provides common interface and utilities.
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        attention_mode: AttentionMode = "bidirectional"
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.attention_mode = attention_mode
        self.drop = nn.Dropout(dropout)
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of attention mechanism."""
        pass
    
    def create_attention_mask(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Create attention mask based on attention mode.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Attention mask or None for bidirectional attention
        """
        if self.attention_mode == "bidirectional":
            return None  # No mask for bidirectional attention
        
        elif self.attention_mode == "causal":
            # Create lower triangular causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), 
                diagonal=1
            )
            return mask.masked_fill(mask == 1, float('-inf'))
        
        else:
            raise ValueError(f"Unknown attention mode: {self.attention_mode}")
    
    def apply_attention_mask(
        self, 
        attention_scores: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention mask to attention scores.
        
        Args:
            attention_scores: Raw attention scores
            attn_mask: Optional attention mask
            
        Returns:
            Masked attention scores
        """
        if attn_mask is not None:
            attention_scores = attention_scores + attn_mask
        
        return attention_scores
    
    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Scaled dot-product attention computation.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            attn_mask: Optional attention mask
            scale: Scaling factor (default: 1/sqrt(d_k))
            
        Returns:
            Attention output
        """
        # Compute attention scores
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask
        attention_scores = self.apply_attention_mask(attention_scores, attn_mask)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.drop(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        return output
    
    def get_attention_pattern(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get attention pattern for analysis.
        
        Args:
            q: Query tensor
            k: Key tensor
            attn_mask: Optional attention mask
            scale: Scaling factor
            
        Returns:
            Attention pattern (softmax of attention scores)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention_scores = self.apply_attention_mask(attention_scores, attn_mask)
        attention_pattern = F.softmax(attention_scores, dim=-1)
        
        return attention_pattern
    
    def set_attention_mode(self, mode: AttentionMode) -> None:
        """Set attention mode (bidirectional or causal)."""
        if mode not in ["bidirectional", "causal"]:
            raise ValueError(f"Invalid attention mode: {mode}")
        self.attention_mode = mode
    
    def get_attention_mode(self) -> AttentionMode:
        """Get current attention mode."""
        return self.attention_mode
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"d_model={self.d_model}, dropout={self.dropout}, mode={self.attention_mode}"
