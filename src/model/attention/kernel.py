"""
Kernel-based Attention implementation.
Uses Random Kitchen Sinks for efficient attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal
from .base import BaseAttention

KernelType = Literal["gaussian", "laplacian", "uniform"]


class KernelAttention(BaseAttention):
    """
    Kernel-based Attention using Random Kitchen Sinks.
    
    Replaces standard attention with kernel-based computation for efficiency.
    Supports different kernel types: Gaussian, Laplacian, and Uniform.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kernel_dim: int = 64,
        kernel_type: KernelType = "gaussian",
        dropout: float = 0.0,
        attention_mode: str = "bidirectional"
    ):
        super().__init__(d_model, dropout, attention_mode)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert kernel_dim > 0, "kernel_dim must be positive"
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.kernel_dim = kernel_dim
        self.kernel_type = kernel_type
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Random kernel matrices (fixed, not trainable)
        self.register_buffer('kernel_matrix', self._create_kernel_matrix())
    
    def _create_kernel_matrix(self) -> torch.Tensor:
        """Create random kernel matrix based on kernel type."""
        if self.kernel_type == "gaussian":
            # Gaussian random matrix
            matrix = torch.randn(self.kernel_dim, self.d_k)
        elif self.kernel_type == "laplacian":
            # Laplacian random matrix (using exponential distribution)
            matrix = -torch.log(torch.rand(self.kernel_dim, self.d_k))
        elif self.kernel_type == "uniform":
            # Uniform random matrix
            matrix = torch.rand(self.kernel_dim, self.d_k) * 2 - 1
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return matrix
    
    def _apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel transformation using Random Kitchen Sinks.
        
        Args:
            x: Input tensor of shape (..., d_k)
            
        Returns:
            Kernel-transformed tensor of shape (..., kernel_dim)
        """
        # Apply kernel transformation: phi(x) = cos(x @ W^T)
        x_proj = torch.matmul(x, self.kernel_matrix.T)
        return torch.cos(x_proj)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of kernel-based attention.
        
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
        
        # Apply kernel transformation
        q_kernel = self._apply_kernel(q)  # (batch_size, n_heads, seq_len, kernel_dim)
        k_kernel = self._apply_kernel(k)  # (batch_size, n_heads, seq_len, kernel_dim)
        
        # Compute attention using kernel features
        # Attention weights = softmax(q_kernel @ k_kernel^T)
        attention_scores = torch.matmul(q_kernel, k_kernel.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.kernel_dim)
        
        # Apply attention mask
        attention_scores = self.apply_attention_mask(attention_scores, attn_mask)
        
        # Apply softmax
        # optional
        # attention_weights = F.softmax(attention_scores, dim=-1)
        
        attention_weights = self.drop(attention_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attention_weights, v)
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
        
        # Apply kernel transformation
        q_kernel = self._apply_kernel(q)
        k_kernel = self._apply_kernel(k)
        
        # Compute attention scores
        attention_scores = torch.matmul(q_kernel, k_kernel.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.kernel_dim)
        attention_scores = self.apply_attention_mask(attention_scores, attn_mask)
        
        # Apply softmax
        # attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights
    
    def get_kernel_info(self) -> dict:
        """
        Get information about the kernel transformation.
        
        Returns:
            Dictionary with kernel information
        """
        return {
            "kernel_type": self.kernel_type,
            "kernel_dim": self.kernel_dim,
            "d_k": self.d_k,
            "compression_ratio": self.kernel_dim / self.d_k,
            "matrix_shape": self.kernel_matrix.shape
        }
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"kernel_dim={self.kernel_dim}, kernel_type={self.kernel_type}, "
                f"mode={self.attention_mode}")
