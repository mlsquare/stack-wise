"""
Core Attention implementation.
Unified attention mechanism supporting MHA, GQA, MLA, and kernel-based attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal
from .kernels import KernelType, create_kernel_matrix, apply_kernel, get_kernel_info

AttentionMode = Literal["bidirectional", "causal"]


class CoreAttention(nn.Module):
    """
    Core Attention implementation supporting MHA, GQA, MLA, and kernel-based attention.
    
    Unified attention mechanism that can be configured for:
    - Multi-Head Attention (MHA) - when attention_type="mha"
    - Grouped Query Attention (GQA) - determined by n_kv_heads < n_heads
    - Multi-Latent Attention (MLA) - when attention_type="mla"
    - Kernel-based attention (Linear, Gaussian, Laplacian, Uniform)
    - Scaled dot-product attention (special case of linear kernel)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,
        r_q: int = None,
        r_kv: int = None,
        kernel_dim: int = 64,
        kernel_type: KernelType = "linear",
        dropout: float = 0.0,
        attention_mode: str = "bidirectional"
    ):
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.d_model = d_model
        self.dropout = dropout
        self.attention_mode = attention_mode
        self.drop = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads ({self.n_kv_heads}) cannot exceed n_heads ({self.n_heads})")
        if self.n_kv_heads < self.n_heads and self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads when using grouped attention")
        self.d_k = d_model // n_heads
        self.kernel_dim = kernel_dim
        self.kernel_type = kernel_type
        self.group_size = n_heads // self.n_kv_heads if self.n_kv_heads < n_heads else 1

        # Low-rank parameters for MLA-style attention
        self.r_q = r_q
        self.r_kv = r_kv
        self.use_low_rank = r_q is not None and r_kv is not None
        
        # Linear projections based on attention type (MLA)
        if self.use_low_rank:
            # Low-rank projections for MLA-style
            self.W_q1 = nn.Linear(d_model, r_q, bias=False)
            self.W_q2 = nn.Linear(r_q, n_heads * self.d_k, bias=False)
            self.W_k1 = nn.Linear(d_model, r_kv, bias=False)
            self.W_k2 = nn.Linear(r_kv, self.n_kv_heads * self.d_k, bias=False)
            self.W_v1 = nn.Linear(d_model, r_kv, bias=False)
            self.W_v2 = nn.Linear(r_kv, self.n_kv_heads * self.d_k, bias=False)
        else:
            # Standard projections
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            if self.n_kv_heads < n_heads:
                # GQA-style projections
                self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
                self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
            else:
                # Standard projections
                self.W_k = nn.Linear(d_model, d_model, bias=False)
                self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Random kernel matrices (fixed, not trainable) - only for non-linear kernels
        if kernel_type != "linear":
            if self.kernel_dim <= 0:
                raise ValueError("kernel_dim must be positive for kernel attention")
            self.register_buffer('kernel_matrix', create_kernel_matrix(self.kernel_type, self.kernel_dim, self.d_k))
        else:
            self.register_buffer('kernel_matrix', torch.empty(1))  # Placeholder
    
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of unified kernel-based attention.
        
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
        
        # Linear projections based on attention type
        if self.use_low_rank:
            # Low-rank projections for MLA-style
            q_latent = self.W_q1(x)
            q = self.W_q2(q_latent)
            k_latent = self.W_k1(x)
            k = self.W_k2(k_latent)
            v_latent = self.W_v1(x)
            v = self.W_v2(v_latent)
        else:
            # Standard projections
            q = self.W_q(x)
            k = self.W_k(x)
            v = self.W_v(x)
        
        # Reshape queries
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Reshape keys and values (handle GQA grouping)
        if self.n_kv_heads < self.n_heads:
            # GQA-style: fewer KV heads
            k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
            # Shape: (batch_size, n_kv_heads, seq_len, d_k)
            
            # Repeat keys and values for each group
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
            # Shape: (batch_size, n_heads, seq_len, d_k)
        else:
            # Standard: same number of heads
            k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Compute attention scores based on kernel type
        if self.kernel_type == "linear":
            # Direct scaled dot-product attention (no kernel transformation needed)
            attention_scores = torch.matmul(q, k.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.d_k)
        else:
            # Apply kernel transformation for non-linear kernels
            q_kernel = apply_kernel(q, self.kernel_matrix, self.kernel_type)
            k_kernel = apply_kernel(k, self.kernel_matrix, self.kernel_type)
            attention_scores = torch.matmul(q_kernel, k_kernel.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.kernel_dim)
        
        # Apply attention mask
        attention_scores = self.apply_attention_mask(attention_scores, attn_mask)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
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
        
        # Linear projections based on attention type
        if self.use_low_rank:
            # Low-rank projections for MLA-style
            q_latent = self.W_q1(x)
            q = self.W_q2(q_latent)
            k_latent = self.W_k1(x)
            k = self.W_k2(k_latent)
        else:
            # Standard projections
            q = self.W_q(x)
            k = self.W_k(x)
        
        # Reshape queries
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Reshape keys (handle GQA grouping)
        if self.n_kv_heads < self.n_heads:
            # GQA-style: fewer KV heads
            k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
            # Repeat keys for each group
            k = k.repeat_interleave(self.group_size, dim=1)
        else:
            # Standard: same number of heads
            k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores based on kernel type
        if self.kernel_type == "linear":
            # Direct scaled dot-product attention (no kernel transformation needed)
            attention_scores = torch.matmul(q, k.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.d_k)
        else:
            # Apply kernel transformation for non-linear kernels
            q_kernel = apply_kernel(q, self.kernel_matrix, self.kernel_type)
            k_kernel = apply_kernel(k, self.kernel_matrix, self.kernel_type)
            attention_scores = torch.matmul(q_kernel, k_kernel.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.kernel_dim)
        
        attention_scores = self.apply_attention_mask(attention_scores, attn_mask)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights
    
    def get_kernel_info(self) -> dict:
        """
        Get information about the kernel transformation.
        
        Returns:
            Dictionary with kernel information
        """
        info = get_kernel_info(self.kernel_type, self.kernel_dim, self.d_k)
        info["matrix_shape"] = self.kernel_matrix.shape
        return info
    
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
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"kernel_dim={self.kernel_dim}, kernel_type={self.kernel_type}, "
                f"mode={self.attention_mode}")
    
    @classmethod
    def from_config(cls, config) -> 'CoreAttention':
        """
        Create CoreAttention from configuration object.
        
        Args:
            config: Configuration object with model parameters
            
        Returns:
            Configured CoreAttention instance
        """
        # Determine if we're using MLA based on attention_preset
        use_mla = config.attention_preset == "mla_attention"
        
        # Set low-rank parameters for MLA
        r_q = config.mla_rq if use_mla else None
        r_kv = config.mla_rkv if use_mla else None
        
        # Get kernel parameters from attention_custom if available
        kernel_dim = getattr(config, 'kernel_dim', 64)
        kernel_type = getattr(config, 'kernel_type', 'linear')
        attention_mode = getattr(config, 'attention_mode', 'bidirectional')
        
        # If attention_custom is available, use those values
        if hasattr(config, 'attention_custom') and config.attention_custom:
            if isinstance(config.attention_custom, dict):
                kernel_dim = config.attention_custom.get('kernel_dim', kernel_dim)
                kernel_type = config.attention_custom.get('kernel_type', kernel_type)
                attention_mode = config.attention_custom.get('attention_mode', attention_mode)
            else:
                kernel_dim = getattr(config.attention_custom, 'kernel_dim', kernel_dim)
                kernel_type = getattr(config.attention_custom, 'kernel_type', kernel_type)
                attention_mode = getattr(config.attention_custom, 'attention_mode', attention_mode)
        
        return cls(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            r_q=r_q,
            r_kv=r_kv,
            kernel_dim=kernel_dim,
            kernel_type=kernel_type,
            dropout=config.dropout,
            attention_mode=attention_mode
        )
