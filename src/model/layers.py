"""
Consolidated layer implementations for StackWise.

This module contains all layer types:
- LexicalKernelManager: Handles pre-trained embeddings and lexical kernels
- SwiGLUFFN: SwiGLU Feed-Forward Network with optional freezing
- MLGKALayer: Multi-Latent Grouped Kernel Attention Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

# transformers is an optional dependency for tokenizer / pretrained embedding
# support. Import lazily and provide a clear error only when the functionality
# is actually requested (LexicalKernelManager uses AutoTokenizer/AutoModel).
try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None
import logging

logger = logging.getLogger(__name__)


class ModelFamily:
    """Model family registry for supported pre-trained models"""
    
    def __init__(self, name: str, embedding_dim: int, vocab_size: int, 
                 model_path: str, requires_auth: bool = False):
        self.name = name
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.requires_auth = requires_auth
    
    def __repr__(self):
        return f"ModelFamily(name='{self.name}', dim={self.embedding_dim}, vocab={self.vocab_size})"


# Supported model families
SUPPORTED_FAMILIES = {
    "llama-3-8b": ModelFamily(
        name="llama-3-8b",
        embedding_dim=4096,
        vocab_size=128256,
        model_path="meta-llama/Llama-3-8B",
        requires_auth=True
    ),
    "llama-3-70b": ModelFamily(
        name="llama-3-70b", 
        embedding_dim=8192,
        vocab_size=128256,
        model_path="meta-llama/Llama-3-70B",
        requires_auth=True
    ),
    "mistral-7b": ModelFamily(
        name="mistral-7b",
        embedding_dim=4096,
        vocab_size=32000,
        model_path="mistralai/Mistral-7B-v0.1",
        requires_auth=False
    ),
    "gpt2": ModelFamily(
        name="gpt2",
        embedding_dim=768,
        vocab_size=50257,
        model_path="gpt2",
        requires_auth=False
    ),
    "gpt2-medium": ModelFamily(
        name="gpt2-medium",
        embedding_dim=1024,
        vocab_size=50257,
        model_path="gpt2-medium",
        requires_auth=False
    )
}


class LexicalKernelManager(nn.Module):
    """
    Manages pre-trained tokenizers and embeddings for lexical kernel initialization.
    
    This class handles:
    - Loading pre-trained tokenizers and embeddings
    - Creating optional MLP adapter for dimension matching
    - Computing lexical kernel for attention mechanisms
    """
    
    def __init__(self, 
                 family: str,
                 embedding_option: str = "embed_tokens",
                 freeze_embeddings: bool = True,
                 target_model_dim: Optional[int] = None,
                 adapter_hidden_dim: Optional[int] = None):
        """
        Initialize lexical kernel manager.
        
        Args:
            family: Model family name (e.g., 'llama-3-8b', 'mistral-7b')
            embedding_option: Which embedding to use ('embed_tokens', 'lm_head')
            freeze_embeddings: Whether to freeze the pre-trained embeddings
            target_model_dim: Target model dimension (if different from embedding dim)
            adapter_hidden_dim: Hidden dimension for MLP adapter
        """
        super().__init__()
        
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(f"Unsupported family: {family}. Supported: {list(SUPPORTED_FAMILIES.keys())}")
        
        self.family_info = SUPPORTED_FAMILIES[family]
        self.embedding_option = embedding_option
        self.freeze_embeddings = freeze_embeddings
        self.target_model_dim = target_model_dim or self.family_info.embedding_dim
        self.adapter_hidden_dim = adapter_hidden_dim or self.target_model_dim
        
        # Load tokenizer and model
        self._load_tokenizer_and_model()
        
        # Create MLP adapter if needed
        self.adapter = None
        if self.family_info.embedding_dim != self.target_model_dim:
            self._create_adapter()
        
        # Freeze embeddings if requested
        if self.freeze_embeddings:
            self._freeze_embeddings()
    
    def _load_tokenizer_and_model(self):
        """Load tokenizer and extract embeddings"""
        # Guard against missing optional dependency
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "transformers is required to load pretrained tokenizers/embeddings. "
                "Install it with: pip install transformers sentencepiece"
            )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.family_info.model_path,
                trust_remote_code=True
            )

            # Load model to extract embeddings
            model = AutoModel.from_pretrained(
                self.family_info.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for compatibility
            )

            # Extract embeddings based on option
            if self.embedding_option == "embed_tokens":
                self.embeddings = model.get_input_embeddings()
            elif self.embedding_option == "lm_head":
                self.embeddings = model.get_output_embeddings()
            else:
                raise ValueError(f"Unknown embedding option: {self.embedding_option}")

            # Clean up model to save memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Failed to load {self.family_info.name}: {e}")
            raise
    
    def _create_adapter(self):
        """Create MLP adapter for dimension matching"""
        self.adapter = nn.Sequential(
            nn.Linear(self.family_info.embedding_dim, self.adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(self.adapter_hidden_dim, self.target_model_dim)
        )
        logger.info(f"Created adapter: {self.family_info.embedding_dim} -> {self.target_model_dim}")
    
    def _freeze_embeddings(self):
        """Freeze pre-trained embeddings"""
        for param in self.embeddings.parameters():
            param.requires_grad = False
        logger.info(f"Frozen {self.family_info.name} embeddings")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embeddings and optional adapter.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, target_model_dim)
        """
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        # Apply adapter if present
        if self.adapter is not None:
            embeddings = self.adapter(embeddings)
        
        return embeddings
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.family_info.vocab_size
    
    def get_embedding_dim(self) -> int:
        """Get original embedding dimension"""
        return self.family_info.embedding_dim
    
    def get_lm_head(self) -> nn.Module:
        """Get language model head using transposed embeddings"""
        # Use the same embedding matrix as the output projection
        # This is a common technique in transformer models
        return self.embeddings
    
    def get_model_dim(self) -> int:
        """Get target model dimension"""
        return self.target_model_dim
    
    def get_adapter_info(self) -> Optional[Dict]:
        """Get adapter information"""
        if self.adapter is None:
            return None
        
        return {
            "input_dim": self.family_info.embedding_dim,
            "hidden_dim": self.adapter_hidden_dim,
            "output_dim": self.target_model_dim,
            "parameters": sum(p.numel() for p in self.adapter.parameters())
        }
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        return sum(p.numel() for p in self.parameters())
    
    def get_frozen_parameter_count(self) -> int:
        """Get frozen parameter count"""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with optionally frozen projections"""
    
    def __init__(self, d_model: int, d_ff: int, freeze_up_proj: bool = True):
        super().__init__()
        
        # Trainable projections
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        # Up projection (optionally frozen)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.freeze_up_proj = freeze_up_proj
        
        if freeze_up_proj:
            self._freeze_up_projection()
    
    def _freeze_up_projection(self):
        """Freeze the up-projection with random initialization"""
        self.up_proj.requires_grad_(False)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass: down_proj(gate_proj(x) * silu(up_proj(x)))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        silu_up = F.silu(up)
        
        # Element-wise multiplication
        swiglu = gate * silu_up
        
        # Down projection
        return self.down_proj(swiglu)


class MLGKALayer(nn.Module):
    """
    Multi-Latent Grouped Kernel Attention Layer.
    
    Combines CoreAttention with SwiGLUFFN to form a complete Transformer layer.
    Uses MLGKA preset configuration for attention.
    """
    
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 n_kv_heads: Optional[int] = None,
                 kernel_type: str = "laplacian",
                 kernel_dim: int = 64,
                 attention_mode: str = "bidirectional",
                 freeze_up_proj: bool = True):
        """
        Initialize MLGKA layer.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads (for GQA)
            kernel_type: Type of kernel attention
            kernel_dim: Kernel dimension
            attention_mode: Attention mode (bidirectional/causal)
            freeze_up_proj: Whether to freeze up-projection in SwiGLU
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from .attention.attention import CoreAttention
        from .attention.presets import AttentionPresets
        
        # Get MLGKA preset configuration
        attention_config = AttentionPresets.mlgka(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            r_q=64,  # Default query rank for MLA
            r_kv=64,  # Default key-value rank for MLA
            kernel_dim=kernel_dim
        )
        
        # Create attention module using from_config method
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        config_obj = Config(attention_config)
        self.attention = CoreAttention.from_config(config_obj)
        
        # Create feed-forward network
        self.ffn = SwiGLUFFN(d_model, d_ff, freeze_up_proj=freeze_up_proj)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Store configuration
        self.kernel_type = attention_config["kernel_type"]
        self.kernel_dim = attention_config["kernel_dim"]
        self.attention_mode = attention_config["attention_mode"]
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_kv_heads = attention_config.get("n_kv_heads", n_heads)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through MLGKA layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, attn_mask=attention_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
    
    def get_attention_info(self) -> Dict:
        """Get attention configuration information"""
        return {
            "type": "MLGKA",
            "preset": "mlgka",
            "kernel_type": self.kernel_type,
            "kernel_dim": self.kernel_dim,
            "attention_mode": self.attention_mode,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff
        }
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """Get trainable parameter count"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
