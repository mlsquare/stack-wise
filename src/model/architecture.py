"""
StackWise Architecture: Block, Stack, and Rack

This module implements the hierarchical architecture:
- Block: Standard transformer block (attention + FFN + layer norm + residual)
- Stack: Collection of multiple Blocks
- Rack: Final model containing multiple Stacks

The naming follows the physical analogy:
- Block = Standard transformer block
- Stack = Multiple blocks stacked together
- Rack = Multiple stacks in a rack (the final model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class Block(nn.Module):
    """
    Standard Transformer Block.
    
    Combines:
    - Self-attention mechanism
    - Feed-forward network (FFN)
    - Layer normalization
    - Residual connections
    
    This is the fundamental building block of the transformer architecture.
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 n_kv_heads: Optional[int] = None,
                 attention_type: str = "mha",
                 kernel_type: str = "linear",
                 kernel_dim: int = 64,
                 attention_mode: str = "bidirectional",
                 dropout: float = 0.0,
                 freeze_up_proj: bool = True,
                 use_rope: bool = True,
                 rope_theta: float = 10000.0):
        """
        Initialize a Transformer Block.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads (GQA when < n_heads)
            attention_type: Type of attention (mha, mla)
            kernel_type: Type of kernel for kernel attention
            kernel_dim: Kernel dimension
            attention_mode: Attention mode (bidirectional/causal)
            dropout: Dropout probability
            freeze_up_proj: Whether to freeze up-projection in SwiGLU
            use_rope: Whether to use RoPE positional encoding
            rope_theta: RoPE theta parameter
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from .attention.attention import CoreAttention
        from .attention.presets import AttentionPresets
        from .layers import SwiGLUFFN
        
        # Create attention mechanism
        if attention_type == "mha":
            # MHA with optional GQA (determined by n_kv_heads < n_heads)
            attention_config = AttentionPresets.bert_style(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            # Override n_kv_heads if provided (enables GQA)
            if n_kv_heads is not None:
                attention_config["n_kv_heads"] = n_kv_heads
            # Override attention_mode if provided
            attention_config["attention_mode"] = attention_mode
        elif attention_type == "mla":
            # MLA with optional GQA (determined by n_kv_heads < n_heads)
            attention_config = AttentionPresets.mla_attention(
                d_model=d_model,
                n_heads=n_heads,
                r_q=64,  # Default query rank for MLA
                r_kv=64,  # Default key-value rank for MLA
                dropout=dropout
            )
            # Override n_kv_heads if provided (enables GQA)
            if n_kv_heads is not None:
                attention_config["n_kv_heads"] = n_kv_heads
            # Override attention_mode if provided
            attention_config["attention_mode"] = attention_mode
            # Override kernel settings if provided
            if kernel_type != "linear":
                attention_config["kernel_type"] = kernel_type
                attention_config["kernel_dim"] = kernel_dim
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        # Create attention module using from_config method
        # Convert dict to object for from_config method
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store configuration
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_kv_heads = attention_config.get("n_kv_heads", n_heads)
        self.attention_type = attention_type
        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.rope_theta = rope_theta
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
    
    def get_attention_info(self) -> Dict:
        """Get attention configuration information"""
        return {
            "type": self.attention_type,
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


class Stack(nn.Module):
    """
    Stack of Transformer Blocks.
    
    A Stack contains multiple Blocks and represents a logical grouping
    of transformer layers. This is useful for:
    - Layer-wise training
    - Block-wise training
    - Fusion training
    - Memory management
    """
    
    def __init__(self,
                 blocks: List[Block],
                 stack_id: int = 0,
                 freeze_blocks: bool = False):
        """
        Initialize a Stack.
        
        Args:
            blocks: List of Block instances
            stack_id: Identifier for this stack
            freeze_blocks: Whether to freeze all blocks in this stack
        """
        super().__init__()
        
        self.blocks = nn.ModuleList(blocks)
        self.stack_id = stack_id
        self.num_blocks = len(blocks)
        
        # Freeze blocks if requested
        if freeze_blocks:
            self.freeze_all_blocks()
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all blocks in the stack.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return x
    
    def forward_from_block(self, x: torch.Tensor, start_block: int, 
                          end_block: Optional[int] = None, 
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass from a specific block to another block.
        
        Args:
            x: Input tensor
            start_block: Starting block index
            end_block: Ending block index (inclusive, None for all remaining)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        if end_block is None:
            end_block = self.num_blocks
        
        for i in range(start_block, end_block):
            x = self.blocks[i](x, attention_mask=attention_mask)
        return x
    
    def freeze_all_blocks(self):
        """Freeze all blocks in this stack"""
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        logger.info(f"Frozen all {self.num_blocks} blocks in stack {self.stack_id}")
    
    def unfreeze_all_blocks(self):
        """Unfreeze all blocks in this stack"""
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(f"Unfrozen all {self.num_blocks} blocks in stack {self.stack_id}")
    
    def freeze_blocks_from(self, start_block: int):
        """Freeze blocks starting from a specific index"""
        for i in range(start_block, self.num_blocks):
            for param in self.blocks[i].parameters():
                param.requires_grad = False
        logger.info(f"Frozen blocks {start_block} to {self.num_blocks-1} in stack {self.stack_id}")
    
    def get_block(self, index: int) -> Block:
        """Get a specific block by index"""
        return self.blocks[index]
    
    def get_parameter_count(self) -> int:
        """Get total parameter count for all blocks"""
        return sum(block.get_parameter_count() for block in self.blocks)
    
    def get_trainable_parameter_count(self) -> int:
        """Get trainable parameter count for all blocks"""
        return sum(block.get_trainable_parameter_count() for block in self.blocks)


class Rack(nn.Module):
    """
    Rack: The final model containing multiple Stacks.
    
    A Rack represents the complete transformer model and contains:
    - Input embeddings
    - Multiple Stacks of Blocks
    - Output layer (language model head)
    - Positional encoding (if used)
    """
    
    def __init__(self,
                 stacks: List[Stack],
                 vocab_size: int,
                 d_model: int,
                 embedding_layer: Optional[nn.Module] = None,
                 tie_embeddings: bool = True,
                 use_rope: bool = True,
                 rope_theta: float = 10000.0):
        """
        Initialize a Rack (complete model).
        
        Args:
            stacks: List of Stack instances
            vocab_size: Vocabulary size
            d_model: Model dimension
            embedding_layer: Custom embedding layer (optional)
            tie_embeddings: Whether to tie input and output embeddings
            use_rope: Whether to use RoPE positional encoding
            rope_theta: RoPE theta parameter
        """
        super().__init__()
        
        self.stacks = nn.ModuleList(stacks)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tie_embeddings = tie_embeddings
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        
        # Create or use provided embedding layer
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            # Handle None vocab_size (common in config files)
            if vocab_size is None:
                vocab_size = 32000  # Default vocabulary size
            if d_model is None:
                raise ValueError("d_model cannot be None")
            self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create output layer
        if tie_embeddings:
            # Use the same weights as input embeddings
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # RoPE positional encoding (if used)
        if use_rope:
            self.apply_rope = self._apply_rope
        else:
            self.apply_rope = None
    
    def _apply_rope(self, x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
        """
        Apply Rotary Position Embedding (RoPE) to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            theta: RoPE theta parameter
            
        Returns:
            Tensor with RoPE applied
        """
        # Simple RoPE implementation - for now just return the input
        # TODO: Implement proper RoPE
        return x
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the complete Rack.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Apply RoPE if used
        if self.apply_rope is not None:
            x = self.apply_rope(x, self.rope_theta)
        
        # Forward through all stacks
        for stack in self.stacks:
            x = stack(x, attention_mask=attention_mask)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        return logits
    
    def forward_from_stack(self, x: torch.Tensor, start_stack: int, 
                          end_stack: Optional[int] = None,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass from a specific stack to another stack.
        
        Args:
            x: Input tensor (already embedded)
            start_stack: Starting stack index
            end_stack: Ending stack index (None for all remaining)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        if end_stack is None:
            end_stack = len(self.stacks)
        
        for i in range(start_stack, end_stack):
            x = self.stacks[i](x, attention_mask=attention_mask)
        return x
    
    def get_embedding_layer(self) -> nn.Module:
        """Get the embedding layer"""
        return self.embedding
    
    def get_lm_head(self) -> nn.Module:
        """Get the language model head"""
        return self.lm_head
    
    def get_stack(self, index: int) -> Stack:
        """Get a specific stack by index"""
        return self.stacks[index]
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        stack_params = sum(stack.get_parameter_count() for stack in self.stacks)
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters())
        return stack_params + embedding_params + lm_head_params + ln_f_params
    
    def get_trainable_parameter_count(self) -> int:
        """Get trainable parameter count"""
        stack_params = sum(stack.get_trainable_parameter_count() for stack in self.stacks)
        embedding_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters() if p.requires_grad)
        return stack_params + embedding_params + lm_head_params + ln_f_params


def create_block_spec(d_model: int, d_ff: int, n_heads: int, 
                     n_kv_heads: Optional[int] = None,
                     attention_type: str = "mha",
                     kernel_type: str = "linear",
                     kernel_dim: int = 64,
                     attention_mode: str = "bidirectional",
                     dropout: float = 0.0,
                     freeze_up_proj: bool = True,
                     use_rope: bool = True,
                     rope_theta: float = 10000.0) -> Dict:
    """
    Create a block specification dictionary.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (GQA when < n_heads)
        attention_type: Type of attention (mha, mla)
        kernel_type: Type of kernel for kernel attention
        kernel_dim: Kernel dimension
        attention_mode: Attention mode (bidirectional/causal)
        dropout: Dropout probability
        freeze_up_proj: Whether to freeze up-projection in SwiGLU
        use_rope: Whether to use RoPE positional encoding
        rope_theta: RoPE theta parameter
        
    Returns:
        Block specification dictionary
    """
    return {
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "attention_type": attention_type,
        "kernel_type": kernel_type,
        "kernel_dim": kernel_dim,
        "attention_mode": attention_mode,
        "dropout": dropout,
        "freeze_up_proj": freeze_up_proj,
        "use_rope": use_rope,
        "rope_theta": rope_theta
    }


def create_stack_from_spec(stack_id: int, n_blocks: int, block_spec: Dict, 
                          freeze_blocks: bool = False) -> Stack:
    """
    Create a Stack from block specification.
    
    Args:
        stack_id: Identifier for this stack
        n_blocks: Number of blocks in this stack
        block_spec: Block specification dictionary
        freeze_blocks: Whether to freeze all blocks in this stack
        
    Returns:
        Stack instance
    """
    # Create identical blocks
    blocks = []
    for i in range(n_blocks):
        block = Block(**block_spec)
        blocks.append(block)
    
    # Create stack
    stack = Stack(blocks, stack_id=stack_id, freeze_blocks=freeze_blocks)
    return stack


def create_stack(stack_id: int, n_blocks: int, d_model: int, d_ff: int, n_heads: int,
                 n_kv_heads: Optional[int] = None,
                      attention_type: str = "mha",
                 kernel_type: str = "linear",
                 kernel_dim: int = 64,
                 attention_mode: str = "bidirectional",
                 dropout: float = 0.0,
                 freeze_up_proj: bool = True,
                 use_rope: bool = True,
                 rope_theta: float = 10000.0,
                 freeze_blocks: bool = False) -> Stack:
    """
    Create a Stack directly from block parameters (simplified version).
    
    Args:
        stack_id: Identifier for this stack
        n_blocks: Number of blocks in this stack
        d_model: Model dimension
        d_ff: Feed-forward dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (GQA when < n_heads)
        attention_type: Type of attention (mha, mla)
        kernel_type: Type of kernel for kernel attention
        kernel_dim: Kernel dimension
        attention_mode: Attention mode (bidirectional/causal)
        dropout: Dropout probability
        freeze_up_proj: Whether to freeze up projection in FFN
        use_rope: Whether to use RoPE positional encoding
        rope_theta: RoPE theta parameter
        freeze_blocks: Whether to freeze all blocks in this stack
        
    Returns:
        Stack instance
    """
    # Create identical blocks directly
    blocks = []
    for i in range(n_blocks):
        block = Block(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            attention_type=attention_type,
            kernel_type=kernel_type,
            kernel_dim=kernel_dim,
            attention_mode=attention_mode,
            dropout=dropout,
            freeze_up_proj=freeze_up_proj,
            use_rope=use_rope,
            rope_theta=rope_theta
        )
        blocks.append(block)
    
    # Create stack
    stack = Stack(blocks, stack_id=stack_id, freeze_blocks=freeze_blocks)
    return stack


def create_stack_from_config(stack_id: int, n_blocks: int, config, freeze_blocks: bool = False) -> Stack:
    """
    Create a Stack from configuration (config-driven approach).
    
    Args:
        stack_id: Identifier for this stack
        n_blocks: Number of blocks in this stack
        config: Configuration object containing all architectural parameters
        freeze_blocks: Whether to freeze all blocks in this stack
        
    Returns:
        Stack instance
    """
    # Read parameters from config: support passing StackWiseConfig or a model-level config
    if hasattr(config, 'model'):
        model_cfg = config.model
    else:
        model_cfg = config

    # Create identical blocks using model configuration parameters
    blocks = []
    for i in range(n_blocks):
        block = Block(
            d_model=getattr(model_cfg, 'd_model', getattr(model_cfg, 'hidden_size', 4096)),
            d_ff=getattr(model_cfg, 'd_ff', 4 * getattr(model_cfg, 'd_model', 4096)),
            n_heads=getattr(model_cfg, 'n_heads', 8),
            n_kv_heads=getattr(model_cfg, 'n_kv_heads', None),
            attention_type=getattr(model_cfg, 'attention_type', 'mha'),
            kernel_type=getattr(model_cfg, 'kernel_type', 'linear'),
            kernel_dim=getattr(model_cfg, 'kernel_dim', 64),
            attention_mode=getattr(model_cfg, 'attention_mode', 'bidirectional'),
            dropout=getattr(model_cfg, 'dropout', 0.0),
            freeze_up_proj=getattr(model_cfg, 'freeze_up_proj', True),
            use_rope=getattr(model_cfg, 'use_rope', True),
            rope_theta=getattr(model_cfg, 'rope_theta', 10000.0)
        )
        blocks.append(block)
    
    # Create stack
    stack = Stack(blocks, stack_id=stack_id, freeze_blocks=freeze_blocks)
    return stack


def create_rack_from_specs(vocab_size: int, d_model: int, stack_specs: List[Dict],
                          tie_embeddings: bool = True,
                          use_rope: bool = True,
                          rope_theta: float = 10000.0,
                          embedding_layer: Optional[nn.Module] = None) -> Rack:
    """
    Create a Rack from stack specifications.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        stack_specs: List of stack specifications, each containing:
            - stack_id: Stack identifier
            - n_blocks: Number of blocks in this stack
            - block_spec: Block specification dictionary
            - freeze_blocks: Whether to freeze blocks in this stack (optional)
        tie_embeddings: Whether to tie input and output embeddings
        use_rope: Whether to use RoPE positional encoding
        rope_theta: RoPE theta parameter
        embedding_layer: Custom embedding layer (optional)
        
    Returns:
        Rack instance
    """
    stacks = []
    
    for stack_spec in stack_specs:
        stack_id = stack_spec["stack_id"]
        n_blocks = stack_spec["n_blocks"]
        block_spec = stack_spec["block_spec"]
        freeze_blocks = stack_spec.get("freeze_blocks", False)
        
        # Create stack
        stack = create_stack_from_spec(stack_id, n_blocks, block_spec, freeze_blocks)
        stacks.append(stack)
    
    # Create rack
    rack = Rack(
        stacks=stacks,
        vocab_size=vocab_size,
        d_model=d_model,
        tie_embeddings=tie_embeddings,
        use_rope=use_rope,
        rope_theta=rope_theta,
        embedding_layer=embedding_layer
    )
    
    return rack


def create_rack_from_config(config: Dict) -> Rack:
    """
    Create a Rack from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Rack instance
    """
    # Extract configuration
    if hasattr(config, 'model'):
        # Config object
        model_config = config.model
        training_config = config.training
        arch_config = getattr(model_config, 'architecture', {})
    else:
        # Dictionary
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        if isinstance(model_config, dict):
            arch_config = model_config.get("architecture", {})
        else:
            arch_config = {}
    
    # Model parameters
    if hasattr(model_config, 'd_model'):
        # Config object
        d_model = model_config.d_model
        n_heads = model_config.n_heads
        n_kv_heads = model_config.n_kv_heads
        d_ff = model_config.d_ff
        vocab_size = model_config.vocab_size
        attention_type = model_config.attention_type
        attention_mode = model_config.attention_mode
        kernel_type = model_config.kernel_type
        kernel_dim = model_config.kernel_dim
    else:
        # Dictionary
        if isinstance(model_config, dict):
            d_model = model_config.get("d_model", 4096)
            n_heads = model_config.get("n_heads", 32)
            n_kv_heads = model_config.get("n_kv_heads", 8)
            d_ff = model_config.get("d_ff", 14336)
            vocab_size = model_config.get("vocab_size", 128000)
            attention_type = model_config.get("attention_type", "mha")
            attention_mode = model_config.get("attention_mode", "bidirectional")
            kernel_type = model_config.get("kernel_type", "linear")
            kernel_dim = model_config.get("kernel_dim", 64)
        else:
            # Default values
            d_model = 4096
            n_heads = 32
            n_kv_heads = 8
            d_ff = 14336
            vocab_size = 128000
            attention_type = "mha"
            attention_mode = "bidirectional"
            kernel_type = "linear"
            kernel_dim = 64
    
    # Architecture parameters
    if hasattr(arch_config, 'n_stacks'):
        # Config object
        n_stacks = arch_config.n_stacks
        blocks_per_stack = arch_config.blocks_per_stack
    elif isinstance(arch_config, dict):
        # Dictionary
        n_stacks = arch_config.get("n_stacks", 2)
        blocks_per_stack = arch_config.get("blocks_per_stack", 4)
    else:
        # Default values
        n_stacks = 2
        blocks_per_stack = 4
    
    # Create block specification
    block_spec = create_block_spec(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        attention_type=attention_type,
        kernel_type=kernel_type,
        kernel_dim=kernel_dim,
        attention_mode=attention_mode,
        use_rope=getattr(model_config, "use_rope", True) if hasattr(model_config, "use_rope") else True,
        rope_theta=getattr(model_config, "rope_theta", 10000.0) if hasattr(model_config, "rope_theta") else 10000.0
    )
    
    # Create stack specifications
    stack_specs = []
    for stack_id in range(n_stacks):
        stack_spec = {
            "stack_id": stack_id,
            "n_blocks": blocks_per_stack,
            "block_spec": block_spec,
            "freeze_blocks": False
        }
        stack_specs.append(stack_spec)
    
    # Create rack from specifications
    rack = create_rack_from_specs(
        vocab_size=vocab_size,
        d_model=d_model,
        stack_specs=stack_specs,
        tie_embeddings=getattr(model_config, "tie_embeddings", True) if hasattr(model_config, "tie_embeddings") else True,
        use_rope=getattr(model_config, "use_rope", True) if hasattr(model_config, "use_rope") else True,
        rope_theta=getattr(model_config, "rope_theta", 10000.0) if hasattr(model_config, "rope_theta") else 10000.0
    )
    
    return rack


def create_simple_rack(n_stacks: int, blocks_per_stack: int, 
                      d_model: int, d_ff: int, n_heads: int,
                      vocab_size: int, n_kv_heads: Optional[int] = None,
                      attention_type: str = "mha",
                      attention_mode: str = "bidirectional",
                      tie_embeddings: bool = True,
                      use_rope: bool = True,
                      rope_theta: float = 10000.0) -> Rack:
    """
    Create a simple rack with identical blocks in all stacks.
    
    Args:
        n_stacks: Number of stacks
        blocks_per_stack: Number of blocks per stack
        d_model: Model dimension
        d_ff: Feed-forward dimension
        n_heads: Number of attention heads
        vocab_size: Vocabulary size
        n_kv_heads: Number of key-value heads (GQA when < n_heads)
        attention_type: Type of attention (mha, mla)
        attention_mode: Attention mode (bidirectional/causal)
        tie_embeddings: Whether to tie input and output embeddings
        use_rope: Whether to use RoPE positional encoding
        rope_theta: RoPE theta parameter
        
    Returns:
        Rack instance
    """
    # Create block specification
    block_spec = create_block_spec(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        attention_type=attention_type,
        attention_mode=attention_mode,
        use_rope=use_rope,
        rope_theta=rope_theta
    )
    
    # Create stack specifications
    stack_specs = []
    for stack_id in range(n_stacks):
        stack_spec = {
            "stack_id": stack_id,
            "n_blocks": blocks_per_stack,
            "block_spec": block_spec,
            "freeze_blocks": False
        }
        stack_specs.append(stack_spec)
    
    # Create rack from specifications
    rack = create_rack_from_specs(
        vocab_size=vocab_size,
        d_model=d_model,
        stack_specs=stack_specs,
        tie_embeddings=tie_embeddings,
        use_rope=use_rope,
        rope_theta=rope_theta
    )
    
    return rack
