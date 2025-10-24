"""
Fusion-specific training logic for progressive training strategies.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class FusionTrainer:
    """
    Fusion-specific training logic for progressive training strategies.
    
    This class handles fusion training modes including frozen and trainable backbones,
    quantized loading, QLoRA adapters, and time-step-based masking.
    """
    
    def __init__(self, config, masking_strategy=None, quantization_manager=None, 
                 cache_manager=None, lexical_kernel_manager=None):
        """
        Initialize FusionTrainer.
        
        Args:
            config: Training configuration
            masking_strategy: Strategy for generating masks
            quantization_manager: Manager for quantization operations
            cache_manager: Manager for activation caching
            lexical_kernel_manager: Manager for lexical kernel operations
        """
        self.config = config
        self.masking_strategy = masking_strategy
        self.quantization_manager = quantization_manager
        self.cache_manager = cache_manager
        self.lexical_kernel_manager = lexical_kernel_manager
        
        # Initialize caches for persistent quantization
        self._quantized_backbone_cache = {}
        self._qlora_cache = {}
        self._trained_blocks_cache = {}
        
        logger.info("Initialized FusionTrainer")


class QLoRAAdapter(nn.Module):
    """
    QLoRA (Quantized Low-Rank Adaptation) adapter for efficient fine-tuning.
    
    This adapter wraps a quantized layer and adds low-rank adaptation matrices
    for efficient fine-tuning while maintaining the original layer's functionality.
    """
    
    def __init__(self, original_layer: nn.Module, rank: int = 16, alpha: float = 32.0, 
                 dropout: float = 0.1, name: str = "qlora_adapter"):
        """
        Initialize QLoRA adapter.
        
        Args:
            original_layer: The original quantized layer to adapt
            rank: Rank of the low-rank adaptation matrices
            alpha: Scaling factor for the adaptation
            dropout: Dropout rate for the adaptation
            name: Name for the adapter
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.name = name
        
        # Get dimensions from original layer
        if hasattr(original_layer, 'weight'):
            self.in_features = original_layer.weight.shape[1]
            self.out_features = original_layer.weight.shape[0]
        else:
            raise ValueError("Original layer must have a weight attribute")
        
        # Create low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize B to zero so that initial adaptation is zero
        nn.init.zeros_(self.lora_B)
        
        logger.debug(f"Created QLoRA adapter {name}: {self.in_features} -> {self.out_features}, rank={rank}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through QLoRA adapter.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with low-rank adaptation
        """
        # Original layer forward pass
        original_output = self.original_layer(x)
        
        # Low-rank adaptation
        # x @ A^T @ B^T = x @ (B @ A)^T
        adaptation = self.dropout_layer(x) @ self.lora_A.T @ self.lora_B.T
        
        # Scale by alpha/rank
        scaled_adaptation = adaptation * (self.alpha / self.rank)
        
        # Add adaptation to original output
        return original_output + scaled_adaptation
    
    def get_adaptation_parameters(self):
        """
        Get only the adaptation parameters (A and B matrices).
        
        Returns:
            List of adaptation parameters
        """
        return [self.lora_A, self.lora_B]
    
    def freeze_original_layer(self):
        """
        Freeze the original layer parameters.
        """
        for param in self.original_layer.parameters():
            param.requires_grad = False
        logger.debug(f"Frozen original layer in {self.name}")

# Removed duplicate FusionTrainer class - using the first one defined above
