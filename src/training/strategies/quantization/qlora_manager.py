"""
QLoRA adapter management for efficient fine-tuning.

⚠️  WARNING: This module is currently UNUSED and marked for DEPRECATION.
The QLoRA classes have config attribute mismatches and are not functional.
The ProgressiveRackBuilder implements its own QLoRA logic instead of using these classes.
"""

import logging
import warnings
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

# Issue deprecation warning
warnings.warn(
    "QLoRAManager is currently unused and marked for deprecation. "
    "The QLoRA classes have config attribute mismatches and are not functional. "
    "The ProgressiveRackBuilder implements its own QLoRA logic instead of using these classes.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)


class QLoRALinear(nn.Module):
    """
    QLoRA linear layer with low-rank adaptation.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32):
        """
        Initialize QLoRA linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            rank: Rank of the adaptation matrix
            alpha: Scaling factor
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank adaptation matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through QLoRA layer."""
        return self.lora_B(self.lora_A(x)) * self.scaling


class QLoRAManager:
    """
    QLoRA adapter management for efficient fine-tuning.
    
    ⚠️  WARNING: This class is currently UNUSED and marked for DEPRECATION.
    The QLoRA classes have config attribute mismatches and are not functional.
    The ProgressiveRackBuilder implements its own QLoRA logic instead of using these classes.
    
    Handles:
    - Adding QLoRA adapters to layers
    - Managing adapter parameters
    - Selective updates (adapters only vs full model)
    """
    
    def __init__(self, config):
        """
        Initialize QLoRA manager.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.qlora_enabled = config.qlora_enabled
        self.qlora_rank = config.qlora_rank
        self.qlora_alpha = config.qlora_alpha
        self.qlora_dropout = config.qlora_dropout
        
        # Track adapters
        self.adapters = {}
        
        logger.info(f"Initialized QLoRAManager: rank={self.qlora_rank}, alpha={self.qlora_alpha}")
    
    def add_adapters_to_block(self, block_layers: List[torch.nn.Module]) -> List[torch.nn.Module]:
        """
        Add QLoRA adapters to block layers.
        
        Args:
            block_layers: Layers in the block
            
        Returns:
            Modified block layers with adapters
        """
        if not self.qlora_enabled:
            return block_layers
        
        modified_layers = []
        
        for layer_idx, layer in enumerate(block_layers):
            modified_layer = self._add_adapters_to_layer(layer, f"block_layer_{layer_idx}")
            modified_layers.append(modified_layer)
        
        logger.debug(f"Added QLoRA adapters to {len(block_layers)} layers")
        return modified_layers
    
    def _add_adapters_to_layer(self, layer: torch.nn.Module, layer_name: str) -> torch.nn.Module:
        """
        Add QLoRA adapters to a single layer.
        
        Args:
            layer: Layer to add adapters to
            layer_name: Name of the layer
            
        Returns:
            Layer with adapters
        """
        # Find linear layers in the module
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Create QLoRA adapter
                adapter = QLoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.qlora_rank,
                    alpha=self.qlora_alpha
                )
                
                # Add adapter to the layer
                setattr(layer, f"{name}_qlora", adapter)
                
                # Store adapter reference
                adapter_key = f"{layer_name}_{name}"
                self.adapters[adapter_key] = adapter
        
        return layer
    
    def get_adapter_parameters(self, block_layers: List[torch.nn.Module]) -> List[torch.nn.Parameter]:
        """
        Get QLoRA adapter parameters for optimization.
        
        Args:
            block_layers: Layers in the block
            
        Returns:
            List of adapter parameters
        """
        adapter_params = []
        
        for layer in block_layers:
            for name, module in layer.named_modules():
                if isinstance(module, QLoRALinear):
                    adapter_params.extend(module.parameters())
        
        logger.debug(f"Found {len(adapter_params)} QLoRA adapter parameters")
        return adapter_params
    
    def freeze_backbone_parameters(self, block_layers: List[torch.nn.Module]):
        """
        Freeze backbone parameters, keeping only adapters trainable.
        
        Args:
            block_layers: Layers in the block
        """
        for layer in block_layers:
            for name, param in layer.named_parameters():
                if 'qlora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        logger.debug("Frozen backbone parameters, keeping QLoRA adapters trainable")
    
    def unfreeze_all_parameters(self, block_layers: List[torch.nn.Module]):
        """
        Unfreeze all parameters for full model training.
        
        Args:
            block_layers: Layers in the block
        """
        for layer in block_layers:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.debug("Unfrozen all parameters for full model training")
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about QLoRA adapters."""
        return {
            "num_adapters": len(self.adapters),
            "qlora_rank": self.qlora_rank,
            "qlora_alpha": self.qlora_alpha,
            "qlora_enabled": self.qlora_enabled
        }
    
    def clear_adapters(self):
        """Clear all adapters."""
        self.adapters.clear()
        logger.debug("Cleared all QLoRA adapters")
