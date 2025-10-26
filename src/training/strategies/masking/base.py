"""
Base abstract class for masking strategies.

This module defines the common interface that all masking strategies must implement
to ensure API compatibility with ProgressiveDataLoader and other training components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch


class BaseMaskingStrategy(ABC):
    """
    Abstract base class for masking strategies.
    
    All masking strategies must implement this interface to ensure compatibility
    with ProgressiveDataLoader and other training components.
    """
    
    def __init__(self, config):
        """
        Initialize the masking strategy.
        
        Args:
            config: Training configuration
        """
        self.config = config
        # All masking strategies must have these attributes for API compatibility
        self.time_interpretation: str = "depth"  # "depth" or "input"
        self.current_time_step: int = 0
    
    @abstractmethod
    def generate_masks_for_stack(self, batch: Dict[str, torch.Tensor], stack_idx: int) -> torch.Tensor:
        """
        Generate masks for a specific stack.
        
        This is the primary method used by ProgressiveDataLoader.
        
        Args:
            batch: Training batch containing input_ids and other data
            stack_idx: Current stack index
            
        Returns:
            Boolean mask tensor of shape (batch_size, seq_len)
        """
        pass
    
    def generate_masks(self, batch: Dict[str, torch.Tensor], layer_idx: int = 0) -> torch.Tensor:
        """
        Generate masks for a batch (alternative interface).
        
        Some masking strategies may use this as their primary method,
        but all must support generate_masks_for_stack for compatibility.
        
        Args:
            batch: Training batch
            layer_idx: Layer index (may be used as stack_idx)
            
        Returns:
            Boolean mask tensor of shape (batch_size, seq_len)
        """
        # Default implementation delegates to generate_masks_for_stack
        return self.generate_masks_for_stack(batch, layer_idx)
    
    def set_current_time_step(self, time_t: int) -> None:
        """
        Set the current time step (for time-as-input interpretation).
        
        Args:
            time_t: Time step to set
        """
        self.current_time_step = time_t
    
    def get_time_step_for_stack(self, stack_idx: int) -> int:
        """
        Get the time step for a specific stack.
        
        Args:
            stack_idx: Stack index
            
        Returns:
            Time step corresponding to the stack
        """
        # Default implementation returns current_time_step
        return self.current_time_step
    
    def clear_cache(self) -> None:
        """
        Clear any cached masks or data.
        
        Default implementation does nothing (for strategies without caching).
        """
        pass
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        # Default implementation returns empty dict
        return {}
