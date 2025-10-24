"""
Time-step-based masking strategy for progressive training.
"""

import logging
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


class TimeStepMasking:
    """
    Time-step-based masking strategy for progressive training.
    
    Implements triplet-based masking: (input_id, mask_pattern, time_t)
    with discrete time steps and progressive masking fractions.
    """
    
    def __init__(self, config):
        """
        Initialize time-step masking strategy.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.num_time_steps = config.num_time_steps
        self.time_step_bins = config.time_step_bins
        self.mask_fractions = config.time_step_mask_fractions
        
        # Initialize mask storage
        self.mask_cache = {}
        
        logger.info(f"Initialized TimeStepMasking with {self.num_time_steps} time steps")
    
    def generate_masks_for_time_step(self, batch: Dict[str, torch.Tensor], time_t: int) -> torch.Tensor:
        """
        Generate masks for a specific time step.
        
        Args:
            batch: Training batch
            time_t: Time step
            
        Returns:
            Mask tensor
        """
        batch_size, seq_len = batch["input_ids"].shape
        
        # Get masking fraction for this time step
        mask_fraction = self._get_mask_fraction(time_t)
        
        # Generate masks for each sample in the batch
        masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for i in range(batch_size):
            input_id = f"sample_{i}"
            mask = self.generate_mask(input_id, time_t, seq_len, mask_fraction)
            masks[i] = mask
        
        return masks
    
    def generate_mask(self, input_id: str, time_t: int, seq_len: int, mask_fraction: float) -> torch.Tensor:
        """
        Generate mask for specific input and time step.
        
        Args:
            input_id: Input identifier
            time_t: Time step
            seq_len: Sequence length
            mask_fraction: Fraction of tokens to mask
            
        Returns:
            Mask tensor
        """
        # Create cache key
        cache_key = (input_id, time_t, seq_len, mask_fraction)
        
        # Check cache first
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        # Generate new mask
        num_masked = int(seq_len * mask_fraction)
        mask_indices = torch.randperm(seq_len)[:num_masked]
        
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[mask_indices] = True
        
        # Cache the mask
        self.mask_cache[cache_key] = mask
        
        logger.debug(f"Generated mask for {input_id}, time {time_t}, fraction {mask_fraction:.2f}")
        return mask
    
    def _get_mask_fraction(self, time_t: int) -> float:
        """
        Get masking fraction for a time step.
        
        Args:
            time_t: Time step
            
        Returns:
            Masking fraction
        """
        # Find the closest time step in the configuration
        if time_t in self.mask_fractions:
            return self.mask_fractions[time_t]
        
        # Interpolate between closest time steps
        sorted_times = sorted(self.mask_fractions.keys())
        
        if time_t <= sorted_times[0]:
            return self.mask_fractions[sorted_times[0]]
        elif time_t >= sorted_times[-1]:
            return self.mask_fractions[sorted_times[-1]]
        
        # Find interpolation points
        for i in range(len(sorted_times) - 1):
            if sorted_times[i] <= time_t <= sorted_times[i + 1]:
                t1, t2 = sorted_times[i], sorted_times[i + 1]
                f1, f2 = self.mask_fractions[t1], self.mask_fractions[t2]
                
                # Linear interpolation
                alpha = (time_t - t1) / (t2 - t1)
                return f1 + alpha * (f2 - f1)
        
        # Fallback
        return 0.5
    
    def get_mask_for_activation(self, input_id: str, time_t: int) -> Optional[torch.Tensor]:
        """
        Get cached mask for a specific input and time step.
        
        Args:
            input_id: Input identifier
            time_t: Time step
            
        Returns:
            Cached mask tensor or None
        """
        # Search cache for matching mask
        for (cached_input_id, cached_time_t, seq_len, mask_fraction), mask in self.mask_cache.items():
            if cached_input_id == input_id and cached_time_t == time_t:
                return mask
        
        return None
    
    def clear_cache(self):
        """Clear the mask cache."""
        self.mask_cache.clear()
        logger.debug("Cleared mask cache")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the mask cache."""
        return {
            "num_cached_masks": len(self.mask_cache),
            "cache_memory_mb": sum(mask.numel() * mask.element_size() for mask in self.mask_cache.values()) / (1024 * 1024)
        }
