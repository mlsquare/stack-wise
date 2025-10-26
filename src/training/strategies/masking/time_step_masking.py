"""
Time-step-based masking strategy for progressive training.

⚠️  WARNING: This module is currently BROKEN and UNUSED.
The masking classes have config attribute mismatches and are not functional.
The ProgressiveTrainer that depends on these classes is also broken.

Supports two time interpretations:
1. Time-as-input: Time is added as input parameter (standard diffusion)
2. Time-as-depth: Time is tied to stack index (progressive training)
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import math

# Issue deprecation warning
warnings.warn(
    "TimeStepMasking is currently broken and unused. "
    "The masking classes have config attribute mismatches and are not functional. "
    "The ProgressiveTrainer that depends on these classes is also broken.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)


class TimeStepMasking:
    """
    Enhanced time-step-based masking strategy for progressive training.
    
    ⚠️  WARNING: This class is currently BROKEN and UNUSED.
    The masking classes have config attribute mismatches and are not functional.
    The ProgressiveTrainer that depends on these classes is also broken.
    
    Supports two time interpretations:
    1. Time-as-input: Time is added as input parameter (standard diffusion)
    2. Time-as-depth: Time is tied to stack index (progressive training)
    
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
        
        # Time interpretation mode
        self.time_interpretation = getattr(config, 'time_interpretation', 'depth')  # 'input' or 'depth'
        self.current_time_step = 0
        self.max_stacks = getattr(config, 'max_stacks', 12)
        
        # Initialize mask storage
        self.mask_cache = {}
        
        # Initialize time components based on interpretation
        if self.time_interpretation == "input":
            self._init_time_as_input()
        elif self.time_interpretation == "depth":
            self._init_time_as_depth()
        else:
            raise ValueError(f"Unknown time interpretation: {self.time_interpretation}")
        
        logger.info(f"Initialized TimeStepMasking with {self.num_time_steps} time steps, "
                  f"interpretation: {self.time_interpretation}")
    
    def _init_time_as_input(self):
        """Initialize for time-as-input (standard diffusion)"""
        self.d_model = getattr(self.config, 'd_model', 512)
        self.time_embedding_dim = getattr(self.config, 'time_embedding_dim', self.d_model)
        self.time_encoding_type = getattr(self.config, 'time_encoding_type', 'sinusoidal')
        
        if self.time_encoding_type == 'learned':
            self.time_embedding = nn.Embedding(self.num_time_steps, self.time_embedding_dim)
        else:
            self.time_embedding = None  # Will use sinusoidal encoding
    
    def _init_time_as_depth(self):
        """Initialize for time-as-depth (progressive training)"""
        self.stack_time_mapping = self._create_stack_time_mapping()
        self.time_per_stack = getattr(self.config, 'time_per_stack', 100)
    
    def _create_stack_time_mapping(self) -> Dict[int, int]:
        """Create mapping from stack index to time step"""
        mapping = {}
        mapping_type = getattr(self.config, 'stack_time_mapping', 'linear')
        
        for stack_idx in range(self.max_stacks):
            if mapping_type == 'linear':
                time_t = min(stack_idx * self.time_per_stack, self.num_time_steps - 1)
            elif mapping_type == 'exponential':
                time_t = min(int(math.exp(stack_idx / 2) * self.time_per_stack), self.num_time_steps - 1)
            else:
                time_t = min(stack_idx, self.num_time_steps - 1)
            
            mapping[stack_idx] = time_t
        
        return mapping
    
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
    
    def generate_masks_for_stack(self, batch: Dict[str, torch.Tensor], stack_idx: int) -> torch.Tensor:
        """
        Generate masks for a specific stack (time-as-depth interpretation).
        
        Args:
            batch: Training batch
            stack_idx: Stack index
            
        Returns:
            Mask tensor
        """
        if self.time_interpretation != "depth":
            raise ValueError("generate_masks_for_stack only available for time-as-depth interpretation")
        
        # Map stack index to time step
        time_t = self._stack_to_time(stack_idx)
        return self.generate_masks_for_time_step(batch, time_t)
    
    def add_time_to_inputs(self, inputs: torch.Tensor, time_t: int) -> torch.Tensor:
        """
        Add time as positional encoding to inputs (time-as-input interpretation).
        
        Args:
            inputs: Input tensor (batch_size, seq_len, d_model)
            time_t: Time step
            
        Returns:
            Inputs with time encoding added
        """
        if self.time_interpretation != "input":
            raise ValueError("add_time_to_inputs only available for time-as-input interpretation")
        
        batch_size, seq_len, d_model = inputs.shape
        
        if self.time_encoding_type == 'learned':
            # Use learned time embedding
            time_embedding = self.time_embedding(torch.tensor([time_t] * batch_size, device=inputs.device))
            time_embedding = time_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # Use sinusoidal time encoding
            time_embedding = self._sinusoidal_time_encoding(time_t, batch_size, seq_len, d_model, inputs.device)
        
        # Add time to inputs
        inputs_with_time = inputs + time_embedding
        
        return inputs_with_time
    
    def _sinusoidal_time_encoding(self, time_t: int, batch_size: int, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal time encoding"""
        # Normalize time to [0, 1] range
        time_norm = time_t / (self.num_time_steps - 1)
        
        # Create sinusoidal encoding
        position = torch.arange(seq_len, device=device).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Add time component
        time_component = time_norm * torch.ones_like(position)
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin((position + time_component) * div_term)
        pe[:, 1::2] = torch.cos((position + time_component) * div_term)
        
        # Expand to batch size
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe
    
    def _stack_to_time(self, stack_idx: int) -> int:
        """Map stack index to time step"""
        if self.time_interpretation == "depth":
            return self.stack_time_mapping.get(stack_idx, min(stack_idx, self.num_time_steps - 1))
        else:
            return self.current_time_step
    
    def set_current_time_step(self, time_t: int):
        """Set current time step (for time-as-input interpretation)"""
        if self.time_interpretation == "input":
            self.current_time_step = time_t
        else:
            logger.warning("set_current_time_step only meaningful for time-as-input interpretation")
    
    def get_time_step_for_stack(self, stack_idx: int) -> int:
        """Get time step for a specific stack"""
        return self._stack_to_time(stack_idx)
    
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
