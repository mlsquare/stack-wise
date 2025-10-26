"""
Progressive DataLoader for enhanced training functionality.

This module provides a progressive dataloader that handles:
- Time interpretation (time-as-input vs time-as-depth)
- Trunk activation caching and injection
- Enhanced batch preparation with masking
- Integration with existing masking strategies
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Iterator
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import hashlib

from .strategies.masking.time_step_masking import TimeStepMasking
from .strategies.masking.progressive_masking import ProgressiveMasking

logger = logging.getLogger(__name__)


class ProgressiveDataLoader:
    """
    Enhanced DataLoader for progressive training.
    
    Handles:
    - Time interpretation (time-as-input vs time-as-depth)
    - Trunk activation caching and injection
    - Enhanced batch preparation with masking
    - Integration with existing masking strategies
    """
    
    def __init__(self, 
                 base_dataloader: DataLoader,
                 masking_strategy: Union[TimeStepMasking, ProgressiveMasking],
                 stack_idx: int = 0,
                 trunk_activations: Optional[Dict] = None,
                 cache_activations: bool = True):
        """
        Initialize progressive dataloader.
        
        Args:
            base_dataloader: Base dataloader to wrap
            masking_strategy: Masking strategy (TimeStepMasking or ProgressiveMasking)
            stack_idx: Current stack index (for time-as-depth interpretation)
            trunk_activations: Cached activations from previous stacks
            cache_activations: Whether to cache activations for future use
        """
        self.base_dataloader = base_dataloader
        self.masking_strategy = masking_strategy
        self.stack_idx = stack_idx
        self.trunk_activations = trunk_activations or {}
        self.cache_activations = cache_activations
        
        # Cache for storing activations
        self.activation_cache = {}
        
        # Store current batch time for consistency
        self.current_batch_time = None
        
        # Training objective configuration
        if masking_strategy is not None:
            self.training_objective = getattr(masking_strategy.config, 'training_objective', 'mlm')
        else:
            self.training_objective = 'mlm'  # Default training objective
        
        logger.info(f"Initialized ProgressiveDataLoader for stack {stack_idx}, objective: {self.training_objective}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over enhanced batches"""
        for batch in self.base_dataloader:
            enhanced_batch = self._enhance_batch(batch)
            yield enhanced_batch
    
    def _enhance_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Enhance batch with time information, masking, and trunk activations.
        
        Args:
            batch: Original batch
            
        Returns:
            Enhanced batch with additional information
        """
        # 1. Generate masks based on time interpretation
        masks = self._generate_masks(batch)
        
        # 2. Prepare inputs and targets
        inputs, targets = self._prepare_inputs_targets(batch)
        
        # 3. Add time information if time-as-input
        if hasattr(self.masking_strategy, 'time_interpretation') and \
           self.masking_strategy.time_interpretation == "input":
            inputs = self._add_time_to_inputs(inputs)
        
        # 4. Handle trunk activations
        if self.trunk_activations:
            inputs = self._inject_trunk_activations(inputs, batch)
        
        # 5. Cache activations if enabled
        if self.cache_activations:
            self._cache_activations(inputs, masks, batch)
        
        # 6. Create enhanced batch
        enhanced_batch = {
            'input_ids': inputs,
            'targets': targets,
            'masks': masks,
            'masking_ids': self._get_masking_ids(masks),
            'time_step': self._get_time_step(),
            'stack_idx': self.stack_idx,
            'trunk_activations': self.trunk_activations,
            'training_objective': self.training_objective,
            'original_batch': batch  # Keep original for reference
        }
        
        # 7. Add CLM-specific mask if needed
        if self.training_objective == 'clm':
            clm_mask = self._create_clm_mask(inputs.shape[1], inputs.device)
            enhanced_batch['clm_mask'] = clm_mask
            # Combine CLM mask with existing masks
            enhanced_batch['combined_masks'] = masks & clm_mask.unsqueeze(0)
        
        return enhanced_batch
    
    def _generate_masks(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate masks based on time interpretation"""
        if isinstance(self.masking_strategy, TimeStepMasking):
            if hasattr(self.masking_strategy, 'time_interpretation'):
                if self.masking_strategy.time_interpretation == "depth":
                    return self.masking_strategy.generate_masks_for_stack(batch, self.stack_idx)
                else:
                    # For time-as-input, sample random time for this batch
                    time_t = self._sample_random_time(batch)
                    return self.masking_strategy.generate_masks_for_time_step(batch, time_t)
            else:
                # Fallback to time step masking
                return self.masking_strategy.generate_masks_for_time_step(
                    batch, self.masking_strategy.current_time_step
                )
        elif isinstance(self.masking_strategy, ProgressiveMasking):
            return self.masking_strategy.generate_masks(batch, self.stack_idx)
        elif self.masking_strategy is None:
            # No masking - return all True masks
            batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else batch['inputs'].shape[0]
            seq_len = batch['input_ids'].shape[1] if 'input_ids' in batch else batch['inputs'].shape[1]
            return torch.ones(batch_size, seq_len, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown masking strategy: {type(self.masking_strategy)}")
    
    def _prepare_inputs_targets(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs and targets from batch based on training objective"""
        if isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs'))
            
            # Handle targets based on training objective
            if self.training_objective == 'mlm':
                # For MLM: targets are the same as inputs (will be masked)
                targets = inputs
            elif self.training_objective == 'clm':
                # For CLM: targets are shifted inputs (next token prediction)
                targets = self._prepare_clm_targets(inputs)
            else:
                # For other objectives: check if targets are explicitly provided
                if 'targets' in batch:
                    targets = batch['targets']
                elif 'labels' in batch:
                    targets = batch['labels']
                else:
                    # Fallback to inputs
                    targets = inputs
        else:
            # Assume batch is just input_ids
            inputs = batch
            if self.training_objective == 'clm':
                targets = self._prepare_clm_targets(inputs)
            else:
                targets = inputs
        
        return inputs, targets
    
    def _prepare_clm_targets(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Prepare targets for CLM (Causal Language Modeling) objective
        
        WARNING: For CLM, the last target token may not be available since we're 
        shifting the sequence by 1 position. The last position in targets will be 
        the same as the last position in inputs, which may not be the correct 
        target for training.
        
        Args:
            inputs: Input tensor (batch_size, seq_len)
            
        Returns:
            targets: Target tensor (batch_size, seq_len) with shifted inputs
        """
        # For CLM, targets are the same as inputs but shifted by 1 position
        # targets[i] = inputs[i+1] for i < seq_len-1
        # WARNING: targets[-1] = inputs[-1] - last target may not be available!
        targets = inputs.clone()
        targets[:, :-1] = inputs[:, 1:]  # Shift left by 1
        
        # Note: targets[:, -1] remains the same as inputs[:, -1]
        # This means the last position doesn't have a proper target
        # Consider using a special token or masking the last position in loss computation
        
        logger.warning("CLM targets prepared: last target token may not be available. "
                      "Consider masking the last position in loss computation.")
        
        return targets
    
    def _create_clm_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a mask for CLM that excludes the last position.
        
        For CLM, we typically don't compute loss on the last position since
        there's no target available (we shifted by 1 position).
        
        Args:
            seq_len: Sequence length
            device: Device for the mask tensor
            
        Returns:
            clm_mask: Boolean mask (seq_len,) - True for positions to include in loss
        """
        # Create mask that excludes the last position
        clm_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        clm_mask[-1] = False  # Exclude last position
        
        return clm_mask
    
    def _add_time_to_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add time information to inputs (time-as-input interpretation)"""
        if hasattr(self.masking_strategy, 'add_time_to_inputs'):
            # Use the same random time that was sampled for this batch
            time_t = self.current_batch_time
            return self.masking_strategy.add_time_to_inputs(inputs, time_t)
        return inputs
    
    def _inject_trunk_activations(self, inputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inject cached trunk activations into inputs"""
        if not self.trunk_activations:
            return inputs
        
        # Get batch identifier for caching
        batch_id = self._get_batch_id(batch)
        
        if batch_id in self.trunk_activations:
            trunk_acts = self.trunk_activations[batch_id]
            
            # For frozen trunk, use cached activations directly as inputs
            if isinstance(trunk_acts, torch.Tensor):
                # Use trunk activations as the input to the new stack
                # This avoids recomputing frozen trunk layers
                return trunk_acts
            elif isinstance(trunk_acts, dict) and 'activations' in trunk_acts:
                # Handle structured trunk activations
                return trunk_acts['activations']
            else:
                logger.warning(f"Unexpected trunk activation format: {type(trunk_acts)}")
                return inputs
        else:
            # No cached activations available, use original inputs
            logger.debug(f"No cached activations found for batch {batch_id}")
            return inputs
    
    def _cache_activations(self, inputs: torch.Tensor, masks: torch.Tensor, batch: Dict[str, torch.Tensor]):
        """Cache activations for future use"""
        batch_id = self._get_batch_id(batch)
        
        # Cache activations for both frozen and QLoRA trunk scenarios
        self.activation_cache[batch_id] = {
            'activations': inputs.detach().cpu(),  # Store as 'activations' for consistency
            'inputs': inputs.detach().cpu(),      # Keep original inputs for reference
            'masks': masks.detach().cpu(),
            'stack_idx': self.stack_idx,
            'time_step': self._get_time_step(),
            'training_objective': self.training_objective
        }
        
        logger.debug(f"Cached activations for batch {batch_id}, stack {self.stack_idx}")
    
    def _get_masking_ids(self, masks: torch.Tensor) -> torch.Tensor:
        """Get indices of masked positions for loss computation"""
        return torch.nonzero(masks, as_tuple=False).squeeze(-1)
    
    def _sample_random_time(self, batch: Dict[str, torch.Tensor]) -> int:
        """Sample random time for time-as-input interpretation"""
        if hasattr(self.masking_strategy, 'time_interpretation') and \
           self.masking_strategy.time_interpretation == "input":
            # Sample random time from uniform [0, 1]
            time_value = torch.rand(1, device=batch["input_ids"].device).item()  # [0, 1]
            # Convert to time step (0 to num_time_steps-1)
            time_t = int(time_value * (self.masking_strategy.num_time_steps - 1))
            # Store for consistency across batch processing
            self.current_batch_time = time_t
            return time_t
        else:
            return self.stack_idx
    
    def _get_time_step(self) -> int:
        """Get current time step"""
        if hasattr(self.masking_strategy, 'get_time_step_for_stack'):
            return self.masking_strategy.get_time_step_for_stack(self.stack_idx)
        elif hasattr(self.masking_strategy, 'current_time_step'):
            return self.masking_strategy.current_time_step
        elif hasattr(self.masking_strategy, 'time_interpretation') and \
             self.masking_strategy.time_interpretation == "input":
            # For time-as-input, return the current batch time
            return self.current_batch_time if self.current_batch_time is not None else 0
        else:
            return self.stack_idx
    
    def _get_batch_id(self, batch: Dict[str, torch.Tensor]) -> str:
        """Generate unique batch identifier for caching"""
        if isinstance(batch, dict) and 'input_ids' in batch:
            # Use stable SHA1 digest for deterministic identification
            input_bytes = batch['input_ids'].cpu().numpy().tobytes()
            digest = hashlib.sha1(input_bytes).hexdigest()
            return f"batch_{digest}"
        else:
            # Fallback to simple counter
            return f"batch_{id(batch)}"
    
    def get_cached_activations(self) -> Dict[str, torch.Tensor]:
        """Get cached activations"""
        return self.activation_cache
    
    def clear_cache(self):
        """Clear activation cache"""
        self.activation_cache.clear()
        logger.debug("Cleared activation cache")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the cache"""
        return {
            "num_cached_batches": len(self.activation_cache),
            "cache_memory_mb": sum(
                act['inputs'].numel() * act['inputs'].element_size() 
                for act in self.activation_cache.values()
            ) / (1024 * 1024)
        }


class CachedDataLoader(DataLoader):
    """
    DataLoader that works with cached activations.
    
    This is a specialized DataLoader that can inject cached activations
    into batches for progressive training.
    """
    
    def __init__(self, 
                 dataset: Dataset,
                 cached_activations: Dict[str, torch.Tensor],
                 batch_size: int = 1,
                 shuffle: bool = False,
                 **kwargs):
        """
        Initialize cached dataloader.
        
        Args:
            dataset: Base dataset
            cached_activations: Cached activations to inject
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.cached_activations = cached_activations
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches with cached activations"""
        for batch in super().__iter__():
            # Inject cached activations if available
            batch_id = self._get_batch_id(batch)
            if batch_id in self.cached_activations:
                cached_act = self.cached_activations[batch_id]
                batch['cached_activations'] = cached_act
            
            yield batch
    
    def _get_batch_id(self, batch: Dict[str, torch.Tensor]) -> str:
        """Generate batch identifier"""
        if isinstance(batch, dict) and 'input_ids' in batch:
            input_bytes = batch['input_ids'].cpu().numpy().tobytes()
            digest = hashlib.sha1(input_bytes).hexdigest()
            return f"batch_{digest}"
        else:
            return f"batch_{id(batch)}"
