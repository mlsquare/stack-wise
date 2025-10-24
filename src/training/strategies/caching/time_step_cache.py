"""
Time-step-aware activation caching for memory-efficient training.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
import pickle

logger = logging.getLogger(__name__)


class TimeStepCache:
    """
    Time-step-aware activation caching for memory-efficient training.
    
    Implements efficient caching that avoids storing all time-step activations
    by only caching the current time step and using discrete time bins.
    """
    
    def __init__(self, config):
        """
        Initialize time-step cache.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.store_all_time_steps = config.store_all_time_steps
        self.time_step_cache_size = config.time_step_cache_size
        self.cache_dir = Path(config.cache_dir)
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache storage
        self.activations = {}  # (block_idx, time_t) -> activations
        self.current_time_step = None
        
        logger.info(f"Initialized TimeStepCache: store_all={self.store_all_time_steps}, cache_size={self.time_step_cache_size}")
    
    def setup_cache(self):
        """Setup cache for training."""
        # Clear existing cache
        self.activations.clear()
        self.current_time_step = None
        
        logger.debug("Cache setup complete")
    
    def store_activation(self, block_idx: int, time_t: int, input_id: str, 
                       activation: torch.Tensor, mask: torch.Tensor):
        """
        Store activation for a specific block and time step.
        
        Args:
            block_idx: Block index
            time_t: Time step
            input_id: Input identifier
            activation: Activation tensor
            mask: Mask tensor
        """
        cache_key = (block_idx, time_t, input_id)
        
        # Store activation
        self.activations[cache_key] = {
            'activation': activation.detach().cpu(),
            'mask': mask.detach().cpu(),
            'block_idx': block_idx,
            'time_t': time_t,
            'input_id': input_id
        }
        
        # Update current time step
        self.current_time_step = time_t
        
        # Cleanup old time steps if not storing all
        if not self.store_all_time_steps:
            self._cleanup_old_time_steps(time_t)
        
        logger.debug(f"Stored activation for block {block_idx}, time {time_t}, input {input_id}")
    
    def get_activation(self, block_idx: int, time_t: int, input_id: str) -> Optional[Dict[str, Any]]:
        """
        Get activation for a specific block, time step, and input.
        
        Args:
            block_idx: Block index
            time_t: Time step
            input_id: Input identifier
            
        Returns:
            Activation data or None
        """
        cache_key = (block_idx, time_t, input_id)
        
        if cache_key in self.activations:
            return self.activations[cache_key]
        
        return None
    
    def get_activations_for_time_step(self, time_t: int) -> List[Dict[str, Any]]:
        """
        Get all activations for a specific time step.
        
        Args:
            time_t: Time step
            
        Returns:
            List of activation data
        """
        activations = []
        
        for (block_idx, cached_time_t, input_id), data in self.activations.items():
            if cached_time_t == time_t:
                activations.append(data)
        
        return activations
    
    def _cleanup_old_time_steps(self, current_time_t: int):
        """Cleanup activations from old time steps."""
        if self.time_step_cache_size <= 1:
            # Only keep current time step
            keys_to_remove = []
            for (block_idx, time_t, input_id) in self.activations.keys():
                if time_t != current_time_t:
                    keys_to_remove.append((block_idx, time_t, input_id))
            
            for key in keys_to_remove:
                del self.activations[key]
        else:
            # Keep last N time steps
            time_steps = set(time_t for (_, time_t, _) in self.activations.keys())
            if len(time_steps) > self.time_step_cache_size:
                # Remove oldest time steps
                sorted_time_steps = sorted(time_steps)
                time_steps_to_remove = sorted_time_steps[:-self.time_step_cache_size]
                
                keys_to_remove = []
                for (block_idx, time_t, input_id) in self.activations.keys():
                    if time_t in time_steps_to_remove:
                        keys_to_remove.append((block_idx, time_t, input_id))
                
                for key in keys_to_remove:
                    del self.activations[key]
    
    def cache_block_activations(self, block_idx: int, block_layers: List[torch.nn.Module]):
        """
        Cache activations for a block.
        
        Args:
            block_idx: Block index
            block_layers: Layers in the block
        """
        # This would implement block-level activation caching
        # For now, just log the action
        logger.debug(f"Cached activations for block {block_idx}")
    
    def cache_fusion_activations(self, block_idx: int, blocks: List[List[torch.nn.Module]]):
        """
        Cache activations for fusion training.
        
        Args:
            block_idx: Current block index
            blocks: All blocks to cache
        """
        # This would implement fusion-specific activation caching
        logger.debug(f"Cached fusion activations for block {block_idx}")
    
    def save_cache_to_disk(self, file_path: Optional[str] = None):
        """
        Save cache to disk.
        
        Args:
            file_path: Path to save cache (optional)
        """
        if file_path is None:
            file_path = self.cache_dir / "time_step_cache.pkl"
        
        cache_data = {
            'activations': self.activations,
            'current_time_step': self.current_time_step,
            'config': {
                'store_all_time_steps': self.store_all_time_steps,
                'time_step_cache_size': self.time_step_cache_size
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved cache to {file_path}")
    
    def load_cache_from_disk(self, file_path: Optional[str] = None):
        """
        Load cache from disk.
        
        Args:
            file_path: Path to load cache from (optional)
        """
        if file_path is None:
            file_path = self.cache_dir / "time_step_cache.pkl"
        
        if not Path(file_path).exists():
            logger.warning(f"Cache file not found: {file_path}")
            return
        
        with open(file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.activations = cache_data['activations']
        self.current_time_step = cache_data['current_time_step']
        
        logger.info(f"Loaded cache from {file_path}")
    
    def cleanup(self):
        """Cleanup cache and free memory."""
        self.activations.clear()
        self.current_time_step = None
        
        logger.debug("Cache cleanup complete")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            "num_activations": len(self.activations),
            "current_time_step": self.current_time_step,
            "store_all_time_steps": self.store_all_time_steps,
            "time_step_cache_size": self.time_step_cache_size,
            "cache_memory_mb": sum(
                data['activation'].numel() * data['activation'].element_size() 
                for data in self.activations.values()
            ) / (1024 * 1024)
        }
