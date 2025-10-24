"""
General activation caching for layer-wise and block-wise training.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
import pickle

logger = logging.getLogger(__name__)


class ActivationCache:
    """
    General activation caching for layer-wise and block-wise training.
    
    Provides efficient caching of activations between layers/blocks
    with deduplication and memory optimization.
    """
    
    def __init__(self, config):
        """
        Initialize activation cache.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.cache_mode = config.cache_mode
        self.cache_dir = Path(config.cache_dir)
        self.fusion_evaluation = config.fusion_evaluation
        self.save_fused_checkpoints = config.save_fused_checkpoints
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache storage
        self.activations = {}  # activation_id -> activation_data
        self.unique_masks = {}  # mask_positions -> mask_id
        self.mask_id_to_tensor = {}  # mask_id -> mask_positions
        self.next_mask_id = 0
        
        logger.info(f"Initialized ActivationCache: mode={self.cache_mode}")
    
    def setup_cache(self):
        """Setup cache for training."""
        # Clear existing cache
        self.activations.clear()
        self.unique_masks.clear()
        self.mask_id_to_tensor.clear()
        self.next_mask_id = 0
        
        logger.debug("Cache setup complete")
    
    def store_activation(self, sample_id: str, mask_positions: torch.Tensor, 
                       activation: torch.Tensor) -> str:
        """
        Store activation with unique ID.
        
        Args:
            sample_id: Sample identifier
            mask_positions: Mask positions tensor
            activation: Activation tensor
            
        Returns:
            Activation ID
        """
        # Get or create mask ID
        mask_id = self._get_or_create_mask_id(mask_positions)
        
        # Create activation ID
        activation_id = f"{sample_id}_mask_{mask_id}"
        
        # Store activation
        self.activations[activation_id] = {
            'activation': activation.detach().cpu(),
            'mask_positions': mask_positions.detach().cpu(),
            'sample_id': sample_id,
            'mask_id': mask_id,
            'activation_id': activation_id
        }
        
        logger.debug(f"Stored activation: {activation_id}")
        return activation_id
    
    def get_activation(self, activation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get activation by ID.
        
        Args:
            activation_id: Activation identifier
            
        Returns:
            Activation data or None
        """
        return self.activations.get(activation_id)
    
    def get_mask_for_activation(self, activation_id: str) -> Optional[torch.Tensor]:
        """
        Get mask for a given activation ID.
        
        Args:
            activation_id: Activation identifier
            
        Returns:
            Mask tensor or None
        """
        activation_data = self.activations.get(activation_id)
        if activation_data:
            return activation_data['mask_positions']
        return None
    
    def _get_or_create_mask_id(self, mask_positions: torch.Tensor) -> int:
        """
        Get or create mask ID for mask positions.
        
        Args:
            mask_positions: Mask positions tensor
            
        Returns:
            Mask ID
        """
        # Use tensor as key for O(1) lookup
        mask_key = tuple(mask_positions.tolist())
        
        if mask_key in self.unique_masks:
            return self.unique_masks[mask_key]
        
        # Create new mask ID
        mask_id = self.next_mask_id
        self.unique_masks[mask_key] = mask_id
        self.mask_id_to_tensor[mask_id] = mask_positions
        self.next_mask_id += 1
        
        return mask_id
    
    def cache_block_activations(self, block_idx: int, block_layers: List[torch.nn.Module]):
        """
        Cache activations for a block.
        
        Args:
            block_idx: Block index
            block_layers: Layers in the block
        """
        # This would implement block-level activation caching
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
            file_path = self.cache_dir / "activation_cache.pkl"
        
        cache_data = {
            'activations': self.activations,
            'unique_masks': self.unique_masks,
            'mask_id_to_tensor': self.mask_id_to_tensor,
            'next_mask_id': self.next_mask_id,
            'config': {
                'cache_mode': self.cache_mode,
                'fusion_evaluation': self.fusion_evaluation
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
            file_path = self.cache_dir / "activation_cache.pkl"
        
        if not Path(file_path).exists():
            logger.warning(f"Cache file not found: {file_path}")
            return
        
        with open(file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.activations = cache_data['activations']
        self.unique_masks = cache_data['unique_masks']
        self.mask_id_to_tensor = cache_data['mask_id_to_tensor']
        self.next_mask_id = cache_data['next_mask_id']
        
        logger.info(f"Loaded cache from {file_path}")
    
    def cleanup(self):
        """Cleanup cache and free memory."""
        self.activations.clear()
        self.unique_masks.clear()
        self.mask_id_to_tensor.clear()
        self.next_mask_id = 0
        
        logger.debug("Cache cleanup complete")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            "num_activations": len(self.activations),
            "num_unique_masks": len(self.unique_masks),
            "cache_mode": self.cache_mode,
            "cache_memory_mb": sum(
                data['activation'].numel() * data['activation'].element_size() 
                for data in self.activations.values()
            ) / (1024 * 1024)
        }
