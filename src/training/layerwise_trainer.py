"""
Layer-wise training system with mask-diffusion objective and hybrid caching.

This module implements:
- Progressive masking strategies with different schedulers
- Hybrid activation caching with unique mask storage
- Dual-mode caching (layerwise and fusion)
- Fixed mask assignment for consistent training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import hashlib
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class MaskScheduler:
    """Base class for mask fraction scheduling strategies"""
    
    def __init__(self, schedule_type: str = "linear"):
        """
        Initialize mask scheduler.
        
        Args:
            schedule_type: Type of scheduling ("linear", "exponential", "cosine")
        """
        self.schedule_type = schedule_type
    
    def get_mask_fraction(self, layer_idx: int, total_layers: int, 
                         min_fraction: float, max_fraction: float) -> float:
        """Calculate mask fraction using the specified scheduling strategy"""
        if self.schedule_type == "linear":
            return self._linear_schedule(layer_idx, total_layers, min_fraction, max_fraction)
        elif self.schedule_type == "exponential":
            return self._exponential_schedule(layer_idx, total_layers, min_fraction, max_fraction)
        elif self.schedule_type == "cosine":
            return self._cosine_schedule(layer_idx, total_layers, min_fraction, max_fraction)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def _linear_schedule(self, layer_idx: int, total_layers: int, 
                        min_fraction: float, max_fraction: float) -> float:
        """Linear progression from min to max fraction"""
        progress = layer_idx / (total_layers - 1) if total_layers > 1 else 0.0
        return min_fraction + progress * (max_fraction - min_fraction)
    
    def _exponential_schedule(self, layer_idx: int, total_layers: int, 
                             min_fraction: float, max_fraction: float) -> float:
        """Exponential progression from min to max fraction"""
        progress = layer_idx / (total_layers - 1) if total_layers > 1 else 0.0
        exp_progress = (math.exp(progress) - 1) / (math.e - 1)
        return min_fraction + exp_progress * (max_fraction - min_fraction)
    
    def _cosine_schedule(self, layer_idx: int, total_layers: int, 
                        min_fraction: float, max_fraction: float) -> float:
        """Cosine progression from min to max fraction"""
        progress = layer_idx / (total_layers - 1) if total_layers > 1 else 0.0
        cos_progress = (1 - math.cos(progress * math.pi)) / 2
        return min_fraction + cos_progress * (max_fraction - min_fraction)


class MaskDiffusionObjective:
    """
    Mask-diffusion objective with progressive masking strategies.
    
    Implements variable masking rates (15%-90%) over token positions,
    with progressive increase from encoder-like to diffusion-based decoder behavior.
    """
    
    def __init__(self, 
                 min_mask_fraction: float = 0.15,
                 max_mask_fraction: float = 0.90,
                 mask_scheduler: Optional[MaskScheduler] = None,
                 mask_token_id: int = 0):
        """
        Initialize mask-diffusion objective.
        
        Args:
            min_mask_fraction: Minimum masking fraction (early layers)
            max_mask_fraction: Maximum masking fraction (late layers)
            mask_scheduler: Mask scheduling strategy
            mask_token_id: ID for mask token
        """
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        self.mask_scheduler = mask_scheduler or MaskScheduler("linear")
        self.mask_token_id = mask_token_id
    
    def create_progressive_mask(self, 
                              token_ids: torch.Tensor, 
                              layer_idx: int, 
                              total_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create progressive mask for the given layer.
        
        Args:
            token_ids: Original token IDs of shape (batch_size, seq_len)
            layer_idx: Current layer index (0-based)
            total_layers: Total number of layers
            
        Returns:
            Tuple of (masked_ids, mask_positions) where:
            - masked_ids: Token IDs with masked positions replaced by mask_token_id
            - mask_positions: Boolean tensor indicating masked positions
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Calculate mask fraction for this layer
        mask_fraction = self.mask_scheduler.get_mask_fraction(
            layer_idx, total_layers, 
            self.min_mask_fraction, self.max_mask_fraction
        )
        
        # Create mask positions
        num_masks = int(seq_len * mask_fraction)
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            # Randomly select positions to mask
            mask_indices = torch.randperm(seq_len, device=device)[:num_masks]
            mask_positions[i, mask_indices] = True
        
        # Create masked input
        masked_ids = token_ids.clone()
        masked_ids[mask_positions] = self.mask_token_id
        
        return masked_ids, mask_positions


class FixedMaskAssigner:
    """Assigns fixed masks to samples for consistent training"""
    
    def __init__(self, layer_idx: int, total_layers: int, mask_diffusion: MaskDiffusionObjective):
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.mask_diffusion = mask_diffusion
        self.assigned_masks = {}  # Cache of assigned masks per sample
        self.sample_hashes = {}  # Cache of sample hashes for deduplication
    
    def get_fixed_mask(self, sample_id: str, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or assign fixed mask for this sample"""
        if sample_id not in self.assigned_masks:
            # Generate mask once and store it
            masked_ids, mask_positions = self.mask_diffusion.create_progressive_mask(
                token_ids, self.layer_idx, self.total_layers
            )
            self.assigned_masks[sample_id] = (masked_ids, mask_positions)
            
            # Store hash for this sample-mask combination
            self.sample_hashes[sample_id] = self._compute_sample_hash(sample_id, mask_positions)
        
        return self.assigned_masks[sample_id]
    
    def _compute_sample_hash(self, sample_id: str, mask_positions: torch.Tensor) -> str:
        """Compute hash for sample-mask combination"""
        # Create deterministic hash from sample_id and mask pattern
        mask_str = self._tensor_to_string(mask_positions)
        combined = f"{sample_id}_{mask_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _tensor_to_string(self, tensor: torch.Tensor) -> str:
        """Convert tensor to deterministic string representation"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor_str = str(tensor.detach().numpy().tolist())
        return hashlib.md5(tensor_str.encode()).hexdigest()[:16]
    
    def get_sample_hash(self, sample_id: str) -> Optional[str]:
        """Get hash for a sample if it exists"""
        return self.sample_hashes.get(sample_id)


class HashBasedActivationCache:
    """
    Hybrid activation cache with unique mask storage and activation IDs.
    
    Stores unique masks once and gives each activation a unique ID.
    Enables efficient deduplication and fast lookups.
    """
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 cache_mode: str = "layerwise",
                 fusion_evaluation: bool = False,
                 save_fused_checkpoints: bool = False):
        """
        Initialize hybrid activation cache.
        
        Args:
            cache_dir: Directory for cache storage
            cache_mode: Caching mode ("layerwise" or "fusion")
            fusion_evaluation: Whether to evaluate fused models
            save_fused_checkpoints: Whether to save fused model checkpoints
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_mode = cache_mode
        self.fusion_evaluation = fusion_evaluation
        self.save_fused_checkpoints = save_fused_checkpoints
        
        # Hybrid storage structure
        self.activations = {}           # activation_id -> activation_tensor
        self.mask_lookup = {}           # activation_id -> mask_id
        self.unique_masks = {}          # mask_positions -> mask_id (tensor as key!)
        self.next_activation_id = 0
        self.next_mask_id = 0
        
        # Fusion mode specific
        self.fused_models = {}  # Layer -> fused model
        self.fused_checkpoints = {}  # Layer -> checkpoint path
    
    def get_or_create_mask_id(self, mask_positions: torch.Tensor) -> int:
        """Get or create mask ID for the given mask positions"""
        # Direct dictionary lookup - O(1) instead of O(n) linear search!
        if mask_positions in self.unique_masks:
            return self.unique_masks[mask_positions]  # Found existing mask
        
        # Create new mask_id
        mask_id = self.next_mask_id
        self.unique_masks[mask_positions] = mask_id  # Use tensor as key directly
        self.next_mask_id += 1
        logger.debug(f"Created new mask_id: {mask_id}")
        return mask_id
    
    def save_activation(self, layer_idx: int, mask_positions: torch.Tensor, activation: torch.Tensor) -> str:
        """Save activation with unique ID and mask reference"""
        # Get or create mask_id
        mask_id = self.get_or_create_mask_id(mask_positions)
        
        # Create unique activation_id
        activation_id = f"L{layer_idx}_{self.next_activation_id}"
        self.next_activation_id += 1
        
        # Store activation and its mask reference
        self.activations[activation_id] = activation.detach().cpu()
        self.mask_lookup[activation_id] = mask_id
        
        logger.debug(f"Cached activation: {activation_id} with mask_id: {mask_id}")
        return activation_id
    
    def get_activation(self, activation_id: str) -> torch.Tensor:
        """Get activation by activation_id - guaranteed to exist by construction"""
        return self.activations[activation_id]
    
    def get_mask_for_activation(self, activation_id: str) -> torch.Tensor:
        """Get mask for a given activation_id - guaranteed to exist by construction"""
        mask_id = self.mask_lookup[activation_id]
        # Find the mask tensor for this mask_id
        for mask_tensor, stored_mask_id in self.unique_masks.items():
            if stored_mask_id == mask_id:
                return mask_tensor
        raise ValueError(f"Mask not found for activation_id: {activation_id}")
    
    def find_activation_by_mask(self, layer_idx: int, mask_positions: torch.Tensor) -> Optional[str]:
        """Find activation_id for a given mask - returns None if not found"""
        if mask_positions not in self.unique_masks:
            return None
        
        mask_id = self.unique_masks[mask_positions]
        
        # Look for existing activation with this mask
        for activation_id, stored_mask_id in self.mask_lookup.items():
            if activation_id.startswith(f"L{layer_idx}_") and stored_mask_id == mask_id:
                return activation_id
        return None
    
    def cache_after_training(self, layer_idx: int, activations: Dict[str, torch.Tensor], 
                           sample_ids: List[str], mask_patterns: Dict[str, torch.Tensor]):
        """Cache activations after layer training"""
        if self.cache_mode == "layerwise":
            self._cache_layer_activations(layer_idx, activations, sample_ids, mask_patterns)
        elif self.cache_mode == "fusion":
            self._cache_fused_activations(layer_idx, activations, sample_ids, mask_patterns)
    
    def _cache_layer_activations(self, layer_idx: int, activations: Dict[str, torch.Tensor], 
                                sample_ids: List[str], mask_patterns: Dict[str, torch.Tensor]):
        """Cache activations for layerwise mode"""
        for sample_id in sample_ids:
            if sample_id in activations and sample_id in mask_patterns:
                activation_id = self.save_activation(
                    layer_idx, mask_patterns[sample_id], activations[sample_id]
                )
                logger.debug(f"Cached activation for sample {sample_id}: {activation_id}")
    
    def _cache_fused_activations(self, layer_idx: int, activations: Dict[str, torch.Tensor], 
                                sample_ids: List[str], mask_patterns: Dict[str, torch.Tensor]):
        """Cache activations for fusion mode with optimized storage"""
        # Create or update fused model
        fused_model = self._create_optimized_fused_model(layer_idx)
        
        # Evaluate fused model and cache activations
        if self.fusion_evaluation:
            self._evaluate_optimized_fused_model(fused_model, layer_idx, sample_ids, mask_patterns)
        
        # Save fused checkpoint if requested
        if self.save_fused_checkpoints:
            self._save_fused_checkpoint(fused_model, layer_idx)
    
    def _create_optimized_fused_model(self, layer_idx: int):
        """Create optimized fused model for current layer (placeholder)"""
        logger.info(f"Creating fused model for layer {layer_idx}")
        return None
    
    def _evaluate_optimized_fused_model(self, fused_model, layer_idx: int, 
                                       sample_ids: List[str], mask_patterns: Dict[str, torch.Tensor]):
        """Evaluate optimized fused model and cache activations (placeholder)"""
        logger.info(f"Evaluating fused model for layer {layer_idx}")
    
    def _save_fused_checkpoint(self, fused_model, layer_idx: int):
        """Save fused model checkpoint (placeholder)"""
        checkpoint_path = self.cache_dir / f"fused_model_L{layer_idx}.pt"
        self.fused_checkpoints[layer_idx] = str(checkpoint_path)
        logger.info(f"Saved fused checkpoint: {checkpoint_path}")
    
    def _load_fused_checkpoint(self, layer_idx: int):
        """Load fused model checkpoint (placeholder)"""
        if layer_idx in self.fused_checkpoints:
            checkpoint_path = self.fused_checkpoints[layer_idx]
            logger.info(f"Loaded fused checkpoint: {checkpoint_path}")
        return None
    
    def clear_cache(self, layer_idx: Optional[int] = None):
        """Clear cache for specific layer or all layers"""
        if layer_idx is not None:
            # Clear specific layer cache
            keys_to_remove = [k for k in self.activations.keys() if k.startswith(f"L{layer_idx}_")]
            for key in keys_to_remove:
                if key in self.activations:
                    del self.activations[key]
                    del self.mask_lookup[key]
            logger.info(f"Cleared cache for layer {layer_idx}")
        else:
            # Clear all cache
            self.activations.clear()
            self.mask_lookup.clear()
            self.unique_masks.clear()
            self.next_activation_id = 0
            self.next_mask_id = 0
            
            # Clear fused model checkpoints
            for checkpoint_path in self.fused_checkpoints.values():
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
            self.fused_checkpoints.clear()
            logger.info("Cleared all cache")


class LayerwiseTrainer:
    """
    Layer-wise trainer with mask-diffusion objective and hash-based caching.
    
    Implements progressive masking, dual-mode caching, and optimized fusion.
    """
    
    def __init__(self, config):
        """
        Initialize layerwise trainer.
        
        Args:
            config: Configuration object with training parameters
        """
        self.config = config
        
        # Initialize mask diffusion objective
        mask_scheduler = MaskScheduler(config.training.mask_schedule_type)
        self.mask_diffusion = MaskDiffusionObjective(
            min_mask_fraction=config.training.min_mask_fraction,
            max_mask_fraction=config.training.max_mask_fraction,
            mask_scheduler=mask_scheduler,
            mask_token_id=config.training.mask_token_id
        )
        
        # Initialize activation cache
        self.activation_cache = HashBasedActivationCache(
            cache_dir=config.training.cache_dir,
            cache_mode=config.training.cache_mode,
            fusion_evaluation=config.training.fusion_evaluation,
            save_fused_checkpoints=config.training.save_fused_checkpoints
        )
        
        # Training state
        self.current_layer = 0
        self.trained_layers = []
    
    def train_layer(self, layer_idx: int, dataloader: DataLoader, model_layer: nn.Module):
        """
        Train a single layer with mask-diffusion objective.
        
        Args:
            layer_idx: Index of layer to train
            dataloader: Data loader for training data
            model_layer: The layer module to train
        """
        logger.info(f"Training layer {layer_idx}")
        
        # Create fixed mask assigner for this layer
        mask_assigner = FixedMaskAssigner(layer_idx, self.config.model.n_layers, self.mask_diffusion)
        
        # Training loop
        model_layer.train()
        optimizer = torch.optim.AdamW(model_layer.parameters(), lr=self.config.training.learning_rate)
        
        for epoch in range(self.config.training.epochs_per_layer):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                token_ids = batch["input_ids"]  # Shape: (batch_size, seq_len)
                sample_ids = batch.get("sample_ids", [f"sample_{i}" for i in range(len(token_ids))])
                
                # Process each sample in the batch individually
                batch_masked_ids = []
                batch_mask_positions = []
                
                for i, sample_id in enumerate(sample_ids):
                    # Get individual sample token_ids
                    sample_token_ids = token_ids[i:i+1]  # Shape: (1, seq_len)
                    
                    # Get fixed mask for this specific sample
                    masked_ids, mask_positions = mask_assigner.get_fixed_mask(sample_id, sample_token_ids)
                    
                    batch_masked_ids.append(masked_ids[0])  # Remove batch dimension
                    batch_mask_positions.append(mask_positions[0])  # Remove batch dimension
                
                # Stack back into batch format
                masked_ids = torch.stack(batch_masked_ids)  # Shape: (batch_size, seq_len)
                mask_positions = torch.stack(batch_mask_positions)  # Shape: (batch_size, seq_len)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model_layer(masked_ids)
                
                # Compute loss (placeholder - would need actual loss function)
                loss = self._compute_loss(outputs, token_ids, mask_positions)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Cache activations with hybrid deduplication
                activations = {sample_ids[i]: outputs[i] for i in range(len(sample_ids))}
                mask_patterns = {sample_ids[i]: mask_positions[i] for i in range(len(sample_ids))}
                
                # Cache all activations - each gets a unique ID by construction
                for i, sample_id in enumerate(sample_ids):
                    # Always create new activation with unique ID
                    activation_id = self.activation_cache.save_activation(
                        layer_idx, mask_positions[i], outputs[i]
                    )
                    logger.debug(f"Cached activation: {activation_id}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Layer {layer_idx}, Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Cache activations after training
        self.activation_cache.cache_after_training(
            layer_idx, activations, sample_ids, mask_patterns
        )
        
        # Save layer checkpoint
        self._save_layer_checkpoint(layer_idx, model_layer)
        
        # Clear previous layer cache to save memory
        if layer_idx > 0:
            self.activation_cache.clear_cache(layer_idx - 1)
        
        self.trained_layers.append(layer_idx)
        self.current_layer = layer_idx + 1
    
    def _compute_loss(self, outputs: torch.Tensor, token_ids: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
        """Compute loss for mask-diffusion objective (placeholder)"""
        # This would implement the actual loss computation
        # For now, return a dummy loss
        return torch.mean((outputs - token_ids) ** 2)
    
    def _save_layer_checkpoint(self, layer_idx: int, model_layer: nn.Module):
        """Save layer checkpoint"""
        checkpoint_path = self.activation_cache.cache_dir / f"layer_{layer_idx}.pt"
        torch.save(model_layer.state_dict(), checkpoint_path)
        logger.info(f"Saved layer {layer_idx} checkpoint: {checkpoint_path}")
    
    def train_all_layers(self, dataloader: DataLoader, model_layers: List[nn.Module]):
        """
        Train all layers sequentially.
        
        Args:
            dataloader: Data loader for training data
            model_layers: List of layer modules to train
        """
        for layer_idx, model_layer in enumerate(model_layers):
            self.train_layer(layer_idx, dataloader, model_layer)
        
        logger.info("Completed training all layers")
    
    def get_training_info(self) -> Dict:
        """Get training information"""
        return {
            "current_layer": self.current_layer,
            "trained_layers": self.trained_layers,
            "cache_mode": self.activation_cache.cache_mode,
            "cache_dir": str(self.activation_cache.cache_dir),
            "mask_schedule": self.mask_diffusion.mask_scheduler.schedule_type,
            "min_mask_fraction": self.mask_diffusion.min_mask_fraction,
            "max_mask_fraction": self.mask_diffusion.max_mask_fraction
        }
