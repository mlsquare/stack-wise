"""
Block-based training logic for the unified trainer.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BlockTrainer:
    """
    Core block-based training logic.
    
    Handles the actual training of blocks with support for:
    - Time-step-based masking
    - Quantization and QLoRA adapters
    - Activation caching
    """
    
    def __init__(self, config, masking_strategy, quantization_manager, cache_manager, lexical_kernel_manager=None):
        """
        Initialize block trainer.
        
        Args:
            config: Training configuration
            masking_strategy: Masking strategy instance
            quantization_manager: Quantization manager instance
            cache_manager: Cache manager instance
            lexical_kernel_manager: Lexical kernel manager for LM head
        """
        self.config = config
        self.masking_strategy = masking_strategy
        self.quantization_manager = quantization_manager
        self.cache_manager = cache_manager
        self.lexical_kernel_manager = lexical_kernel_manager
        
        logger.info("Initialized BlockTrainer")
    
    def train_block(self, block_idx: int, dataloader: DataLoader, block_layers: List[torch.nn.Module]):
        """
        Train a single block.
        
        Args:
            block_idx: Index of the block to train
            dataloader: Data loader for training data
            block_layers: Layers in the current block
        """
        logger.info(f"Training block {block_idx} with {len(block_layers)} layers")
        
        # Setup optimizer for the block
        optimizer = self._setup_optimizer(block_layers)
        
        # Train for specified epochs
        for epoch in range(self.config.epochs_per_block):
            logger.info(f"Block {block_idx}, Epoch {epoch}/{self.config.epochs_per_block}")
            
            # Train with time-step-based masking if enabled
            if self.config.time_step_masking:
                self._train_with_all_time_steps(block_idx, dataloader, block_layers, optimizer)
            else:
                self._train_with_fixed_time_step(block_idx, dataloader, block_layers, optimizer)
        
        logger.info(f"Completed training block {block_idx}")
    
    def _setup_optimizer(self, block_layers: List[torch.nn.Module]) -> torch.optim.Optimizer:
        """
        Setup optimizer for the block.
        
        Args:
            block_layers: Layers in the current block
            
        Returns:
            Optimizer instance
        """
        # Collect only trainable parameters from all layers in the block
        parameters = []
        for layer in block_layers:
            parameters.extend([p for p in layer.parameters() if p.requires_grad])
        
        if not parameters:
            logger.warning("No trainable parameters found in block")
            # Return a dummy optimizer to avoid errors
            return torch.optim.AdamW([torch.tensor(0.0, requires_grad=True)], lr=self.config.learning_rate)
        
        # Create optimizer with parameter groups for potential different learning rates
        optimizer = torch.optim.AdamW(parameters, lr=self.config.learning_rate)
        
        logger.debug(f"Setup optimizer for {len(parameters)} trainable parameters")
        return optimizer
    
    def _get_block_time_step(self, block_idx: int) -> int:
        """
        Get the fixed time step for a block based on its depth/position.
        
        Args:
            block_idx: Block index
            
        Returns:
            Time step corresponding to this block's depth
        """
        # Map block index to time step
        # This could be configurable, but for now use a simple mapping
        if hasattr(self.config, 'time_step_bins') and self.config.time_step_bins:
            # Use the time step bins to map block to time step
            time_step_index = min(block_idx, len(self.config.time_step_bins) - 1)
            return self.config.time_step_bins[time_step_index]
        else:
            # Default: use block index as time step
            return block_idx
    
    def _train_with_all_time_steps(self, block_idx: int, dataloader: DataLoader, 
                              block_layers: List[torch.nn.Module], optimizer: torch.optim.Optimizer):
        """
        Train block with all time steps (all masking proportions).
        
        This method trains the block with every time step in the configuration,
        allowing the block to learn from all masking proportions.
        
        Args:
            block_idx: Index of the block
            dataloader: Data loader
            block_layers: Layers in the block
            optimizer: Optimizer instance
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Train with each time step
            for time_t in self.config.time_step_bins:
                # Generate masks for this time step
                masks = self.masking_strategy.generate_masks_for_time_step(batch, time_t)
                
                # Train with masks
                loss = self._train_batch_with_masks(batch, masks, block_layers, optimizer)
                total_loss += loss
                num_batches += 1
                
                # Cache activations if needed
                if not self.config.store_all_time_steps:
                    self._cache_current_time_step_activations(block_idx, time_t, batch, masks)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Block {block_idx} average loss: {avg_loss:.4f}")
    
    def _train_with_fixed_time_step(self, block_idx: int, dataloader: DataLoader,
                                 block_layers: List[torch.nn.Module], optimizer: torch.optim.Optimizer):
        """
        Train block at a fixed time step corresponding to its depth/position.
        
        Args:
            block_idx: Index of the block (determines the fixed time step)
            dataloader: Data loader
            block_layers: Layers in the block
            optimizer: Optimizer instance
        """
        # Determine the fixed time step for this block based on its depth
        fixed_time_step = self._get_block_time_step(block_idx)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Generate masks for the fixed time step corresponding to this block's depth
            masks = self.masking_strategy.generate_masks_for_time_step(batch, fixed_time_step)
            
            # Train with masks at the fixed time step
            loss = self._train_batch_with_masks(batch, masks, block_layers, optimizer)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Block {block_idx} (time step {fixed_time_step}) average loss: {avg_loss:.4f}")
    
    def _train_batch_with_masks(self, batch: Dict[str, torch.Tensor], masks: torch.Tensor,
                                block_layers: List[torch.nn.Module], optimizer: torch.optim.Optimizer) -> float:
        """
        Train a single batch with masks.
        
        Args:
            batch: Training batch
            masks: Mask tensor
            block_layers: Layers in the block
            optimizer: Optimizer instance
            
        Returns:
            Loss value
        """
        optimizer.zero_grad()
        
        # Forward pass through block layers
        hidden_states = batch["input_ids"]
        
        for layer in block_layers:
            hidden_states = layer(hidden_states)
        
        # Apply language model head (tied embeddings)
        if self.lexical_kernel_manager is not None:
            lm_head = self.lexical_kernel_manager.get_lm_head()
            logits = lm_head(hidden_states)
        else:
            # Fallback: use hidden states directly (placeholder)
            logits = hidden_states
        
        # Compute loss with proper logits
        loss = self._compute_loss(logits, batch["input_ids"], masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for the mask-diffusion objective.
        
        Only computes loss on masked positions for efficiency.
        Handles batches with varying mask patterns per sample.
        
        Args:
            logits: Model logits from language model head (batch_size, seq_len, vocab_size)
            targets: Target tokens (batch_size, seq_len)
            masks: Mask tensor (batch_size, seq_len) - boolean
            
        Returns:
            Loss tensor
        """
        # Flatten for easier indexing
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.view(-1)  # (batch_size * seq_len,)
        masks_flat = masks.view(-1)  # (batch_size * seq_len,)
        
        # Get indices where mask is True
        mask_indices = masks_flat.nonzero(as_tuple=False).squeeze(-1)  # (num_masked,)
        
        if len(mask_indices) == 0:
            # No masked positions, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Extract logits and targets only for masked positions
        masked_logits = logits_flat[mask_indices]  # (num_masked, vocab_size)
        masked_targets = targets_flat[mask_indices]  # (num_masked,)
        
        # Compute cross-entropy loss only on masked positions
        loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets.long())
        
        return loss
    
    def _cache_current_time_step_activations(self, block_idx: int, time_t: int, 
                                           batch: Dict[str, torch.Tensor], masks: torch.Tensor):
        """
        Cache activations for current time step.
        
        Args:
            block_idx: Block index
            time_t: Time step
            batch: Training batch
            masks: Mask tensor
        """
        # TODO: Implement time-step-specific activation caching
        # This should cache activations with time-step information for progressive training
        # Key considerations:
        # - Store activations with (block_idx, time_t, sample_id, mask_id) keys
        # - Handle memory-efficient storage for multiple time steps
        # - Support retrieval by time step for progressive training
        
        logger.warning(f"Time-step caching not implemented yet - block {block_idx}, time step {time_t}")
        logger.debug(f"Would cache activations for {len(batch['input_ids'])} samples with {masks.sum().item()} masked positions")
    
    def train_layer(self, layer_idx: int, dataloader: DataLoader, layer: torch.nn.Module):
        """
        Train a single layer (for layer-wise training with block_size=1).
        
        Args:
            layer_idx: Index of the layer
            dataloader: Data loader
            layer: Layer module
        """
        logger.info(f"Training layer {layer_idx}")
        
        # Setup optimizer for single layer
        optimizer = torch.optim.AdamW(layer.parameters(), lr=self.config.learning_rate)
        
        # Train layer
        self.train_block(layer_idx, dataloader, [layer])
        
        logger.info(f"Completed training layer {layer_idx}")
