"""
Architecture Trainer for Block/Stack/Rack Training

This module provides training capabilities for the new hierarchical architecture:
- BlockTrainer: Train individual blocks
- StackTrainer: Train stacks of blocks
- RackTrainer: Train the complete rack (all stacks)

Supports the new naming convention:
- Block: Standard transformer block (attention + FFN + layer norm + residual)
- Stack: Collection of multiple blocks
- Rack: Final model containing multiple stacks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from ..model.architecture import Block, Stack, Rack
from ..config.base import StackWiseConfig

logger = logging.getLogger(__name__)


class BlockTrainer:
    """
    Trainer for individual Blocks.
    
    This trainer handles training individual transformer blocks,
    which is useful for layer-wise training scenarios.
    """
    
    def __init__(self, config: StackWiseConfig):
        """
        Initialize Block trainer.
        
        Args:
            config: StackWise configuration
        """
        self.config = config
        self.training_config = config.training
        
        # Training state
        self.current_block = 0
        self.trained_blocks = []
        self.activation_cache = {}
        
        logger.info("Initialized BlockTrainer for individual block training")
    
    def train_block(self, block: Block, block_idx: int, dataloader: DataLoader, 
                   input_activations: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Train a single block.
        
        Args:
            block: Block to train
            block_idx: Index of the block
            dataloader: Data loader for training
            input_activations: Cached activations from previous blocks (optional)
            
        Returns:
            Training statistics
        """
        logger.info(f"Training block {block_idx}")
        
        # Set up training
        block.train()
        from ..config.base import create_optimizer
        optimizer = create_optimizer(
            block.parameters(),
            self.training_config.optimizer
        )
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.training_config.epochs_per_stack):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Get input data
                if input_activations is not None:
                    # Use cached activations from previous blocks
                    x = input_activations
                else:
                    # Use fresh input data
                    if isinstance(batch, dict):
                        x = batch['input_ids']
                    else:
                        x = batch
                
                # Forward pass
                output = block(x)
                
                # Compute loss (placeholder - should be replaced with actual loss)
                loss = self._compute_block_loss(output, x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                total_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logger.info(f"Block {block_idx}, Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
        
        # Cache activations for next block
        self._cache_block_activations(block_idx, block, dataloader)
        
        # Update training state
        self.trained_blocks.append(block_idx)
        self.current_block = block_idx + 1
        
        # Return training statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {
            'block_idx': block_idx,
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'parameters': block.get_parameter_count(),
            'trainable_parameters': block.get_trainable_parameter_count()
        }
    
    def _compute_block_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for block training.
        
        This is a placeholder implementation. In practice, this should
        implement the actual loss function (e.g., cross-entropy for language modeling).
        """
        # Placeholder: MSE loss between output and target
        return torch.nn.functional.mse_loss(output, target)
    
    def _cache_block_activations(self, block_idx: int, block: Block, dataloader: DataLoader):
        """Cache activations from the trained block"""
        # This is a placeholder implementation
        # In practice, this should cache the activations for use by subsequent blocks
        logger.info(f"Cached activations for block {block_idx}")


class StackTrainer:
    """
    Trainer for Stacks of Blocks.
    
    This trainer handles training stacks of blocks together,
    which is useful for block-wise training scenarios.
    """
    
    def __init__(self, config: StackWiseConfig):
        """
        Initialize Stack trainer.
        
        Args:
            config: StackWise configuration
        """
        self.config = config
        self.training_config = config.training
        
        # Training state
        self.current_stack = 0
        self.trained_stacks = []
        self.activation_cache = {}
        
        logger.info("Initialized StackTrainer for stack training")
    
    def train_stack(self, stack: Stack, stack_idx: int, dataloader: DataLoader,
                   input_activations: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Train a single stack.
        
        Args:
            stack: Stack to train
            stack_idx: Index of the stack
            dataloader: Data loader for training
            input_activations: Cached activations from previous stacks (optional)
            
        Returns:
            Training statistics
        """
        logger.info(f"Training stack {stack_idx} with {stack.num_blocks} blocks")
        
        # Set up training
        stack.train()
        from ..config.base import create_optimizer
        optimizer = create_optimizer(
            stack.parameters(),
            self.training_config.optimizer
        )
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.training_config.epochs_per_stack):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Get input data
                if input_activations is not None:
                    x = input_activations
                else:
                    if isinstance(batch, dict):
                        x = batch['input_ids']
                    else:
                        x = batch
                
                # Forward pass through entire stack
                output = stack(x)
                
                # Compute loss
                loss = self._compute_stack_loss(output, x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                total_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logger.info(f"Stack {stack_idx}, Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
        
        # Cache activations for next stack
        self._cache_stack_activations(stack_idx, stack, dataloader)
        
        # Update training state
        self.trained_stacks.append(stack_idx)
        self.current_stack = stack_idx + 1
        
        # Return training statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {
            'stack_idx': stack_idx,
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'num_blocks': stack.num_blocks,
            'parameters': stack.get_parameter_count(),
            'trainable_parameters': stack.get_trainable_parameter_count()
        }
    
    def _compute_stack_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss for stack training"""
        # Placeholder: MSE loss between output and target
        return torch.nn.functional.mse_loss(output, target)
    
    def _cache_stack_activations(self, stack_idx: int, stack: Stack, dataloader: DataLoader):
        """Cache activations from the trained stack"""
        logger.info(f"Cached activations for stack {stack_idx}")


class RackTrainer:
    """
    Trainer for the complete Rack (all stacks).
    
    This trainer handles training the complete model,
    which is useful for end-to-end training scenarios.
    """
    
    def __init__(self, config: StackWiseConfig):
        """
        Initialize Rack trainer.
        
        Args:
            config: StackWise configuration
        """
        self.config = config
        self.training_config = config.training
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        logger.info("Initialized RackTrainer for complete model training")
    
    def train_rack(self, rack: Rack, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Train the complete rack.
        
        Args:
            rack: Rack to train
            dataloader: Data loader for training
            
        Returns:
            Training statistics
        """
        logger.info(f"Training complete rack with {len(rack.stacks)} stacks")
        
        # Set up training
        rack.train()
        from ..config.base import create_optimizer
        optimizer = create_optimizer(
            rack.parameters(),
            self.training_config.optimizer
        )
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.training_config.epochs_per_stack):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Get input data
                if isinstance(batch, dict):
                    input_ids = batch['input_ids']
                    labels = batch.get('labels', input_ids)
                else:
                    input_ids = batch
                    labels = batch
                
                # Forward pass through entire rack
                logits = rack(input_ids)
                
                # Compute loss
                loss = self._compute_rack_loss(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                total_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logger.info(f"Rack, Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_epoch_loss
            })
        
        # Update training state
        self.current_epoch = epoch + 1
        
        # Return training statistics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'num_stacks': len(rack.stacks),
            'parameters': rack.get_parameter_count(),
            'trainable_parameters': rack.get_trainable_parameter_count(),
            'training_history': self.training_history
        }
    
    def _compute_rack_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for rack training"""
        # Cross-entropy loss for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


class Trainer:
    """
    Unified trainer for the Block/Stack/Rack architecture.
    
    This trainer provides a single interface for training at different
    levels of the hierarchy based on configuration.
    """
    
    def __init__(self, config: StackWiseConfig):
        """
        Initialize trainer.
        
        Args:
            config: StackWise configuration
        """
        self.config = config
        self.training_config = config.training
        
        # Determine training architecture
        self.training_architecture = getattr(
            self.training_config, 
            'training_architecture', 
            'blockwise'
        )
        
        # Initialize appropriate trainer
        if self.training_architecture == 'blockwise':
            self.trainer = BlockTrainer(config)
        elif self.training_architecture == 'stackwise':
            self.trainer = StackTrainer(config)
        elif self.training_architecture == 'rackwise':
            self.trainer = RackTrainer(config)
        else:
            raise ValueError(f"Unknown training architecture: {self.training_architecture}")
        
        logger.info(f"Initialized Trainer with {self.training_architecture} training")
    
    def train_architecture(self, rack: Rack, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Train the architecture based on the configured training mode.
        
        Args:
            rack: Rack to train
            dataloader: Data loader for training
            
        Returns:
            Training statistics
        """
        if self.training_architecture == 'blockwise':
            return self._train_blockwise(rack, dataloader)
        elif self.training_architecture == 'stackwise':
            return self._train_stackwise(rack, dataloader)
        elif self.training_architecture == 'rackwise':
            return self._train_rackwise(rack, dataloader)
        else:
            raise ValueError(f"Unknown training architecture: {self.training_architecture}")
    
    def _train_blockwise(self, rack: Rack, dataloader: DataLoader) -> Dict[str, Any]:
        """Train block-wise (each block independently)"""
        logger.info("Starting block-wise training")
        
        results = []
        input_activations = None
        
        # Train each block in each stack
        for stack_idx, stack in enumerate(rack.stacks):
            logger.info(f"Training stack {stack_idx}")
            
            for block_idx, block in enumerate(stack.blocks):
                # Train individual block
                result = self.trainer.train_block(
                    block, 
                    block_idx, 
                    dataloader, 
                    input_activations
                )
                result['stack_idx'] = stack_idx
                results.append(result)
                
                # Update input activations for next block
                # (This is a placeholder - in practice, you'd cache actual activations)
                input_activations = torch.randn(2, 16, rack.d_model)
        
        total_blocks = sum(len(stack.blocks) for stack in rack.stacks)
        return {
            'training_architecture': 'blockwise',
            'results': results,
            'total_blocks': total_blocks,
            'n_stacks': len(rack.stacks),
            'blocks_per_stack': rack.stacks[0].num_blocks if rack.stacks else 0
        }
    
    def _train_stackwise(self, rack: Rack, dataloader: DataLoader) -> Dict[str, Any]:
        """Train stack-wise (each stack independently)"""
        logger.info("Starting stack-wise training")
        
        results = []
        input_activations = None
        
        # Train each stack
        for stack_idx, stack in enumerate(rack.stacks):
            # Train individual stack
            result = self.trainer.train_stack(
                stack,
                stack_idx,
                dataloader,
                input_activations
            )
            results.append(result)
            
            # Update input activations for next stack
            # (This is a placeholder - in practice, you'd cache actual activations)
            input_activations = torch.randn(2, 16, rack.d_model)
        
        return {
            'training_architecture': 'stackwise',
            'results': results,
            'total_stacks': len(rack.stacks),
            'blocks_per_stack': rack.stacks[0].num_blocks if rack.stacks else 0
        }
    
    def _train_rackwise(self, rack: Rack, dataloader: DataLoader) -> Dict[str, Any]:
        """Train rack-wise (entire model together)"""
        logger.info("Starting rack-wise training")
        
        # Train entire rack
        result = self.trainer.train_rack(rack, dataloader)
        result['training_architecture'] = 'rackwise'
        result['n_stacks'] = len(rack.stacks)
        result['blocks_per_stack'] = rack.stacks[0].num_blocks if rack.stacks else 0
        
        return result
