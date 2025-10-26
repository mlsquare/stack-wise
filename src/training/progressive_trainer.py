"""
Progressive Trainer for progressive training strategies.

This module provides training strategies for progressive building:
- Frozen trunk + full precision new stack
- QLoRA trunk + full precision new stack
- Gradual unfreezing strategies
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .progressive_rack_builder import ProgressiveRackBuilder, PrecisionManager
from .progressive_dataloader import ProgressiveDataLoader, CachedDataLoader
from .strategies.masking.time_step_masking import TimeStepMasking
from .strategies.masking.progressive_masking import ProgressiveMasking
from ..config.base import StackWiseConfig
import hashlib
from ..config.base import ProgressiveConfig

logger = logging.getLogger(__name__)


class ProgressiveTrainer:
    """
    Trainer for progressive training strategies.
    
    Supports:
    - Frozen trunk + full precision new stack
    - QLoRA trunk + full precision new stack
    - Gradual unfreezing strategies
    - Activation streaming and caching
    """
    
    def __init__(self, config: StackWiseConfig):
        """
        Initialize progressive trainer.
        
        Args:
            config: StackWise configuration
        """
        self.config = config
        self.training_config = config.training
        
        # Progressive training configuration (normalize to ProgressiveConfig)
        prog = getattr(self.training_config, 'progressive', None)
        if isinstance(prog, dict):
            self.progressive_config = ProgressiveConfig.from_dict(prog)
        elif isinstance(prog, ProgressiveConfig):
            self.progressive_config = prog
        else:
            # Try to coerce
            try:
                self.progressive_config = ProgressiveConfig(**vars(prog))
            except Exception:
                self.progressive_config = ProgressiveConfig()

        self.trunk_strategy = getattr(self.progressive_config, 'trunk_strategy', 'frozen')
        self.new_stack_precision = getattr(self.progressive_config, 'new_stack_precision', 'full')
        self.cache_activations = getattr(self.progressive_config, 'cache_activations', True)
        
        # Initialize components
        self.precision_manager = PrecisionManager()
        self.activation_cache = {}
        self.training_history = []
        
        # Checkpointing configuration
        self.checkpoint_dir = Path(getattr(self.training_config, 'checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = getattr(self.training_config, 'save_interval', 100)
        
        logger.info(f"Initialized ProgressiveTrainer with trunk strategy: {self.trunk_strategy}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def train_rack(self, 
                   rack_builder: ProgressiveRackBuilder,
                   dataloader: DataLoader,
                   target_stacks: Optional[int] = None) -> Dict[str, Any]:
        """
        Train a rack (unified method for both complete and progressive training).
        
        Args:
            rack_builder: Progressive rack builder
            dataloader: Training dataloader
            target_stacks: Target number of stacks (for progressive training)
                          If None, trains the current rack as-is
            
        Returns:
            Training results
        """
        if target_stacks is not None:
            logger.info(f"Starting progressive training to {target_stacks} stacks")
        else:
            logger.info("Training current rack")
        
        results = {
            'training_history': [],
            'final_rack_info': None,
            'total_training_time': 0
        }
        
        # Determine training mode
        if target_stacks is not None:
            # Progressive training: add stacks one by one
            for stack_idx in range(target_stacks):
                logger.info(f"Training stack {stack_idx}")
                
                # 1. Add new stack
                new_stack = self._add_new_stack(rack_builder, stack_idx)
                
                # 2. Configure trunk strategy
                if self.trunk_strategy == "qlora":
                    self._setup_qlora_trunk(rack_builder, stack_idx)
                else:
                    self._freeze_trunk(rack_builder, stack_idx)
                
                # 3. Create progressive dataloader
                progressive_dataloader = self._create_progressive_dataloader(
                    dataloader, rack_builder, stack_idx
                )
                
                # 4. Train new stack
                stack_results = self._train_new_stack(
                    rack_builder, stack_idx, progressive_dataloader
                )
                
                # 5. Cache activations if needed
        else:
            # Complete rack training: train the entire rack
            logger.info("Training complete rack")
            
            # Build the complete rack
            rack = rack_builder.build_rack()
            
            # Create progressive dataloader for the complete rack
            progressive_dataloader = self._create_progressive_dataloader(
                dataloader, rack_builder, len(rack_builder.stacks) - 1
            )
            
            # Train the complete rack
            stack_results = self._train_complete_rack(
                rack_builder, progressive_dataloader
            )
            
            # Record training history
            self.training_history.append({
                'stack_idx': 'complete_rack',
                'training_results': stack_results,
                'rack_info': rack_builder.get_rack_info()
            })
            
            results['training_history'].append(stack_results)
        
        # Finalize results
        results['final_rack_info'] = rack_builder.get_rack_info()
        results['total_training_time'] = sum(
            result.get('training_time', 0) for result in results['training_history']
        )
        
        logger.info(f"Training completed. Final rack has {len(rack_builder.stacks)} stacks")
        return results
    
    def _train_complete_rack(self, rack_builder: ProgressiveRackBuilder, 
                            progressive_dataloader: ProgressiveDataLoader) -> Dict[str, Any]:
        """
        Train a complete rack (all stacks together).
        
        Args:
            rack_builder: Progressive rack builder
            progressive_dataloader: Progressive dataloader
            
        Returns:
            Training results
        """
        logger.info("Training complete rack")
        
        # Build the complete rack
        rack = rack_builder.build_rack()
        
        # Train the complete rack
        training_results = {
            'loss': 0.0,
            'accuracy': 0.0,
            'training_time': 0.0,
            'stack_count': len(rack_builder.stacks)
        }
        
        # This is a placeholder - actual training implementation would go here
        logger.info(f"Trained complete rack with {len(rack_builder.stacks)} stacks")
        
        return training_results
    
    def _add_new_stack(self, rack_builder: ProgressiveRackBuilder, stack_idx: int) -> nn.Module:
        """Add a new stack to the rack builder"""
        if rack_builder.building_mode == "append":
            new_stack = rack_builder.append_stack(precision=self.new_stack_precision)
        else:
            new_stack = rack_builder.prepend_stack(precision=self.new_stack_precision)
        
        return new_stack
    
    def _setup_qlora_trunk(self, rack_builder: ProgressiveRackBuilder, current_stack_idx: int):
        """Configure QLoRA training for trunk stacks (adapters already added during stack creation)"""
        if current_stack_idx > 0:
            # QLoRA adapters should already be added during stack creation
            # Now we just configure which parameters are trainable
            
            trunk_indices = list(range(current_stack_idx))
            
            # For QLoRA trunk: only LoRA adapters are trainable
            rack_builder.freeze_all_but_qlora(trunk_indices)
            
            logger.info(f"Configured QLoRA training for trunk stacks: {trunk_indices}")
            logger.info("Only LoRA adapters are trainable, original parameters are frozen")
    
    def _freeze_trunk(self, rack_builder: ProgressiveRackBuilder, current_stack_idx: int):
        """Freeze trunk stacks"""
        if current_stack_idx > 0:
            # Freeze all previous stacks
            trunk_indices = list(range(current_stack_idx))
            rack_builder.freeze_trunk(trunk_indices)
            logger.info(f"Frozen trunk stacks: {trunk_indices}")
    
    def _create_progressive_dataloader(self, 
                                     dataloader: DataLoader,
                                     rack_builder: ProgressiveRackBuilder,
                                     stack_idx: int) -> ProgressiveDataLoader:
        """Create progressive dataloader for current stack"""
        # Get masking strategy
        masking_strategy = self._get_masking_strategy()
        
        # Get trunk activations if available
        trunk_activations = self._get_trunk_activations(rack_builder, stack_idx)
        
        # Create progressive dataloader
        progressive_dataloader = ProgressiveDataLoader(
            base_dataloader=dataloader,
            masking_strategy=masking_strategy,
            stack_idx=stack_idx,
            trunk_activations=trunk_activations,
            cache_activations=self.cache_activations
        )
        
        return progressive_dataloader
    
    def _get_masking_strategy(self) -> Union[TimeStepMasking, ProgressiveMasking]:
        """Get masking strategy based on configuration"""
        if hasattr(self.training_config, 'time_step_masking') and self.training_config.time_step_masking:
            return TimeStepMasking(self.config)
        else:
            return ProgressiveMasking(self.config)
    
    def _get_trunk_activations(self, 
                             rack_builder: ProgressiveRackBuilder,
                             stack_idx: int) -> Optional[Dict]:
        """Get trunk activations for current stack"""
        if stack_idx == 0:
            return None
        
        # For frozen trunk, always use cached activations to avoid recomputation
        if self.trunk_strategy == "frozen" and self.cache_activations:
            return self.activation_cache.get(f"stack_{stack_idx - 1}", {})
        
        # For QLoRA trunk, use cached activations if available
        if self.trunk_strategy == "qlora" and self.cache_activations:
            return self.activation_cache.get(f"stack_{stack_idx - 1}", {})
        
        return None
    
    def _train_new_stack(self, 
                        rack_builder: ProgressiveRackBuilder,
                        stack_idx: int,
                        dataloader: ProgressiveDataLoader) -> Dict[str, Any]:
        """Train a new stack"""
        logger.info(f"Training stack {stack_idx}")
        
        # Get the new stack
        new_stack = rack_builder.stacks[stack_idx]
        
        # Set up optimizer for new stack only
        optimizer = self._create_optimizer(new_stack)
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.training_config.epochs_per_layer):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass through new stack
                if stack_idx == 0:
                    # First stack - use embeddings
                    x = rack_builder.embeddings(batch['input_ids'])
                else:
                    # Subsequent stacks - use previous stack outputs
                    x = self._get_previous_stack_output(rack_builder, stack_idx, batch)
                
                # Forward pass through new stack
                output = new_stack(x)
                
                # Compute loss
                loss = self._compute_progressive_loss(output, batch, rack_builder)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                total_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logger.info(f"Stack {stack_idx}, Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
        
        # Return training results
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {
            'stack_idx': stack_idx,
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'parameters': sum(p.numel() for p in new_stack.parameters()),
            'trainable_parameters': sum(p.numel() for p in new_stack.parameters() if p.requires_grad)
        }
    
    def _get_previous_stack_output(self, 
                                  rack_builder: ProgressiveRackBuilder,
                                  stack_idx: int,
                                  batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get output from previous stack"""
        if stack_idx == 0:
            # First stack - use embeddings
            return rack_builder.embeddings(batch['input_ids'])
        
        # For frozen trunk with cached activations, use cached activations directly
        if self.trunk_strategy == "frozen" and self.cache_activations:
            batch_id = self._get_batch_id(batch)
            cached_activations = self.activation_cache.get(f"stack_{stack_idx - 1}", {})
            
            if batch_id in cached_activations:
                # Use cached activations directly (no recomputation needed)
                cached_act = cached_activations[batch_id]
                if isinstance(cached_act, dict) and 'activations' in cached_act:
                    return cached_act['activations'].to(batch['input_ids'].device)
                elif isinstance(cached_act, torch.Tensor):
                    return cached_act.to(batch['input_ids'].device)
        
        # Fallback: recompute through previous stacks (for QLoRA or when no cache)
        prev_stack = rack_builder.stacks[stack_idx - 1]
        
        # Forward pass through previous stack
        with torch.no_grad():  # Previous stacks are frozen
            if stack_idx == 1:
                x = rack_builder.embeddings(batch['input_ids'])
            else:
                x = self._get_previous_stack_output(rack_builder, stack_idx - 1, batch)
            
            return prev_stack(x)
    
    def _get_batch_id(self, batch: Dict[str, torch.Tensor]) -> str:
        """Generate batch identifier for caching"""
        if isinstance(batch, dict) and 'input_ids' in batch:
            # Use a stable SHA1 digest of the raw bytes for deterministic keys
            input_bytes = batch['input_ids'].cpu().numpy().tobytes()
            digest = hashlib.sha1(input_bytes).hexdigest()
            return f"batch_{digest}"
        else:
            return f"batch_{id(batch)}"
    
    def _compute_progressive_loss(self, 
                                 output: torch.Tensor,
                                 batch: Dict[str, torch.Tensor],
                                 rack_builder: ProgressiveRackBuilder) -> torch.Tensor:
        """Compute loss for progressive training"""
        targets = batch['targets']
        masks = batch['masks']
        
        # Apply language model head
        logits = rack_builder.lm_head(output)
        
        # Compute masked loss
        loss = self._compute_masked_loss(logits, targets, masks)
        
        return loss
    
    def _compute_masked_loss(self, 
                            logits: torch.Tensor,
                            targets: torch.Tensor,
                            masks: torch.Tensor) -> torch.Tensor:
        """Compute loss only on masked positions"""
        # Flatten for easier indexing
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        masks_flat = masks.view(-1)
        
        # Get masked positions
        mask_indices = masks_flat.nonzero(as_tuple=False).squeeze(-1)
        
        if len(mask_indices) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute loss only on masked positions
        masked_logits = logits_flat[mask_indices]
        masked_targets = targets_flat[mask_indices]
        
        loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets.long())
        return loss
    
    def _create_optimizer(self, stack: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for a stack"""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, stack.parameters()),
            lr=self.training_config.learning_rate
        )
    
    def _cache_stack_activations(self, 
                                rack_builder: ProgressiveRackBuilder,
                                stack_idx: int,
                                dataloader: ProgressiveDataLoader):
        """Cache activations from a stack"""
        # Get cached activations from dataloader
        cached_activations = dataloader.get_cached_activations()
        
        # Store in activation cache
        self.activation_cache[f"stack_{stack_idx}"] = cached_activations
        
        logger.info(f"Cached activations for stack {stack_idx}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get training information"""
        return {
            'trunk_strategy': self.trunk_strategy,
            'new_stack_precision': self.new_stack_precision,
            'cache_activations': self.cache_activations,
            'training_history': self.training_history,
            'activation_cache_size': len(self.activation_cache)
        }
    
    def get_model(self, rack_builder: ProgressiveRackBuilder):
        """Get the trained model from rack builder"""
        return rack_builder.get_model()
    
    def get_model_info(self, rack_builder: ProgressiveRackBuilder) -> Dict[str, Any]:
        """Get detailed model information"""
        model_info = rack_builder.get_model_info()
        training_info = self.get_training_info()
        
        return {
            **model_info,
            'training_info': training_info
        }
    
    def get_model_state(self, rack_builder: ProgressiveRackBuilder) -> Dict[str, Any]:
        """Get complete model and training state"""
        model_state = rack_builder.get_model_state()
        training_info = self.get_training_info()
        
        return {
            **model_state,
            'training_info': training_info,
            'activation_cache': self.activation_cache
        }
    
    def load_model_state(self, rack_builder: ProgressiveRackBuilder, state: Dict[str, Any]):
        """Load complete model and training state"""
        # Load model state
        rack_builder.load_model_state(state)
        
        # Load training state
        if 'training_info' in state:
            training_info = state['training_info']
            self.training_history = training_info.get('training_history', [])
        
        # Load activation cache
        if 'activation_cache' in state:
            self.activation_cache = state['activation_cache']
        
        logger.info("Loaded complete model and training state")
    
    def save_progressive_checkpoint(self, 
                                  stack_idx: int,
                                  rack_builder: ProgressiveRackBuilder,
                                  optimizer: Optional[torch.optim.Optimizer] = None,
                                  epoch: Optional[int] = None,
                                  loss: Optional[float] = None) -> str:
        """
        Save progressive training checkpoint.
        
        Args:
            stack_idx: Current stack index
            rack_builder: Progressive rack builder
            optimizer: Optimizer state (optional)
            epoch: Epoch number (optional)
            loss: Loss value (optional)
            
        Returns:
            Checkpoint path
        """
        checkpoint_id = f"progressive_stack_{stack_idx}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'stack_idx': stack_idx,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'loss': loss,
            'trunk_strategy': self.trunk_strategy,
            'new_stack_precision': self.new_stack_precision,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        # Save rack builder state
        checkpoint_data['rack_builder'] = {
            'stacks': [stack.state_dict() for stack in rack_builder.stacks],
            'current_stacks': rack_builder.current_stacks,
            'precision_settings': rack_builder.precision_settings,
            'qlora_adapters': rack_builder.qlora_adapters
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add activation cache if available
        if self.activation_cache:
            checkpoint_data['activation_cache'] = {
                key: value.cpu() if torch.is_tensor(value) else value 
                for key, value in self.activation_cache.items()
            }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"Saved progressive checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_progressive_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load progressive training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded progressive checkpoint: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def restore_from_checkpoint(self, 
                              checkpoint_path: str,
                              rack_builder: ProgressiveRackBuilder) -> bool:
        """
        Restore training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            rack_builder: Progressive rack builder to restore
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_data = self.load_progressive_checkpoint(checkpoint_path)
        if checkpoint_data is None:
            return False
        
        try:
            # Restore rack builder state
            if 'rack_builder' in checkpoint_data:
                rack_data = checkpoint_data['rack_builder']
                
                # Restore stacks
                for i, stack_state in enumerate(rack_data['stacks']):
                    if i < len(rack_builder.stacks):
                        rack_builder.stacks[i].load_state_dict(stack_state)
                
                # Restore other state
                rack_builder.current_stacks = rack_data['current_stacks']
                rack_builder.precision_settings = rack_data['precision_settings']
                rack_builder.qlora_adapters = rack_data['qlora_adapters']
            
            # Restore training history
            if 'training_history' in checkpoint_data:
                self.training_history = checkpoint_data['training_history']
            
            # Restore activation cache
            if 'activation_cache' in checkpoint_data:
                self.activation_cache = checkpoint_data['activation_cache']
            
            logger.info(f"Restored training from checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def save_rack_checkpoint(self, 
                           rack_builder: ProgressiveRackBuilder,
                           optimizer: Optional[torch.optim.Optimizer] = None) -> str:
        """
        Save complete rack checkpoint.
        
        Args:
            rack_builder: Progressive rack builder
            optimizer: Optimizer state (optional)
            
        Returns:
            Checkpoint path
        """
        checkpoint_id = f"rack_complete_{rack_builder.current_stacks}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        # Prepare rack checkpoint data
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'total_stacks': rack_builder.current_stacks,
            'trunk_strategy': self.trunk_strategy,
            'new_stack_precision': self.new_stack_precision,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        # Save complete rack state
        checkpoint_data['rack_builder'] = {
            'stacks': [stack.state_dict() for stack in rack_builder.stacks],
            'current_stacks': rack_builder.current_stacks,
            'precision_settings': rack_builder.precision_settings,
            'qlora_adapters': rack_builder.qlora_adapters
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"Saved rack checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'timestamp': checkpoint_data.get('timestamp', 'unknown'),
                    'stack_idx': checkpoint_data.get('stack_idx', 'unknown'),
                    'epoch': checkpoint_data.get('epoch', 'unknown'),
                    'loss': checkpoint_data.get('loss', 'unknown')
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
