"""
Fusion-specific training logic for progressive training strategies.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class QLoRAAdapter(nn.Module):
    """
    QLoRA (Quantized Low-Rank Adaptation) adapter for efficient fine-tuning.
    
    This adapter wraps a quantized layer and adds low-rank adaptation matrices
    for efficient fine-tuning while maintaining the original layer's functionality.
    """
    
    def __init__(self, original_layer: nn.Module, rank: int = 16, alpha: float = 32.0, 
                 dropout: float = 0.1, name: str = "qlora_adapter"):
        """
        Initialize QLoRA adapter.
        
        Args:
            original_layer: The original quantized layer to adapt
            rank: Rank of the low-rank adaptation matrices
            alpha: Scaling factor for the adaptation
            dropout: Dropout rate for the adaptation
            name: Name for the adapter
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.name = name
        
        # Get dimensions from original layer
        if hasattr(original_layer, 'weight'):
            self.in_features = original_layer.weight.shape[1]
            self.out_features = original_layer.weight.shape[0]
        else:
            raise ValueError("Original layer must have a weight attribute")
        
        # Create low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize B to zero so that initial adaptation is zero
        nn.init.zeros_(self.lora_B)
        
        logger.debug(f"Created QLoRA adapter {name}: {self.in_features} -> {self.out_features}, rank={rank}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through QLoRA adapter.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with low-rank adaptation
        """
        # Original layer forward pass
        original_output = self.original_layer(x)
        
        # Low-rank adaptation
        # x @ A^T @ B^T = x @ (B @ A)^T
        adaptation = self.dropout_layer(x) @ self.lora_A.T @ self.lora_B.T
        
        # Scale by alpha/rank
        scaled_adaptation = adaptation * (self.alpha / self.rank)
        
        # Add adaptation to original output
        return original_output + scaled_adaptation
    
    def get_adaptation_parameters(self):
        """
        Get only the adaptation parameters (A and B matrices).
        
        Returns:
            List of adaptation parameters
        """
        return [self.lora_A, self.lora_B]
    
    def freeze_original_layer(self):
        """
        Freeze the original layer parameters.
        """
        for param in self.original_layer.parameters():
            param.requires_grad = False
        logger.debug(f"Frozen original layer in {self.name}")

# Removed duplicate FusionTrainer class - using the first one defined above

class FusionTrainer:
    """
    Fusion-specific training logic for progressive training strategies.
    
    This class handles fusion training modes including frozen and trainable backbones,
    quantized loading, QLoRA adapters, and time-step-based masking.
    """
    
    def __init__(self, config, masking_strategy=None, quantization_manager=None, 
                 cache_manager=None, lexical_kernel_manager=None):
        """
        Initialize FusionTrainer.
        
        Args:
            config: Training configuration
            masking_strategy: Strategy for generating masks
            quantization_manager: Manager for quantization operations
            cache_manager: Manager for activation caching
            lexical_kernel_manager: Manager for lexical kernel operations
        """
        self.config = config
        self.masking_strategy = masking_strategy
        self.quantization_manager = quantization_manager
        self.cache_manager = cache_manager
        self.lexical_kernel_manager = lexical_kernel_manager
        
        # Initialize caches for persistent quantization
        self._quantized_backbone_cache = {}
        self._qlora_cache = {}
        self._trained_blocks_cache = {}
        
        logger.info("Initialized FusionTrainer")
    
    def train_fused_blocks(self, block_idx: int, dataloader: DataLoader, 
                          all_blocks: List[List[torch.nn.Module]]):
        """
        Train with fused blocks strategy.
        
        Args:
            block_idx: Current block index
            dataloader: Data loader
            all_blocks: All blocks in the model
        """
        logger.info(f"Training fused blocks up to block {block_idx}")
        
        if self.config.fusion_mode == "frozen":
            self._train_with_frozen_backbone(block_idx, dataloader, all_blocks)
        elif self.config.fusion_mode == "trainable":
            self._train_with_trainable_backbone(block_idx, dataloader, all_blocks)
        else:
            raise ValueError(f"Unknown fusion mode: {self.config.fusion_mode}")
    
    def _train_with_frozen_backbone(self, block_idx: int, dataloader: DataLoader,
                                   all_blocks: List[List[torch.nn.Module]]):
        """
        Train current block with frozen backbone (previous blocks).
        
        Args:
            block_idx: Current block index
            dataloader: Data loader
            all_blocks: All blocks in the model
        """
        logger.info(f"Training with frozen blocks up to {block_idx}")
        
        # Freeze previous blocks
        for i in range(block_idx):
            self._freeze_block(all_blocks[i])
        
        # Setup optimizer for current block only
        current_block = all_blocks[block_idx]
        optimizer = self._setup_fusion_optimizer([current_block], qlora_enabled=False, current_block_idx=0)
        
        # Train current block
        self._train_current_block(block_idx, dataloader, current_block, optimizer)
        
        # No caching needed in fusion training - we always resample fresh inputs
        # Use block_trainer for layer-wise training where caching is beneficial
        
        # Memory management: Convert trained current block to low precision
        # current_block is already a list of layers, so we wrap it in a list to make it a list of blocks
        self._convert_trained_blocks_to_low_precision([current_block])
    
    def _train_with_trainable_backbone(self, block_idx: int, dataloader: DataLoader,
                                      all_blocks: List[List[torch.nn.Module]]):
        """
        Train current block with trainable backbone (all previous blocks).
        
        Args:
            block_idx: Current block index
            dataloader: Data loader
            all_blocks: All blocks in the model
        """
        logger.info(f"Training with all blocks trainable up to {block_idx}")
        
        # Make all blocks up to current trainable
        trainable_blocks = all_blocks[:block_idx + 1]
        
        # Setup optimizer for all trainable blocks with QLoRA support
        optimizer = self._setup_fusion_optimizer(trainable_blocks, qlora_enabled=True, current_block_idx=block_idx)
        
        # Train all blocks together with quantized backbone + QLoRA
        self._train_all_blocks_together(block_idx, dataloader, trainable_blocks, optimizer,
                                       backbone_precision="fp16", qlora_enabled=True)
        
        # No caching needed in fusion training - we always resample fresh inputs
        # Use block_trainer for layer-wise training where caching is beneficial
        
        # Memory management: Convert trained blocks to low precision and remove from memory
        self._convert_trained_blocks_to_low_precision(trainable_blocks)
    
    def _freeze_block(self, block_layers: List[torch.nn.Module]):
        """
        Freeze a block's parameters.
        
        Args:
            block_layers: Layers in the block
        """
        for layer in block_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        logger.debug("Frozen block parameters")
    
    def _train_current_block(self, block_idx: int, dataloader: DataLoader,
                            block_layers: List[torch.nn.Module], 
                            optimizer: torch.optim.Optimizer):
        """
        Train current block with frozen previous blocks.
        
        Args:
            block_idx: Block index
            dataloader: Data loader
            block_layers: Layers in current block
            optimizer: Pre-configured optimizer for the block
        """
        
        # Train current block
        for epoch in range(self.config.epochs_per_block):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                loss = self._train_batch_with_quantized_backbone(batch, block_layers, optimizer)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Block {block_idx}, Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def _train_all_blocks_together(self, block_idx: int, dataloader: DataLoader,
                                  trainable_blocks: List[List[torch.nn.Module]], 
                                  optimizer: torch.optim.Optimizer,
                                  backbone_precision: str = "fp16",
                                  qlora_enabled: bool = False):
        """
        Train all blocks together.
        
        Args:
            block_idx: Current block index
            dataloader: Data loader
            trainable_blocks: All trainable blocks
            optimizer: Optimizer for all blocks
        """
        for epoch in range(self.config.epochs_per_block):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                loss = self._train_batch_with_quantized_backbone_all_blocks(batch, trainable_blocks, optimizer,
                                                                           block_idx, backbone_precision, qlora_enabled)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"All blocks, Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def _train_batch_with_quantized_backbone_single_block(self, batch: Dict[str, torch.Tensor],
                                                        block_layers: List[torch.nn.Module],
                                                        optimizer: torch.optim.Optimizer) -> float:
        """
        Train batch with quantized backbone for single block.
        All previous blocks are frozen and loaded in quantized precision.
        The current block is trained in full precision.
        Args:
            batch: Training batch
            block_layers: Current block layers
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        optimizer.zero_grad()
        
        # Generate masks for current block (use default masking strategy)
        if self.masking_strategy is not None:
            masks = self.masking_strategy.generate_masks(
                batch["input_ids"], 
                block_idx=0,  # Use block 0 for single block training
                time_t=None  # Use default time step for fusion training
            )
        else:
            # Fallback: create simple masks (15% masking)
            batch_size, seq_len = batch["input_ids"].shape
            masks = torch.rand(batch_size, seq_len) < 0.15
            masks = masks.to(batch["input_ids"].device)
        
        # Apply masks to input
        masked_input = batch["input_ids"].clone()
        masked_input[masks] = self.config.mask_token_id
        
        # Forward pass through current block
        hidden_states = masked_input
        
        for layer in block_layers:
            hidden_states = layer(hidden_states)
        
        # Apply language model head (tied embeddings)
        if self.lexical_kernel_manager is not None:
            lm_head = self.lexical_kernel_manager.get_lm_head()
            logits = lm_head(hidden_states)
        else:
            # Fallback: use hidden states directly (placeholder)
            logits = hidden_states
        
        # Compute loss with proper logits and masks
        loss = self._compute_fusion_loss(logits, batch["input_ids"], masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _train_batch_with_quantized_backbone_all_blocks(self, batch: Dict[str, torch.Tensor],
                                                        trainable_blocks: List[List[torch.nn.Module]],
                                                        optimizer: torch.optim.Optimizer,
                                                        block_idx: int,
                                                        backbone_precision: str = "fp16",
                                                        qlora_enabled: bool = False) -> float:
        """
        Train batch with all blocks using quantized backbone + optional QLoRA for each block.
        The current block is trained in full precision.
        
        Args:
            batch: Training batch
            trainable_blocks: All trainable blocks
            optimizer: Optimizer
            block_idx: Current block index
            backbone_precision: Backbone precision ("nf_fp4", "fp8", "fp16", "fp32")
            qlora_enabled: Use QLoRA adapters for backbone fine-tuning
            
        Returns:
            Loss value
        """
        optimizer.zero_grad()
        
        # Unified approach: Free backbone, resample inputs, create masks at current depth
        backbone_blocks = trainable_blocks[:-1]  # All except current block
        current_block = trainable_blocks[-1]      # Current block in full precision
        
        # Step 1: Free backbone (convert to low precision for memory efficiency)
        if backbone_blocks:
            backbone_blocks = self._freeze_and_quantize_backbone(
                backbone_blocks, precision=backbone_precision, qlora_enabled=qlora_enabled
            )
        
        # Step 2: Resample fresh inputs from dataloader with adaptive strategy
        resampling_strategy = self._get_resampling_strategy_for_block(block_idx)
        resampled_batch = self._resample_fresh_inputs(batch, resampling_strategy, block_idx)
        
        # Step 3: Create masks at current depth (time-step-based)
        masks = self._create_masks_at_current_depth(resampled_batch, block_idx)
        
        # Step 4: Forward pass through backbone to get activations
        hidden_states = self._forward_through_backbone(resampled_batch, backbone_blocks, masks)
        
        # Step 5: Forward through current block (full precision) using backbone activations
        current_hidden_states = hidden_states
        for layer in current_block:
            current_hidden_states = layer(current_hidden_states)
        
        # Update hidden_states for loss computation
        hidden_states = current_hidden_states
        
        # Apply language model head (tied embeddings)
        if self.lexical_kernel_manager is not None:
            lm_head = self.lexical_kernel_manager.get_lm_head()
            logits = lm_head(hidden_states)
        else:
            # Fallback: use hidden states directly (placeholder)
            logits = hidden_states
        
        # Compute loss with proper logits and masks
        loss = self._compute_fusion_loss(logits, batch["input_ids"], masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _compute_fusion_loss(self, hidden_states: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Compute cross-entropy loss for fusion training.
        
        Args:
            hidden_states: Model hidden states
            targets: Target tokens
            masks: Mask tensor (optional, for masked training)
            
        Returns:
            Loss tensor
        """
        # Apply language model head (tied embeddings)
        if self.lexical_kernel_manager is not None:
            lm_head = self.lexical_kernel_manager.get_lm_head()
            logits = lm_head(hidden_states)
        else:
            # Fallback: use hidden states directly (placeholder)
            logits = hidden_states
        
        if masks is not None:
            # Masked training: only compute loss on masked positions
            # Flatten for easier indexing
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
            targets_flat = targets.view(-1)  # (batch_size * seq_len,)
            masks_flat = masks.view(-1)  # (batch_size * seq_len,)
            
            # Get indices where mask is True
            mask_indices = masks_flat.nonzero(as_tuple=False).squeeze(-1)  # (num_masked,)
            
            if len(mask_indices) == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            masked_logits = logits_flat[mask_indices]  # (num_masked, vocab_size)
            masked_targets = targets_flat[mask_indices]  # (num_masked,)
            loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets.long())
        else:
            # Full sequence training: compute loss on all positions
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1).long()
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=0)
        
        return loss
    
    def _setup_fusion_optimizer(self, trainable_blocks: List[List[torch.nn.Module]], 
                               qlora_enabled: bool = False, current_block_idx: int = None) -> torch.optim.Optimizer:
        """
        Setup optimizer for fusion training with proper parameter collection.
        
        Args:
            trainable_blocks: All trainable blocks
            qlora_enabled: Whether to use QLoRA parameters for backbone
            current_block_idx: Index of current block (always trainable)
            
        Returns:
            Optimizer instance
        """
        # Create parameter groups with different learning rates
        param_groups = []
        
        
        # Validate current_block_idx
        if current_block_idx is None or current_block_idx >= len(trainable_blocks):
            logger.error(f"Invalid current_block_idx: {current_block_idx}, trainable_blocks length: {len(trainable_blocks)}")
            # Fallback: use all parameters with single learning rate
            all_params = []
            for block in trainable_blocks:
                for layer in block:
                    all_params.extend([p for p in layer.parameters() if p.requires_grad])
            param_groups.append({
                'params': all_params,
                'lr': self.config.learning_rate,
                'name': 'all_parameters'
            })
        else:
            # Add QLoRA backbone parameters if enabled
            if qlora_enabled and current_block_idx > 0:
                backbone_blocks = trainable_blocks[:current_block_idx]
                qlora_params = self._collect_qlora_parameters(backbone_blocks)
                
                if qlora_params:  # Only add if we have QLoRA parameters
                    qlora_lr = getattr(self.config, 'qlora_lr', self.config.learning_rate * 0.1)
                    param_groups.append({
                        'params': qlora_params,
                        'lr': qlora_lr,
                        'name': 'qlora_backbone'
                    })
                    logger.debug(f"Added QLoRA backbone group: {len(qlora_params)} params, lr={qlora_lr}")
                else:
                    logger.warning("QLoRA enabled but no QLoRA parameters found in backbone")
            
            # Always add current block parameters
            current_block = trainable_blocks[current_block_idx]
            current_params = []
            for layer in current_block:
                current_params.extend([p for p in layer.parameters() if p.requires_grad])
            
            if current_params:
                current_lr = getattr(self.config, 'current_block_lr', self.config.learning_rate)
                param_groups.append({
                    'params': current_params,
                    'lr': current_lr,
                    'name': 'current_block'
                })
                logger.debug(f"Added current block group: {len(current_params)} params, lr={current_lr}")
            else:
                logger.warning("No trainable parameters found in current block")
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(param_groups)
        
        # Log parameter group details
        total_params = sum(len(group['params']) for group in param_groups)
        lr_info = {group['name']: f"{len(group['params'])} params @ lr={group['lr']}" for group in param_groups}
        logger.info(f"Setup fusion optimizer: {total_params} total parameters")
        logger.info(f"Parameter groups: {lr_info}")
        
        return optimizer
    
    def _get_or_load_quantized_backbone(self, blocks: List[List[torch.nn.Module]], 
                                       precision: str = "fp16") -> List[List[torch.nn.Module]]:
        """
        Get cached quantized backbone or load and cache it.
        
        Args:
            blocks: Backbone blocks to load
            precision: Target precision ("nf_fp4", "fp8", "fp16", "fp32")
            
        Returns:
            Cached or newly quantized blocks
        """
        # Create cache key based on block structure and precision
        cache_key = f"backbone_{id(blocks)}_{precision}"
        
        # Check if already cached
        if hasattr(self, '_quantized_backbone_cache') and cache_key in self._quantized_backbone_cache:
            logger.debug(f"Using cached quantized backbone ({precision})")
            return self._quantized_backbone_cache[cache_key]
        
        # Load and cache backbone in specified precision
        logger.info(f"Loading and caching backbone in {precision} precision")
        quantized_blocks = self._load_backbone_in_precision(blocks, precision)
        
        # Cache the quantized backbone
        if not hasattr(self, '_quantized_backbone_cache'):
            self._quantized_backbone_cache = {}
        self._quantized_backbone_cache[cache_key] = quantized_blocks
        
        return quantized_blocks
    
    def _cache_trained_block_in_low_precision(self, trained_blocks: List[List[torch.nn.Module]], 
                                             cache_precision: str = "fp8") -> List[List[torch.nn.Module]]:
        """
        Cache trained blocks in low precision for memory efficiency.
        
        Args:
            trained_blocks: Blocks trained in FP16/FP32
            cache_precision: Precision for caching ("nf_fp4", "fp8")
            
        Returns:
            Blocks cached in low precision
        """
        # Create cache key for trained blocks
        cache_key = f"trained_{id(trained_blocks)}_{cache_precision}"
        
        # Convert trained blocks to low precision for caching
        logger.info(f"Caching trained blocks in {cache_precision} precision")
        cached_blocks = self._load_backbone_in_precision(trained_blocks, cache_precision)
        
        # Cache the low-precision blocks
        if not hasattr(self, '_trained_blocks_cache'):
            self._trained_blocks_cache = {}
        self._trained_blocks_cache[cache_key] = cached_blocks
        
        logger.debug(f"Cached trained blocks in {cache_precision} precision")
        return cached_blocks
    
    def _get_cached_trained_blocks(self, block_id: int, cache_precision: str = "fp8") -> List[List[torch.nn.Module]]:
        """
        Retrieve cached trained blocks in low precision.
        
        Args:
            block_id: ID of the trained block
            cache_precision: Precision of cached blocks ("nf_fp4", "fp8")
            
        Returns:
            Cached trained blocks in low precision
        """
        cache_key = f"trained_{block_id}_{cache_precision}"
        
        if hasattr(self, '_trained_blocks_cache') and cache_key in self._trained_blocks_cache:
            logger.debug(f"Retrieved cached trained blocks ({cache_precision})")
            return self._trained_blocks_cache[cache_key]
        
        logger.warning(f"No cached trained blocks found for block {block_id} in {cache_precision}")
        return None
    
    def _convert_trained_blocks_to_low_precision(self, 
                                                trained_blocks: List[List[torch.nn.Module]],
                                                cache_precision: str = "fp8"):
        """
        Convert trained blocks to low precision and remove full-precision versions from memory.
        
        This is critical for memory management - after training, we convert to low precision
        and remove the full-precision weights to prevent memory accumulation.
        
        Args:
            trained_blocks: Blocks trained in full precision
            cache_precision: Target precision for caching ("nf_fp4", "fp8")
        """
        logger.info(f"Converting trained blocks to {cache_precision} precision for memory efficiency")
        
        # Save full-precision weights to disk before conversion (optional backup)
        self._save_full_precision_weights_to_disk(trained_blocks, cache_precision)
        
        # Convert blocks in-place to low precision (memory efficient)
        for block in trained_blocks:
            for layer in block:
                # Convert layer weights to low precision
                if cache_precision == "fp8":
                    # TODO: Implement FP8 conversion
                    logger.warning("FP8 conversion not implemented yet - using FP16 fallback")
                    layer.half()  # Fallback to FP16
                elif cache_precision == "nf_fp4":
                    # TODO: Implement NF FP4 conversion
                    logger.warning("NF FP4 conversion not implemented yet - using FP16 fallback")
                    layer.half()  # Fallback to FP16
                else:
                    # FP16 conversion (supported by PyTorch)
                    layer.half()
        
        # Clear gradients to free memory
        for block in trained_blocks:
            for layer in block:
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    layer.weight.grad = None
                if hasattr(layer, 'bias') and layer.bias is not None and layer.bias.grad is not None:
                    layer.bias.grad = None
        
        logger.info(f"Converted blocks to {cache_precision} precision and saved full-precision backup to disk")
    
    def _save_full_precision_weights_to_disk(self, trained_blocks: List[List[torch.nn.Module]], 
                                           cache_precision: str):
        """
        Save full-precision weights to disk before converting to low precision.
        
        This provides a backup mechanism in case we need to restore full precision later.
        Uses run_id from config for organized file structure.
        
        Args:
            trained_blocks: Blocks to save
            cache_precision: Target precision (for filename)
        """
        import os
        import torch
        import json
        from datetime import datetime
        
        # Get run_id from config
        run_id = getattr(self.config, 'run_id', 'default_run')
        
        # Create run-specific backup directory
        backup_dir = "checkpoints/full_precision_backups"
        run_dir = f"{backup_dir}/{run_id}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Save run metadata
        metadata = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'precision': cache_precision,
            'block_count': len(trained_blocks),
            'block_shapes': []
        }
        
        # Save each block separately for easier restoration
        for block_idx, block in enumerate(trained_blocks):
            block_state = {}
            block_shapes = []
            
            for layer_idx, layer in enumerate(block):
                layer_name = f"block_{block_idx}_layer_{layer_idx}"
                layer_state = {}
                
                # Save all parameters
                for param_name, param in layer.named_parameters():
                    layer_state[param_name] = param.clone().detach()
                    block_shapes.append({
                        'layer': layer_name,
                        'param': param_name,
                        'shape': list(param.shape),
                        'dtype': str(param.dtype)
                    })
                
                # Save all buffers (like running stats in BatchNorm)
                for buffer_name, buffer in layer.named_buffers():
                    layer_state[buffer_name] = buffer.clone().detach()
                    block_shapes.append({
                        'layer': layer_name,
                        'buffer': buffer_name,
                        'shape': list(buffer.shape),
                        'dtype': str(buffer.dtype)
                    })
                
                block_state[layer_name] = layer_state
            
            # Save block to disk with run_id organization
            filename = f"{run_dir}/block_{block_idx}_{cache_precision}.pt"
            torch.save(block_state, filename)
            logger.debug(f"Saved full-precision weights for block {block_idx} to {filename}")
            
            metadata['block_shapes'].append(block_shapes)
        
        # Save run metadata
        metadata_file = f"{run_dir}/metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved full-precision backup for {len(trained_blocks)} blocks to {run_dir}")
        
        # Validate saved weights
        self._validate_saved_weights(run_id, cache_precision, len(trained_blocks))
    
    def _validate_saved_weights(self, run_id: str, cache_precision: str, expected_blocks: int) -> bool:
        """
        Validate that saved weights can be loaded correctly.
        
        Args:
            run_id: Run identifier
            cache_precision: Precision of saved weights
            expected_blocks: Expected number of blocks
            
        Returns:
            True if validation passes, False otherwise
        """
        import os
        import torch
        import json
        
        try:
            backup_dir = "checkpoints/full_precision_backups"
            run_dir = f"{backup_dir}/{run_id}"
            
            # Check if run directory exists
            if not os.path.exists(run_dir):
                logger.error(f"Run directory not found: {run_dir}")
                return False
            
            # Check metadata file
            metadata_file = f"{run_dir}/metadata.json"
            if not os.path.exists(metadata_file):
                logger.error(f"Metadata file not found: {metadata_file}")
                return False
            
            # Load and validate metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata['block_count'] != expected_blocks:
                logger.error(f"Block count mismatch: expected {expected_blocks}, got {metadata['block_count']}")
                return False
            
            # Validate each block file
            for block_idx in range(expected_blocks):
                filename = f"{run_dir}/block_{block_idx}_{cache_precision}.pt"
                if not os.path.exists(filename):
                    logger.error(f"Block file not found: {filename}")
                    return False
                
                # Try to load the block
                try:
                    block_state = torch.load(filename, map_location='cpu')
                    if not isinstance(block_state, dict):
                        logger.error(f"Invalid block state format for block {block_idx}")
                        return False
                except Exception as e:
                    logger.error(f"Failed to load block {block_idx}: {e}")
                    return False
            
            logger.info(f"Successfully validated {expected_blocks} blocks for run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _restore_full_precision_weights_from_disk(self, run_id: str, 
                                                 cache_precision: str, 
                                                 block_indices: List[int] = None) -> Dict[int, Dict[str, torch.nn.Module]]:
        """
        Restore full-precision weights from disk backup.
        
        Args:
            run_id: Run identifier
            cache_precision: Precision of saved weights
            block_indices: List of block indices to restore (None for all)
            
        Returns:
            Dictionary mapping block_idx to restored block state
        """
        import os
        import torch
        import json
        
        backup_dir = "checkpoints/full_precision_backups"
        run_dir = f"{backup_dir}/{run_id}"
        
        if not os.path.exists(run_dir):
            logger.error(f"Run directory not found: {run_dir}")
            return {}
        
        # Load metadata to get block count
        metadata_file = f"{run_dir}/metadata.json"
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file not found: {metadata_file}")
            return {}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        total_blocks = metadata['block_count']
        
        # Determine which blocks to restore
        if block_indices is None:
            block_indices = list(range(total_blocks))
        
        restored_blocks = {}
        
        for block_idx in block_indices:
            if block_idx >= total_blocks:
                logger.error(f"Block index {block_idx} out of range (max: {total_blocks-1})")
                continue
            
            filename = f"{run_dir}/block_{block_idx}_{cache_precision}.pt"
            if not os.path.exists(filename):
                logger.error(f"Block file not found: {filename}")
                continue
            
            try:
                # Load the block state
                block_state = torch.load(filename, map_location='cpu')
                restored_blocks[block_idx] = block_state
                logger.debug(f"Successfully restored block {block_idx} from {filename}")
            except Exception as e:
                logger.error(f"Failed to restore block {block_idx}: {e}")
                continue
        
        logger.info(f"Restored {len(restored_blocks)} blocks from run {run_id}")
        return restored_blocks
    
    def _reconstruct_model_from_disk(self, run_id: str, cache_precision: str) -> List[Dict[str, torch.nn.Module]]:
        """
        Reconstruct the full model from disk backups.
        
        Args:
            run_id: Run identifier
            cache_precision: Precision of saved weights
            
        Returns:
            List of block states (each block is a dict of layer states)
        """
        import os
        import torch
        import json
        
        backup_dir = "checkpoints/full_precision_backups"
        run_dir = f"{backup_dir}/{run_id}"
        
        if not os.path.exists(run_dir):
            logger.error(f"Run directory not found: {run_dir}")
            return []
        
        # Load metadata
        metadata_file = f"{run_dir}/metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        total_blocks = metadata['block_count']
        reconstructed_blocks = []
        
        for block_idx in range(total_blocks):
            filename = f"{run_dir}/block_{block_idx}_{cache_precision}.pt"
            if not os.path.exists(filename):
                logger.error(f"Block file not found: {filename}")
                continue
            
            try:
                block_state = torch.load(filename, map_location='cpu')
                # Note: This returns the state dict, not the actual model
                # The caller would need to reconstruct the model architecture
                reconstructed_blocks.append(block_state)
                logger.debug(f"Loaded block {block_idx} state from {filename}")
            except Exception as e:
                logger.error(f"Failed to load block {block_idx}: {e}")
                continue
        
        logger.info(f"Reconstructed {len(reconstructed_blocks)} blocks from run {run_id}")
        return reconstructed_blocks
    
    def _freeze_and_quantize_backbone(self, backbone_blocks: List[List[torch.nn.Module]], 
                                     precision: str = "fp16", qlora_enabled: bool = False) -> List[List[torch.nn.Module]]:
        """
        Freeze backbone and convert to low precision for memory efficiency.
        Unified approach for both frozen and trainable backbone modes.
        
        Args:
            backbone_blocks: Backbone blocks to freeze and quantize
            precision: Target precision for quantization
            qlora_enabled: Apply QLoRA adapters for trainable backbone
            
        Returns:
            Frozen and quantized backbone blocks
        """
        logger.info(f"Freezing and quantizing backbone in {precision} precision")
        
        # Freeze all backbone parameters
        for block in backbone_blocks:
            for layer in block:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Convert to low precision
        quantized_backbone = self._load_backbone_in_precision(backbone_blocks, precision)
        
        # Apply QLoRA adapters if enabled (for trainable backbone mode)
        if qlora_enabled:
            quantized_backbone = self._get_or_apply_qlora_adapters(quantized_backbone)
        
        logger.debug(f"Backbone frozen and quantized in {precision} precision")
        return quantized_backbone
    
    def _resample_fresh_inputs(self, batch: Dict[str, torch.Tensor], 
                              resampling_strategy: str = "random", block_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Resample fresh inputs from dataloader for unified training approach.
        
        Args:
            batch: Original batch from dataloader
            resampling_strategy: Strategy for resampling ("random", "sequential", "weighted")
            
        Returns:
            Resampled batch with fresh inputs
        """
        logger.debug(f"Resampling fresh inputs using {resampling_strategy} strategy")
        
        if resampling_strategy == "random":
            return self._random_resample(batch)
        elif resampling_strategy == "sequential":
            return self._sequential_resample(batch)
        elif resampling_strategy == "weighted":
            return self._weighted_resample(batch)
        elif resampling_strategy == "adaptive":
            return self._adaptive_resample(batch, block_idx)
        else:
            logger.warning(f"Unknown resampling strategy: {resampling_strategy}, using random")
            return self._random_resample(batch)
    
    def _random_resample(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Random resampling strategy - shuffle the batch randomly.
        
        Args:
            batch: Original batch
            
        Returns:
            Randomly resampled batch
        """
        batch_size = batch["input_ids"].shape[0]
        
        # Create random permutation indices
        perm_indices = torch.randperm(batch_size, device=batch["input_ids"].device)
        
        # Apply permutation to all batch components
        resampled_batch = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == batch_size:
                resampled_batch[key] = tensor[perm_indices]
            else:
                resampled_batch[key] = tensor
        
        logger.debug(f"Random resampling: shuffled {batch_size} samples")
        return resampled_batch
    
    def _sequential_resample(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Sequential resampling strategy - maintain order but potentially repeat samples.
        
        Args:
            batch: Original batch
            
        Returns:
            Sequentially resampled batch
        """
        batch_size = batch["input_ids"].shape[0]
        
        # Create sequential indices (potentially with repetition)
        seq_indices = torch.arange(batch_size, device=batch["input_ids"].device)
        
        # Apply sequential sampling to all batch components
        resampled_batch = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == batch_size:
                resampled_batch[key] = tensor[seq_indices]
            else:
                resampled_batch[key] = tensor
        
        logger.debug(f"Sequential resampling: maintained order for {batch_size} samples")
        return resampled_batch
    
    def _weighted_resample(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Weighted resampling strategy - resample based on sample weights or importance.
        
        Args:
            batch: Original batch
            
        Returns:
            Weighted resampled batch
        """
        batch_size = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        
        # Create weights (could be based on sequence length, difficulty, etc.)
        # For now, use uniform weights
        weights = torch.ones(batch_size, device=device)
        
        # Create weighted sampling indices
        weighted_indices = torch.multinomial(weights, batch_size, replacement=True)
        
        # Apply weighted sampling to all batch components
        resampled_batch = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == batch_size:
                resampled_batch[key] = tensor[weighted_indices]
            else:
                resampled_batch[key] = tensor
        
        logger.debug(f"Weighted resampling: sampled {batch_size} samples with weights")
        return resampled_batch
    
    def _adaptive_resample(self, batch: Dict[str, torch.Tensor], 
                          block_idx: int, training_history: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Adaptive resampling strategy - adjust resampling based on training progress.
        
        Args:
            batch: Original batch
            block_idx: Current block index
            training_history: History of training metrics (optional)
            
        Returns:
            Adaptively resampled batch
        """
        batch_size = batch["input_ids"].shape[0]
        
        # Adaptive strategy based on block depth
        if block_idx == 0:
            # Early blocks: use sequential resampling for stability
            return self._sequential_resample(batch)
        elif block_idx < 3:
            # Middle blocks: use weighted resampling
            return self._weighted_resample(batch)
        else:
            # Later blocks: use random resampling for diversity
            return self._random_resample(batch)
    
    def _create_resampling_schedule(self, total_blocks: int) -> List[str]:
        """
        Create a resampling schedule for progressive training.
        
        Args:
            total_blocks: Total number of blocks to train
            
        Returns:
            List of resampling strategies for each block
        """
        schedule = []
        
        for block_idx in range(total_blocks):
            if block_idx == 0:
                # First block: sequential for stability
                schedule.append("sequential")
            elif block_idx < total_blocks // 3:
                # Early blocks: weighted for importance
                schedule.append("weighted")
            elif block_idx < 2 * total_blocks // 3:
                # Middle blocks: random for diversity
                schedule.append("random")
            else:
                # Late blocks: adaptive for optimization
                schedule.append("adaptive")
        
        logger.debug(f"Created resampling schedule: {schedule}")
        return schedule
    
    def _get_resampling_strategy_for_block(self, block_idx: int) -> str:
        """
        Get the appropriate resampling strategy for a specific block.
        
        Args:
            block_idx: Current block index
            
        Returns:
            Resampling strategy for this block
        """
        # Initialize resampling schedule if not exists
        if not hasattr(self, '_resampling_schedule'):
            # Create default schedule for reasonable number of blocks
            total_blocks = getattr(self.config, 'total_blocks', 12)  # Default 12 blocks
            self._initialize_resampling_schedule(total_blocks)
        
        # Use pre-computed schedule
        if block_idx < len(self._resampling_schedule):
            strategy = self._resampling_schedule[block_idx]
            logger.debug(f"Using resampling strategy '{strategy}' for block {block_idx}")
            return strategy
        else:
            # Fallback to adaptive for blocks beyond schedule
            logger.debug(f"Block {block_idx} beyond schedule, using adaptive strategy")
            return "adaptive"
    
    def _initialize_resampling_schedule(self, total_blocks: int):
        """
        Initialize the resampling schedule for progressive training.
        
        Args:
            total_blocks: Total number of blocks to train
        """
        self._resampling_schedule = self._create_resampling_schedule(total_blocks)
        logger.info(f"Initialized resampling schedule for {total_blocks} blocks: {self._resampling_schedule}")
    
    def _create_masks_at_current_depth(self, batch: Dict[str, torch.Tensor], 
                                      block_idx: int) -> torch.Tensor:
        """
        Create masks at current block depth using time-step-based masking.
        
        Args:
            batch: Input batch
            block_idx: Current block index (determines mask fraction)
            
        Returns:
            Mask tensor for current depth
        """
        if self.masking_strategy is not None:
            # Use the masking strategy to generate masks
            masks = self.masking_strategy.generate_masks(
                batch["input_ids"], 
                block_idx=block_idx,
                time_t=block_idx  # Use block index as time step
            )
        else:
            # Fallback: create simple progressive masking
            batch_size, seq_len = batch["input_ids"].shape
            device = batch["input_ids"].device
            
            # Progressive masking: more masking in later blocks
            base_mask_fraction = 0.15  # 15% base masking
            progressive_factor = min(block_idx * 0.1, 0.75)  # Up to 75% max
            mask_fraction = base_mask_fraction + progressive_factor
            
            # Create random masks
            masks = torch.rand(batch_size, seq_len, device=device) < mask_fraction
            
        logger.debug(f"Created masks at depth {block_idx} with {masks.sum().item()} masked positions")
        return masks
    
    def _forward_through_backbone(self, batch: Dict[str, torch.Tensor], 
                                 backbone_blocks: List[List[torch.nn.Module]], 
                                 masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frozen backbone to get activations.
        
        Args:
            batch: Input batch
            backbone_blocks: Frozen backbone blocks
            masks: Mask tensor
            
        Returns:
            Hidden states from backbone
        """
        # Apply masks to input
        masked_input = batch["input_ids"].clone()
        mask_token_id = getattr(self.config, 'mask_token_id', 0)  # Default to 0 if not set
        masked_input[masks] = mask_token_id
        
        # Forward through backbone blocks
        hidden_states = masked_input
        for block_idx, block in enumerate(backbone_blocks):
            for layer_idx, layer in enumerate(block):
                try:
                    hidden_states = layer(hidden_states)
                except Exception as e:
                    logger.error(f"Error in backbone block {block_idx}, layer {layer_idx}: {e}")
                    raise
        
        logger.debug(f"Forward pass through backbone completed: {hidden_states.shape}")
        return hidden_states
    
    def _store_activations_for_training(self, hidden_states: torch.Tensor, 
                                       masks: torch.Tensor, block_idx: int):
        """
        Store activations for current block training with time-step-based organization.
        
        Args:
            hidden_states: Activations from backbone
            masks: Mask tensor
            block_idx: Current block index (used as time step)
        """
        # Initialize activation storage if not exists
        if not hasattr(self, '_activation_storage'):
            self._activation_storage = {}
        
        # Create time-step key for this block
        time_step = block_idx
        
        # Store activations with time-step and mask information
        activation_data = {
            'hidden_states': hidden_states.detach().cpu(),
            'masks': masks.detach().cpu(),
            'time_step': time_step,
            'block_idx': block_idx,
            'batch_size': hidden_states.shape[0],
            'sequence_length': hidden_states.shape[1],
            'hidden_dim': hidden_states.shape[2]
        }
        
        # Store by time step for efficient retrieval
        if time_step not in self._activation_storage:
            self._activation_storage[time_step] = []
        
        self._activation_storage[time_step].append(activation_data)
        
        logger.debug(f"Stored activations for block {block_idx} (time_step {time_step}): "
                    f"shape={hidden_states.shape}, masked_positions={masks.sum().item()}")
    
    def _get_activations_for_training(self, time_step: int) -> List[Dict]:
        """
        Retrieve stored activations for a specific time step.
        
        Args:
            time_step: Time step to retrieve activations for
            
        Returns:
            List of activation data for the time step
        """
        if not hasattr(self, '_activation_storage'):
            logger.warning("No activation storage found")
            return []
        
        if time_step not in self._activation_storage:
            logger.warning(f"No activations found for time step {time_step}")
            return []
        
        activations = self._activation_storage[time_step]
        logger.debug(f"Retrieved {len(activations)} activation batches for time step {time_step}")
        return activations
    
    def _clear_activation_storage(self, time_step: Optional[int] = None):
        """
        Clear activation storage for memory management.
        
        Args:
            time_step: Specific time step to clear (None for all)
        """
        if not hasattr(self, '_activation_storage'):
            return
        
        if time_step is None:
            # Clear all activations
            self._activation_storage.clear()
            logger.debug("Cleared all activation storage")
        else:
            # Clear specific time step
            if time_step in self._activation_storage:
                del self._activation_storage[time_step]
                logger.debug(f"Cleared activation storage for time step {time_step}")
    
    def _get_activation_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about activation storage.
        
        Returns:
            Dictionary with storage statistics
        """
        if not hasattr(self, '_activation_storage'):
            return {"total_time_steps": 0, "total_activations": 0}
        
        total_activations = sum(len(activations) for activations in self._activation_storage.values())
        total_time_steps = len(self._activation_storage)
        
        stats = {
            "total_time_steps": total_time_steps,
            "total_activations": total_activations,
            "time_steps": list(self._activation_storage.keys()),
            "activations_per_time_step": {
                ts: len(activations) for ts, activations in self._activation_storage.items()
            }
        }
        
        logger.debug(f"Activation storage stats: {stats}")
        return stats
    
    def _create_activation_dataloader(self, time_step: int, batch_size: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        Create a dataloader-like structure from stored activations.
        
        Args:
            time_step: Time step to create dataloader for
            batch_size: Batch size for the dataloader
            
        Returns:
            List of batches for training
        """
        activations = self._get_activations_for_training(time_step)
        
        if not activations:
            logger.warning(f"No activations found for time step {time_step}")
            return []
        
        # Create batches from stored activations
        batches = []
        for i in range(0, len(activations), batch_size):
            batch_activations = activations[i:i + batch_size]
            
            # Combine activations into a single batch
            if batch_activations:
                # Stack hidden states and masks
                hidden_states = torch.stack([act['hidden_states'] for act in batch_activations])
                masks = torch.stack([act['masks'] for act in batch_activations])
                
                # Create batch dictionary
                batch = {
                    'hidden_states': hidden_states,
                    'masks': masks,
                    'time_step': time_step,
                    'batch_size': len(batch_activations)
                }
                
                batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches from {len(activations)} activations for time step {time_step}")
        return batches
    
    def _load_backbone_in_precision(self, blocks: List[List[torch.nn.Module]], 
                                   precision: str = "fp16") -> List[List[torch.nn.Module]]:
        """
        Load backbone blocks in specified precision (internal method).
        
        Args:
            blocks: Backbone blocks to load
            precision: Target precision ("nf_fp4", "fp8", "fp16", "fp32")
            
        Returns:
            Blocks loaded in specified precision
        """
        # TODO: Implement precision loading
        # This should handle:
        # - NVFP4 FP4: NVIDIA FP4 format
        # - FP8: 8-bit floating point
        # - FP16: Half precision
        # - FP32: Full precision
        
        if precision == "nf_fp4":
            # TODO: Implement NF FP4 loading
            logger.warning("NF FP4 loading not implemented yet - using FP16 fallback")
            for block in blocks:
                for layer in block:
                    layer.half()  # Fallback to FP16
        elif precision == "fp8":
            # TODO: Implement FP8 loading
            logger.warning("FP8 loading not implemented yet - using FP16 fallback")
            for block in blocks:
                for layer in block:
                    layer.half()  # Fallback to FP16
        elif precision == "fp16":
            # Load in half precision (supported by PyTorch)
            for block in blocks:
                for layer in block:
                    layer.half()
        elif precision == "fp32":
            # Keep in full precision
            pass
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        
        return blocks
    
    def _get_or_apply_qlora_adapters(self, blocks: List[List[torch.nn.Module]]) -> List[List[torch.nn.Module]]:
        """
        Get cached QLoRA adapters or apply and cache them.
        
        Args:
            blocks: Quantized backbone blocks
            
        Returns:
            Blocks with cached or newly applied QLoRA adapters
        """
        # Create cache key based on block structure
        cache_key = f"qlora_{id(blocks)}"
        
        # Check if QLoRA adapters are already cached
        if hasattr(self, '_qlora_cache') and cache_key in self._qlora_cache:
            logger.debug("Using cached QLoRA adapters")
            return self._qlora_cache[cache_key]
        
        # Apply and cache QLoRA adapters
        logger.info("Applying and caching QLoRA adapters to backbone")
        blocks_with_qlora = self._apply_qlora_adapters(blocks)
        
        # Cache the QLoRA adapters
        if not hasattr(self, '_qlora_cache'):
            self._qlora_cache = {}
        self._qlora_cache[cache_key] = blocks_with_qlora
        
        return blocks_with_qlora
    
    def _apply_qlora_adapters(self, blocks: List[List[torch.nn.Module]]) -> List[List[torch.nn.Module]]:
        """
        Apply QLoRA adapters to quantized backbone blocks (internal method).
        
        Args:
            blocks: Quantized backbone blocks
            
        Returns:
            Blocks with QLoRA adapters
        """
        logger.info("Applying QLoRA adapters to quantized backbone")
        
        for block_idx, block in enumerate(blocks):
            for layer_idx, layer in enumerate(block):
                # Apply QLoRA to linear layers (most common case)
                if hasattr(layer, 'weight') and len(layer.weight.shape) == 2:
                    # Add QLoRA adapters to this layer
                    layer = self._add_qlora_to_layer(layer, block_idx, layer_idx)
                    block[layer_idx] = layer
                elif hasattr(layer, 'linear') and hasattr(layer.linear, 'weight'):
                    # Handle wrapped linear layers (e.g., in attention modules)
                    layer.linear = self._add_qlora_to_layer(layer.linear, block_idx, layer_idx)
        
        logger.debug("QLoRA adapters applied to quantized backbone")
        return blocks
    
    def _add_qlora_to_layer(self, layer: torch.nn.Module, block_idx: int, layer_idx: int) -> torch.nn.Module:
        """
        Add QLoRA adapters to a specific layer.
        
        Args:
            layer: Layer to add QLoRA adapters to
            block_idx: Block index for naming
            layer_idx: Layer index for naming
            
        Returns:
            Layer with QLoRA adapters
        """
        # Create QLoRA adapter wrapper
        qlora_adapter = QLoRAAdapter(
            original_layer=layer,
            rank=self.config.qlora_rank if hasattr(self.config, 'qlora_rank') else 16,
            alpha=self.config.qlora_alpha if hasattr(self.config, 'qlora_alpha') else 32,
            dropout=self.config.qlora_dropout if hasattr(self.config, 'qlora_dropout') else 0.1,
            name=f"block_{block_idx}_layer_{layer_idx}"
        )
        
        # Freeze the original layer parameters
        qlora_adapter.freeze_original_layer()
        
        logger.debug(f"Added QLoRA adapter to block {block_idx}, layer {layer_idx}")
        return qlora_adapter
    
    def _collect_qlora_parameters(self, blocks: List[List[torch.nn.Module]]) -> List[torch.nn.Parameter]:
        """
        Collect only QLoRA adaptation parameters from blocks.
        
        Args:
            blocks: Blocks with QLoRA adapters
            
        Returns:
            List of QLoRA parameters for training
        """
        qlora_params = []
        
        for block in blocks:
            for layer in block:
                if isinstance(layer, QLoRAAdapter):
                    # Get only the adaptation parameters (A and B matrices)
                    qlora_params.extend(layer.get_adaptation_parameters())
                elif hasattr(layer, 'linear') and isinstance(layer.linear, QLoRAAdapter):
                    # Handle wrapped linear layers
                    qlora_params.extend(layer.linear.get_adaptation_parameters())
        
        logger.debug(f"Collected {len(qlora_params)} QLoRA parameters for training")
        return qlora_params
    
    def _clear_quantization_cache(self):
        """
        Clear cached quantized backbones, QLoRA adapters, and trained blocks.
        Useful for memory management or when switching precision.
        """
        if hasattr(self, '_quantized_backbone_cache'):
            self._quantized_backbone_cache.clear()
            logger.debug("Cleared quantized backbone cache")
        
        if hasattr(self, '_qlora_cache'):
            self._qlora_cache.clear()
            logger.debug("Cleared QLoRA cache")
        
        if hasattr(self, '_trained_blocks_cache'):
            self._trained_blocks_cache.clear()
            logger.debug("Cleared trained blocks cache")
    
    def _cache_fusion_activations(self, block_idx: int, blocks: List[List[torch.nn.Module]]):
        """
        Cache activations for fusion training.
        
        Args:
            block_idx: Current block index
            blocks: All blocks to cache
        """
        # Cache activations for fusion training
        self.cache_manager.cache_fusion_activations(block_idx, blocks)
        logger.debug(f"Cached fusion activations for block {block_idx}")
