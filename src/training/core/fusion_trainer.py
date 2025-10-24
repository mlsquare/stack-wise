"""
Fusion-specific training logic for progressive training strategies.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class FusionTrainer:
    """
    Fusion-specific training logic for progressive training strategies.
    
    Handles:
    - Progressive fusion of blocks
    - Frozen vs trainable block strategies
    - Quantized backbone with QLoRA adapters
    """
    
    def __init__(self, config, quantization_manager, cache_manager, lexical_kernel_manager=None):
        """
        Initialize fusion trainer.
        
        Args:
            config: Training configuration
            quantization_manager: Quantization manager instance
            cache_manager: Cache manager instance
            lexical_kernel_manager: Lexical kernel manager for LM head
        """
        self.config = config
        self.quantization_manager = quantization_manager
        self.cache_manager = cache_manager
        self.lexical_kernel_manager = lexical_kernel_manager
        
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
        optimizer = self._setup_fusion_optimizer([current_block])
        
        # Train current block
        self._train_current_block(block_idx, dataloader, current_block, optimizer)
        
        # Cache activations only for frozen previous blocks (not current block)
        # Current block is training so its activations change
        if block_idx > 0:  # Only cache if there are previous frozen blocks
            frozen_blocks = all_blocks[:block_idx]  # Previous frozen blocks only
            self._cache_fusion_activations(block_idx, frozen_blocks)
        else:
            logger.debug("No previous blocks to cache in frozen backbone mode")
        
        # Memory management: Convert trained current block to low precision
        self._convert_trained_blocks_to_low_precision(block_idx, [current_block])
    
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
        
        # Setup optimizer for all trainable blocks
        optimizer = self._setup_fusion_optimizer(trainable_blocks)
        
        # Train all blocks together with quantized backbone + QLoRA
        self._train_all_blocks_together(block_idx, dataloader, trainable_blocks, optimizer,
                                       backbone_precision="fp16", qlora_enabled=True)
        
        # Cache activations only for frozen backbone mode
        # In trainable mode, all blocks are updated so caching is not useful
        if self.config.fusion_mode == "frozen":
            self._cache_fusion_activations(block_idx, trainable_blocks)
        else:
            logger.debug("Skipping activation caching for trainable backbone mode")
        
        # Memory management: Convert trained blocks to low precision and remove from memory
        self._convert_trained_blocks_to_low_precision(block_idx, trainable_blocks)
    
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
                                                                           backbone_precision, qlora_enabled)
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
        
        # Generate masks for current block
        masks = self.masking_strategy.generate_masks(
            batch["input_ids"], 
            block_idx=block_idx,
            time_t=None  # Use default time step for fusion training
        )
        
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
                                                        backbone_precision: str = "fp16",
                                                        qlora_enabled: bool = False) -> float:
        """
        Train batch with all blocks using quantized backbone + optional QLoRA for each block.
        The current block is trained in full precision.
        
        Args:
            batch: Training batch
            trainable_blocks: All trainable blocks
            optimizer: Optimizer
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
        
        # Step 2: Resample fresh inputs from dataloader
        resampled_batch = self._resample_fresh_inputs(batch)
        
        # Step 3: Create masks at current depth (time-step-based)
        masks = self._create_masks_at_current_depth(resampled_batch, block_idx)
        
        # Step 4: Forward pass through backbone to get activations
        hidden_states = self._forward_through_backbone(resampled_batch, backbone_blocks, masks)
        
        # Step 5: Store activations for current block training
        self._store_activations_for_training(hidden_states, masks, block_idx)
        
        # Step 6: Forward through current block (full precision)
        for layer in current_block:
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
    
    def _setup_fusion_optimizer(self, trainable_blocks: List[List[torch.nn.Module]]) -> torch.optim.Optimizer:
        """
        Setup optimizer for fusion training.
        
        Args:
            trainable_blocks: All trainable blocks
            
        Returns:
            Optimizer instance
        """
        # Collect parameters from all trainable blocks
        parameters = []
        for block in trainable_blocks:
            for layer in block:
                parameters.extend(layer.parameters())
        
        # Create optimizer
        optimizer = torch.optim.AdamW(parameters, lr=self.config.learning_rate)
        
        logger.debug(f"Setup fusion optimizer for {len(parameters)} parameters")
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
    
    def _convert_trained_blocks_to_low_precision(self, block_idx: int, 
                                                trained_blocks: List[List[torch.nn.Module]],
                                                cache_precision: str = "fp8"):
        """
        Convert trained blocks to low precision and remove full-precision versions from memory.
        
        This is critical for memory management - after training, we convert to low precision
        and remove the full-precision weights to prevent memory accumulation.
        
        Args:
            block_idx: Current block index
            trained_blocks: Blocks trained in full precision
            cache_precision: Target precision for caching ("nf_fp4", "fp8")
        """
        logger.info(f"Converting trained blocks to {cache_precision} precision for memory efficiency")
        
        # Cache the trained blocks in low precision
        cached_blocks = self._cache_trained_block_in_low_precision(
            trained_blocks, cache_precision=cache_precision
        )
        
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
        
        logger.debug(f"Converted block {block_idx} to {cache_precision} precision and freed gradients")
    
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
    
    def _resample_fresh_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Resample fresh inputs from dataloader for unified training approach.
        
        Args:
            batch: Original batch from dataloader
            
        Returns:
            Resampled batch with fresh inputs
        """
        # For now, return the original batch
        # TODO: Implement proper resampling logic
        logger.debug("Resampling fresh inputs from dataloader")
        return batch
    
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
        # Generate masks based on current block depth
        masks = self.masking_strategy.generate_masks(
            batch["input_ids"], 
            block_idx=block_idx,
            time_t=block_idx  # Use block index as time step
        )
        
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
        masked_input[masks] = self.config.mask_token_id
        
        # Forward through backbone blocks
        hidden_states = masked_input
        for block in backbone_blocks:
            for layer in block:
                hidden_states = layer(hidden_states)
        
        logger.debug(f"Forward pass through backbone completed")
        return hidden_states
    
    def _store_activations_for_training(self, hidden_states: torch.Tensor, 
                                       masks: torch.Tensor, block_idx: int):
        """
        Store activations for current block training.
        
        Args:
            hidden_states: Activations from backbone
            masks: Mask tensor
            block_idx: Current block index
        """
        # TODO: Implement activation storage for training
        # This should store activations with mask information for current block
        logger.debug(f"Storing activations for block {block_idx} training")
    
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
        # TODO: Implement QLoRA adapter application
        # This should:
        # - Add low-rank adapters to quantized blocks
        # - Enable efficient fine-tuning
        # - Maintain memory efficiency
        
        logger.warning("QLoRA adapter implementation not complete yet")
        
        return blocks
    
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
