"""
Unified trainer supporting all training modes with modular components.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..strategies.masking import TimeStepMasking, ProgressiveMasking
from ..strategies.quantization import QLoRAManager, QuantizationManager
from ..strategies.caching import TimeStepCache, ActivationCache
from ..utils.config_validator import ConfigValidator
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.metrics import TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration for unified trainer."""
    
    # Basic training
    mode: str = "layerwise"  # layerwise, blockwise, fused
    block_size: int = 1
    fusion_mode: str = "frozen"  # frozen, trainable
    
    # Time-step masking
    time_step_masking: bool = True
    num_time_steps: int = 10
    time_step_bins: List[int] = field(default_factory=lambda: list(range(10)))
    time_step_mask_fractions: Dict[int, float] = field(
        default_factory=lambda: {0: 0.15, 5: 0.50, 9: 0.90}
    )
    store_all_time_steps: bool = False
    time_step_cache_size: int = 1
    
    # Quantization
    quantization_enabled: bool = True
    quantization_type: str = "nf_fp8"  # nf_fp8, fp16, fp32
    load_quantized: bool = True
    mixed_precision: bool = True
    backbone_quantized: bool = True
    adapters_full_precision: bool = True
    
    # QLoRA
    qlora_enabled: bool = True
    qlora_rank: int = 16
    qlora_alpha: int = 32
    qlora_dropout: float = 0.1
    
    # Caching
    cache_mode: str = "layerwise"
    cache_dir: str = "./cache"
    fusion_evaluation: bool = False
    save_fused_checkpoints: bool = False
    
    # Training parameters
    epochs_per_block: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 4
    seq_len: int = 512


class UnifiedTrainer:
    """
    Unified trainer supporting all training modes with modular components.
    
    This trainer provides a single interface for:
    - Layer-wise training (block_size=1)
    - Block-wise training (block_size=4)
    - Fused training with quantization and QLoRA adapters
    - Time-step-based masking strategies
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the unified trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.block_size = config.block_size
        self.training_mode = config.mode
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self.masking_strategy = self._init_masking_strategy()
        self.quantization_manager = self._init_quantization_manager()
        self.cache_manager = self._init_cache_manager()
        self.block_trainer = self._init_block_trainer()
        self.checkpoint_manager = self._init_checkpoint_manager()
        self.metrics = self._init_metrics()
        
        logger.info(f"Initialized UnifiedTrainer with mode: {self.training_mode}, block_size: {self.block_size}")
    
    def _validate_config(self):
        """Validate training configuration."""
        validator = ConfigValidator()
        validator.validate(self.config)
    
    def _init_masking_strategy(self):
        """Initialize masking strategy based on configuration."""
        if self.config.time_step_masking:
            return TimeStepMasking(self.config)
        else:
            return ProgressiveMasking(self.config)
    
    def _init_quantization_manager(self):
        """Initialize quantization manager."""
        return QuantizationManager(self.config)
    
    def _init_cache_manager(self):
        """Initialize cache manager."""
        if self.config.time_step_masking:
            return TimeStepCache(self.config)
        else:
            return ActivationCache(self.config)
    
    def _init_block_trainer(self):
        """Initialize block trainer."""
        from .block_trainer import BlockTrainer
        return BlockTrainer(
            config=self.config,
            masking_strategy=self.masking_strategy,
            quantization_manager=self.quantization_manager,
            cache_manager=self.cache_manager
        )
    
    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager."""
        return CheckpointManager(self.config)
    
    def _init_metrics(self):
        """Initialize training metrics."""
        return TrainingMetrics()
    
    def train_all_layers(self, dataloader: DataLoader, model_layers: List[torch.nn.Module]):
        """
        Main training entry point for all layers.
        
        Args:
            dataloader: Data loader for training data
            model_layers: List of layer modules to train
        """
        logger.info(f"Starting {self.training_mode} training for {len(model_layers)} layers")
        
        # Setup training environment
        self._setup_training_environment()
        
        # Create blocks from layers
        blocks = self._create_blocks(model_layers)
        logger.info(f"Created {len(blocks)} blocks with size {self.block_size}")
        
        # Train each block
        for block_idx, block_layers in enumerate(blocks):
            logger.info(f"Training block {block_idx}/{len(blocks)-1}")
            self._train_block(block_idx, dataloader, block_layers)
        
        # Finalize training
        self._finalize_training()
        
        logger.info("Completed training all layers")
    
    def _setup_training_environment(self):
        """Setup training environment and components."""
        # Initialize cache directory
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup quantization if enabled
        if self.config.quantization_enabled:
            self.quantization_manager.setup_quantization()
        
        # Setup caching
        self.cache_manager.setup_cache()
        
        logger.info("Training environment setup complete")
    
    def _create_blocks(self, model_layers: List[torch.nn.Module]) -> List[List[torch.nn.Module]]:
        """
        Create blocks from model layers.
        
        Args:
            model_layers: List of layer modules
            
        Returns:
            List of blocks, where each block contains block_size layers
        """
        blocks = []
        for i in range(0, len(model_layers), self.block_size):
            block = model_layers[i:i + self.block_size]
            blocks.append(block)
        
        return blocks
    
    def _train_block(self, block_idx: int, dataloader: DataLoader, block_layers: List[torch.nn.Module]):
        """
        Train a single block.
        
        Args:
            block_idx: Index of the block to train
            dataloader: Data loader for training data
            block_layers: Layers in the current block
        """
        # Setup block (quantization, QLoRA, etc.)
        self._setup_block(block_idx, block_layers)
        
        # Train block using block trainer
        self.block_trainer.train_block(block_idx, dataloader, block_layers)
        
        # Cache activations for next block
        self._cache_block_activations(block_idx, block_layers)
    
    def _setup_block(self, block_idx: int, block_layers: List[torch.nn.Module]):
        """Setup block with quantization and adapters."""
        # Add QLoRA adapters if enabled
        if self.config.qlora_enabled:
            self.quantization_manager.add_qlora_adapters(block_layers)
        
        # Setup mixed precision if enabled
        if self.config.mixed_precision:
            self.quantization_manager.setup_mixed_precision(block_layers)
        
        logger.debug(f"Setup complete for block {block_idx}")
    
    def _cache_block_activations(self, block_idx: int, block_layers: List[torch.nn.Module]):
        """Cache activations for the current block."""
        self.cache_manager.cache_block_activations(block_idx, block_layers)
        logger.debug(f"Cached activations for block {block_idx}")
    
    def _finalize_training(self):
        """Finalize training and cleanup."""
        # Save final checkpoints
        self.checkpoint_manager.save_final_checkpoints()
        
        # Cleanup cache if needed
        self.cache_manager.cleanup()
        
        # Log final metrics
        self.metrics.log_final_metrics()
        
        logger.info("Training finalized")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        return {
            "training_mode": self.training_mode,
            "block_size": self.block_size,
            "time_step_masking": self.config.time_step_masking,
            "quantization_enabled": self.config.quantization_enabled,
            "qlora_enabled": self.config.qlora_enabled,
            "cache_mode": self.config.cache_mode,
            "metrics": self.metrics.get_metrics()
        }
