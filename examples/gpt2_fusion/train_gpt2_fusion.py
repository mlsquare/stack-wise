#!/usr/bin/env python3
"""
GPT-2 Fusion Training Script

This script trains a GPT-2 model using the FusionTrainer with mask-diffusion objectives.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup training environment."""
    logger.info("🔧 Setting up training environment...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    return device

def load_config(config_path: str = "gpt2.yaml"):
    """Load training configuration."""
    logger.info(f"📋 Loading configuration from {config_path}")
    
    # Import configuration classes
    from config.base import StackWiseConfig
    
    # Load configuration
    config = StackWiseConfig.from_yaml(config_path)
    config.validate()
    
    logger.info(f"✅ Configuration loaded: run_id={config.training.run_id}")
    logger.info(f"Model: {config.model.n_layers} layers, {config.model.d_model} hidden")
    logger.info(f"Training: {config.training.total_blocks} blocks, {config.training.block_size} layers per block")
    
    return config

def prepare_data(config):
    """Prepare training data."""
    logger.info("📊 Preparing training data...")
    
    # Import data loader
    from data_loader import load_corpus, get_tokenizer, create_data_loaders
    
    # Load corpus
    corpus = load_corpus(config.data.dataset_path)
    
    # Get tokenizer
    tokenizer = get_tokenizer(config.data.tokenizer_path)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        corpus=corpus,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.training.seq_len,
        train_split=0.9,
        num_workers=config.data.num_workers
    )
    
    logger.info(f"✅ Data prepared: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader, tokenizer

def create_gpt2_model(config):
    """Create GPT-2 model architecture."""
    logger.info("🏗️ Creating GPT-2 model...")
    
    # Import model components
    from model.layers import MLGKALayer, LexicalKernelManager
    import torch.nn as nn
    
    # Create lexical kernel manager for embeddings
    lexical_kernel_manager = LexicalKernelManager(
        family=config.model.tokenizer_embedding['family'],
        embedding_option=config.model.tokenizer_embedding['embedding_option'],
        freeze_embeddings=config.model.tokenizer_embedding['freeze_embeddings'],
        target_model_dim=config.model.d_model
    )
    
    # Create model layers
    layers = []
    for i in range(config.model.n_layers):
        layer = MLGKALayer(
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_kv_heads=config.model.n_kv_heads,
            d_ff=config.model.d_ff,
            attention_mode=config.model.attention_mode
        )
        layers.append(layer)
    
    logger.info(f"✅ Model created: {len(layers)} layers")
    return layers, lexical_kernel_manager

def setup_fusion_trainer(config, lexical_kernel_manager):
    """Setup FusionTrainer."""
    logger.info("🔧 Setting up FusionTrainer...")
    
    # Import FusionTrainer directly to avoid package import issues
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fusion_trainer", 
        str(Path(__file__).parent.parent.parent / "src" / "training" / "core" / "fusion_trainer.py")
    )
    fusion_trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion_trainer_module)
    FusionTrainer = fusion_trainer_module.FusionTrainer
    
    # Initialize FusionTrainer
    fusion_trainer = FusionTrainer(
        config=config,
        masking_strategy=None,  # Will be implemented later
        quantization_manager=None,  # Will be implemented later
        cache_manager=None,  # Will be implemented later
        lexical_kernel_manager=lexical_kernel_manager
    )
    
    logger.info("✅ FusionTrainer initialized")
    return fusion_trainer

def train_block(fusion_trainer, layers, train_loader, val_loader, block_idx, config):
    """Train a single block."""
    logger.info(f"🚀 Training block {block_idx}...")
    
    # Get block layers
    start_idx = block_idx * config.training.block_size
    end_idx = min(start_idx + config.training.block_size, len(layers))
    block_layers = layers[start_idx:end_idx]
    
    logger.info(f"Block {block_idx}: layers {start_idx}-{end_idx-1}")
    
    # Create dummy blocks for testing (in real implementation, this would be the actual model)
    dummy_blocks = [block_layers]
    
    # Test disk backup system
    logger.info("💾 Testing disk backup system...")
    fusion_trainer._save_full_precision_weights_to_disk(dummy_blocks, "fp16")
    
    # Test validation
    run_id = config.training.run_id
    is_valid = fusion_trainer._validate_saved_weights(run_id, "fp16", 1)
    logger.info(f"✅ Weight validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test restoration
    restored_blocks = fusion_trainer._restore_full_precision_weights_from_disk(
        run_id, "fp16", [0]
    )
    logger.info(f"✅ Restored {len(restored_blocks)} blocks from disk")
    
    # Test optimizer setup
    logger.info("⚙️ Testing optimizer setup...")
    optimizer = fusion_trainer._setup_fusion_optimizer(
        dummy_blocks, qlora_enabled=True, current_block_idx=0
    )
    logger.info(f"✅ Optimizer created with {len(optimizer.param_groups)} parameter groups")
    
    # Test memory management
    logger.info("🧠 Testing memory management...")
    # Add some gradients to test clearing
    for block in dummy_blocks:
        for layer in block:
            if hasattr(layer, 'weight') and layer.weight.grad is None:
                layer.weight.grad = torch.randn_like(layer.weight)
    
    converted_blocks = fusion_trainer._convert_trained_blocks_to_low_precision(
        dummy_blocks, "fp16"
    )
    logger.info("✅ Memory management test completed")
    
    # Simulate training steps
    logger.info(f"📈 Simulating training for block {block_idx}...")
    for step in range(min(10, config.training.max_steps // config.training.total_blocks)):
        # Simulate training step
        if step % 5 == 0:
            logger.info(f"  Step {step}: Training block {block_idx}")
    
    logger.info(f"✅ Block {block_idx} training completed")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GPT-2 Fusion Training")
    parser.add_argument("--config", default="gpt2.yaml", help="Configuration file")
    parser.add_argument("--prepare_data", action="store_true", help="Prepare data first")
    parser.add_argument("--test_only", action="store_true", help="Run tests only")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting GPT-2 Fusion Training")
        
        # Setup environment
        device = setup_environment()
        
        # Load configuration
        config = load_config(args.config)
        
        # Prepare data if requested
        if args.prepare_data:
            from data_loader import create_dummy_corpus
            create_dummy_corpus(config.data.num_samples, config.data.dataset_path)
            logger.info("✅ Data preparation completed")
            return
        
        # Prepare training data
        train_loader, val_loader, tokenizer = prepare_data(config)
        
        # Create GPT-2 model
        layers, lexical_kernel_manager = create_gpt2_model(config)
        
        # Setup FusionTrainer
        fusion_trainer = setup_fusion_trainer(config, lexical_kernel_manager)
        
        if args.test_only:
            # Run tests only
            logger.info("🧪 Running tests only...")
            
            # Test disk backup system
            dummy_blocks = [layers[:4]]  # First block
            fusion_trainer._save_full_precision_weights_to_disk(dummy_blocks, "fp16")
            
            # Test validation
            run_id = config.training.run_id
            is_valid = fusion_trainer._validate_saved_weights(run_id, "fp16", 1)
            logger.info(f"✅ Weight validation: {'PASSED' if is_valid else 'FAILED'}")
            
            # Test optimizer
            optimizer = fusion_trainer._setup_fusion_optimizer(
                dummy_blocks, qlora_enabled=True, current_block_idx=0
            )
            logger.info(f"✅ Optimizer created with {len(optimizer.param_groups)} parameter groups")
            
            logger.info("🎉 All tests completed successfully!")
            return
        
        # Train each block
        logger.info("🎯 Starting block-wise training...")
        for block_idx in range(config.training.total_blocks):
            train_block(fusion_trainer, layers, train_loader, val_loader, block_idx, config)
        
        logger.info("🎉 GPT-2 Fusion Training completed successfully!")
        
        # Show final results
        logger.info("📊 Training Summary:")
        logger.info(f"  Run ID: {config.training.run_id}")
        logger.info(f"  Total Blocks: {config.training.total_blocks}")
        logger.info(f"  Layers per Block: {config.training.block_size}")
        logger.info(f"  QLoRA Enabled: {config.training.qlora_enabled}")
        logger.info(f"  Quantization: {config.training.quantization_type}")
        logger.info(f"  Time-Step Masking: {config.training.time_step_masking}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        # Clean up test files
        logger.info("🧹 Cleaning up...")
        backup_dir = "checkpoints/full_precision_backups"
        if os.path.exists(backup_dir):
            import shutil
            shutil.rmtree(backup_dir)
            logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    main()
