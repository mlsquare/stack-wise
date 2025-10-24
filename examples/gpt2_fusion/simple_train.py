#!/usr/bin/env python3
"""
Simple GPT-2 Fusion Training Example

This script demonstrates the core functionality without complex imports.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating GPT-2 fusion training concepts."""
    logger.info("🚀 Starting Simple GPT-2 Fusion Training Demo")
    
    try:
        # 1. Load configuration
        logger.info("📋 Loading configuration...")
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from config.base import StackWiseConfig
        config = StackWiseConfig.from_yaml("gpt2.yaml")
        config.validate()
        logger.info(f"✅ Configuration loaded: run_id={config.training.run_id}")
        
        # 2. Test data loading
        logger.info("📊 Testing data loading...")
        from data_loader import load_corpus, get_tokenizer, create_data_loaders
        
        corpus = load_corpus(config.data.dataset_path)
        tokenizer = get_tokenizer(config.data.tokenizer_path)
        train_loader, val_loader = create_data_loaders(
            corpus=corpus,
            tokenizer=tokenizer,
            batch_size=config.training.batch_size,
            max_length=config.training.seq_len,
            train_split=0.9,
            num_workers=config.data.num_workers
        )
        logger.info(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # 3. Test FusionTrainer
        logger.info("🔧 Testing FusionTrainer...")
        
        # Import FusionTrainer directly
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
            masking_strategy=None,
            quantization_manager=None,
            cache_manager=None,
            lexical_kernel_manager=None
        )
        logger.info("✅ FusionTrainer initialized")
        
        # 4. Test disk backup system
        logger.info("💾 Testing disk backup system...")
        
        # Create dummy blocks for testing
        dummy_blocks = []
        for block_idx in range(config.training.total_blocks):
            block = []
            for layer_idx in range(config.training.block_size):
                # Create a simple linear layer
                layer = torch.nn.Linear(128, 128)
                block.append(layer)
            dummy_blocks.append(block)
        
        # Test saving weights to disk
        fusion_trainer._save_full_precision_weights_to_disk(dummy_blocks, "fp16")
        
        # Test validation
        run_id = config.training.run_id
        is_valid = fusion_trainer._validate_saved_weights(run_id, "fp16", len(dummy_blocks))
        logger.info(f"✅ Weight validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test restoration
        restored_blocks = fusion_trainer._restore_full_precision_weights_from_disk(
            run_id, "fp16", list(range(len(dummy_blocks)))
        )
        logger.info(f"✅ Restored {len(restored_blocks)} blocks from disk")
        
        # Test model reconstruction
        reconstructed = fusion_trainer._reconstruct_model_from_disk(run_id, "fp16")
        logger.info(f"✅ Reconstructed {len(reconstructed)} blocks from disk")
        
        # 5. Test optimizer setup
        logger.info("⚙️ Testing optimizer setup...")
        optimizer = fusion_trainer._setup_fusion_optimizer(
            dummy_blocks, qlora_enabled=True, current_block_idx=1
        )
        logger.info(f"✅ Optimizer created with {len(optimizer.param_groups)} parameter groups")
        
        # 6. Test memory management
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
        
        # 7. Show configuration summary
        logger.info("📊 Configuration Summary:")
        logger.info(f"  Run ID: {config.training.run_id}")
        logger.info(f"  Total Blocks: {config.training.total_blocks}")
        logger.info(f"  Layers per Block: {config.training.block_size}")
        logger.info(f"  QLoRA Enabled: {config.training.qlora_enabled}")
        logger.info(f"  Quantization Type: {config.training.quantization_type}")
        logger.info(f"  Time-Step Masking: {config.training.time_step_masking}")
        logger.info(f"  Number of Time Steps: {config.training.num_time_steps}")
        
        logger.info("🎉 Simple GPT-2 Fusion Training Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
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
