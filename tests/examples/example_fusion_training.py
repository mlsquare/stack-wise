#!/usr/bin/env python3
"""
Example script demonstrating FusionTrainer usage.

This script shows how to use the FusionTrainer for layer-wise training
with mask-diffusion objectives, quantization, and QLoRA adapters.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate FusionTrainer usage."""
    logger.info("🚀 Starting FusionTrainer Example")
    
    try:
        # Import required modules
        from config.base import StackWiseConfig
        
        # Import FusionTrainer directly to avoid package import issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fusion_trainer", 
            "src/training/core/fusion_trainer.py"
        )
        fusion_trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fusion_trainer_module)
        FusionTrainer = fusion_trainer_module.FusionTrainer
        
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = StackWiseConfig.from_yaml("config.yaml")
        config.validate()
        logger.info(f"✅ Configuration loaded: run_id={config.training.run_id}")
        
        # Initialize FusionTrainer
        logger.info("🔧 Initializing FusionTrainer...")
        fusion_trainer = FusionTrainer(
            config=config,
            masking_strategy=None,  # Will be set by UnifiedTrainer
            quantization_manager=None,  # Will be set by UnifiedTrainer
            cache_manager=None,  # Will be set by UnifiedTrainer
            lexical_kernel_manager=None  # Will be set by UnifiedTrainer
        )
        logger.info("✅ FusionTrainer initialized")
        
        # Create example model blocks
        logger.info("🏗️ Creating example model blocks...")
        example_blocks = []
        for block_idx in range(config.training.total_blocks):
            block = []
            for layer_idx in range(2):  # 2 layers per block
                # Create a simple transformer-like layer
                layer = torch.nn.Sequential(
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128)
                )
                block.append(layer)
            example_blocks.append(block)
            logger.info(f"  Created block {block_idx} with {len(block)} layers")
        
        # Demonstrate disk backup system
        logger.info("💾 Testing disk backup system...")
        fusion_trainer._save_full_precision_weights_to_disk(example_blocks, "fp16")
        
        # Validate saved weights
        run_id = config.training.run_id
        is_valid = fusion_trainer._validate_saved_weights(run_id, "fp16", len(example_blocks))
        logger.info(f"✅ Weight validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test restoration
        restored_blocks = fusion_trainer._restore_full_precision_weights_from_disk(
            run_id, "fp16", list(range(len(example_blocks)))
        )
        logger.info(f"✅ Restored {len(restored_blocks)} blocks from disk")
        
        # Test model reconstruction
        reconstructed = fusion_trainer._reconstruct_model_from_disk(run_id, "fp16")
        logger.info(f"✅ Reconstructed {len(reconstructed)} blocks from disk")
        
        # Demonstrate optimizer setup
        logger.info("⚙️ Testing optimizer setup...")
        optimizer = fusion_trainer._setup_fusion_optimizer(
            example_blocks, qlora_enabled=True, current_block_idx=1
        )
        logger.info(f"✅ Optimizer created with {len(optimizer.param_groups)} parameter groups")
        
        # Demonstrate memory management
        logger.info("🧠 Testing memory management...")
        # Add some gradients to test clearing
        for block in example_blocks:
            for layer in block:
                if hasattr(layer, 'weight') and layer.weight.grad is None:
                    layer.weight.grad = torch.randn_like(layer.weight)
        
        converted_blocks = fusion_trainer._convert_trained_blocks_to_low_precision(
            example_blocks, "fp16"
        )
        logger.info("✅ Memory management test completed")
        
        # Show configuration details
        logger.info("📊 Configuration Summary:")
        logger.info(f"  Run ID: {config.training.run_id}")
        logger.info(f"  Total Blocks: {config.training.total_blocks}")
        logger.info(f"  QLoRA Enabled: {config.training.qlora_enabled}")
        logger.info(f"  Quantization Type: {config.training.quantization_type}")
        logger.info(f"  Time-Step Masking: {config.training.time_step_masking}")
        logger.info(f"  Number of Time Steps: {config.training.num_time_steps}")
        
        logger.info("🎉 FusionTrainer example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise
    
    finally:
        # Clean up test files
        logger.info("🧹 Cleaning up test files...")
        backup_dir = "checkpoints/full_precision_backups"
        if os.path.exists(backup_dir):
            import shutil
            shutil.rmtree(backup_dir)
            logger.info("✅ Test files cleaned up")

if __name__ == "__main__":
    main()
