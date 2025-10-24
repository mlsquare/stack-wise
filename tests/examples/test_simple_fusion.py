#!/usr/bin/env python3
"""
Simplified test script for FusionTrainer functionality.

This script tests the core functionality without requiring all strategy modules.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.base import StackWiseConfig
from training.core.fusion_trainer import FusionTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading and validation."""
    logger.info("Testing configuration loading...")
    
    try:
        # Load config from YAML
        config = StackWiseConfig.from_yaml("config.yaml")
        
        # Validate configuration
        config.validate()
        
        # Check new parameters
        assert hasattr(config.training, 'run_id'), "run_id not found in config"
        assert hasattr(config.training, 'qlora_lr'), "qlora_lr not found in config"
        assert hasattr(config.training, 'current_block_lr'), "current_block_lr not found in config"
        assert hasattr(config.training, 'quantization_type'), "quantization_type not found in config"
        assert hasattr(config.training, 'time_step_masking'), "time_step_masking not found in config"
        
        logger.info("✅ Configuration loading successful")
        return config
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        raise

def test_fusion_trainer_initialization():
    """Test FusionTrainer initialization."""
    logger.info("Testing FusionTrainer initialization...")
    
    try:
        config = StackWiseConfig.from_yaml("config.yaml")
        
        # Initialize FusionTrainer with minimal dependencies
        fusion_trainer = FusionTrainer(
            config=config,
            masking_strategy=None,
            quantization_manager=None,
            cache_manager=None,
            lexical_kernel_manager=None
        )
        
        # Check that FusionTrainer has required methods
        required_methods = [
            '_setup_fusion_optimizer',
            '_save_full_precision_weights_to_disk',
            '_validate_saved_weights',
            '_restore_full_precision_weights_from_disk',
            '_reconstruct_model_from_disk'
        ]
        
        for method_name in required_methods:
            assert hasattr(fusion_trainer, method_name), f"{method_name} not found"
        
        logger.info("✅ FusionTrainer initialization successful")
        return fusion_trainer
        
    except Exception as e:
        logger.error(f"❌ FusionTrainer initialization failed: {e}")
        raise

def test_disk_backup_system():
    """Test disk backup system functionality."""
    logger.info("Testing disk backup system...")
    
    try:
        config = StackWiseConfig.from_yaml("config.yaml")
        fusion_trainer = FusionTrainer(
            config=config,
            masking_strategy=None,
            quantization_manager=None,
            cache_manager=None,
            lexical_kernel_manager=None
        )
        
        # Create dummy blocks for testing
        dummy_blocks = []
        for block_idx in range(2):
            block = []
            for layer_idx in range(2):
                # Create a simple linear layer
                layer = torch.nn.Linear(128, 128)
                block.append(layer)
            dummy_blocks.append(block)
        
        # Test saving weights to disk
        fusion_trainer._save_full_precision_weights_to_disk(dummy_blocks, "fp16")
        
        # Test validation
        run_id = config.training.run_id
        is_valid = fusion_trainer._validate_saved_weights(run_id, "fp16", 2)
        assert is_valid, "Weight validation failed"
        
        # Test restoration
        restored_blocks = fusion_trainer._restore_full_precision_weights_from_disk(
            run_id, "fp16", [0, 1]
        )
        assert len(restored_blocks) == 2, "Failed to restore all blocks"
        
        # Test model reconstruction
        reconstructed = fusion_trainer._reconstruct_model_from_disk(run_id, "fp16")
        assert len(reconstructed) == 2, "Failed to reconstruct model"
        
        logger.info("✅ Disk backup system test successful")
        
    except Exception as e:
        logger.error(f"❌ Disk backup system test failed: {e}")
        raise

def test_memory_management():
    """Test memory management functionality."""
    logger.info("Testing memory management...")
    
    try:
        config = StackWiseConfig.from_yaml("config.yaml")
        fusion_trainer = FusionTrainer(
            config=config,
            masking_strategy=None,
            quantization_manager=None,
            cache_manager=None,
            lexical_kernel_manager=None
        )
        
        # Create dummy blocks
        dummy_blocks = []
        for block_idx in range(2):
            block = []
            for layer_idx in range(2):
                layer = torch.nn.Linear(128, 128)
                # Add some gradients to test clearing
                if layer.weight.grad is None:
                    layer.weight.grad = torch.randn_like(layer.weight)
                block.append(layer)
            dummy_blocks.append(block)
        
        # Test precision conversion (this will clear gradients)
        converted_blocks = fusion_trainer._convert_trained_blocks_to_low_precision(
            dummy_blocks, "fp16"
        )
        
        # Check that gradients are cleared
        for block in converted_blocks:
            for layer in block:
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    assert torch.allclose(layer.weight.grad, torch.zeros_like(layer.weight.grad)), \
                        "Gradients not properly cleared"
        
        logger.info("✅ Memory management test successful")
        
    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        raise

def test_optimizer_setup():
    """Test optimizer setup with parameter groups."""
    logger.info("Testing optimizer setup...")
    
    try:
        config = StackWiseConfig.from_yaml("config.yaml")
        fusion_trainer = FusionTrainer(
            config=config,
            masking_strategy=None,
            quantization_manager=None,
            cache_manager=None,
            lexical_kernel_manager=None
        )
        
        # Create dummy blocks
        dummy_blocks = []
        for block_idx in range(2):
            block = []
            for layer_idx in range(2):
                layer = torch.nn.Linear(128, 128)
                block.append(layer)
            dummy_blocks.append(block)
        
        # Test optimizer setup with QLoRA enabled
        optimizer = fusion_trainer._setup_fusion_optimizer(
            dummy_blocks, qlora_enabled=True, current_block_idx=1
        )
        
        # Check that optimizer has parameter groups
        assert hasattr(optimizer, 'param_groups'), "Optimizer missing param_groups"
        assert len(optimizer.param_groups) > 0, "No parameter groups found"
        
        logger.info("✅ Optimizer setup test successful")
        
    except Exception as e:
        logger.error(f"❌ Optimizer setup test failed: {e}")
        raise

def cleanup_test_files():
    """Clean up test files."""
    logger.info("Cleaning up test files...")
    
    try:
        # Remove test backup directory
        backup_dir = "checkpoints/full_precision_backups"
        if os.path.exists(backup_dir):
            import shutil
            shutil.rmtree(backup_dir)
            logger.info("✅ Test files cleaned up")
        
    except Exception as e:
        logger.warning(f"Failed to clean up test files: {e}")

def main():
    """Run all tests."""
    logger.info("Starting simplified FusionTrainer tests...")
    
    try:
        # Test 1: Configuration loading
        config = test_config_loading()
        
        # Test 2: FusionTrainer initialization
        fusion_trainer = test_fusion_trainer_initialization()
        
        # Test 3: Disk backup system
        test_disk_backup_system()
        
        # Test 4: Memory management
        test_memory_management()
        
        # Test 5: Optimizer setup
        test_optimizer_setup()
        
        logger.info("🎉 All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise
    
    finally:
        # Clean up test files
        cleanup_test_files()

if __name__ == "__main__":
    main()
