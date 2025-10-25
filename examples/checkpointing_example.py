#!/usr/bin/env python3
"""
Checkpointing Example for Stack-Wise Progressive Training

This example demonstrates:
- Progressive training with checkpointing
- Rack-level checkpointing and loading
- Stack-level checkpointing and loading
- Training resumption from checkpoints
- LoRA adapter checkpointing
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.base import StackWiseConfig
from src.training import ProgressiveTrainer, ProgressiveRackBuilder
from src.model.architecture import create_stack_from_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    
    def __init__(self, num_samples: int = 100, seq_len: int = 16, vocab_size: int = 1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input and target
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        target_ids = input_ids.clone()
        
        # Create some masking for MLM
        mask_prob = 0.15
        mask_indices = torch.rand(self.seq_len) < mask_prob
        target_ids[mask_indices] = -100  # Ignore in loss
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': torch.ones(self.seq_len)
        }


def create_sample_data(batch_size: int = 4, seq_len: int = 16, vocab_size: int = 1000):
    """Create sample dataloader"""
    dataset = DummyDataset(num_samples=50, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def test_progressive_checkpointing():
    """Test progressive training with checkpointing"""
    logger.info("=== Testing Progressive Checkpointing ===")
    
    # Create configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Create trainer and rack builder
    trainer = ProgressiveTrainer(config=config)
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Create sample data
    dataloader = create_sample_data(batch_size=2, seq_len=8, vocab_size=1000)
    
    # Add first stack
    logger.info("Adding first stack...")
    stack1 = rack_builder.append_stack(n_blocks=2, precision="full")
    
    # Train first stack with checkpointing
    logger.info("Training first stack...")
    results1 = trainer.train_rack(rack_builder, dataloader, target_stacks=1)
    
    # Save progressive checkpoint
    checkpoint_path = trainer.save_progressive_checkpoint(
        stack_idx=0,
        rack_builder=rack_builder,
        epoch=1,
        loss=results1.get('final_loss', 0.5)
    )
    logger.info(f"Saved progressive checkpoint: {checkpoint_path}")
    
    # Add second stack
    logger.info("Adding second stack...")
    stack2 = rack_builder.append_stack(n_blocks=2, precision="half")
    
    # Train second stack
    logger.info("Training second stack...")
    results2 = trainer.train_rack(rack_builder, dataloader, target_stacks=2)
    
    # Save rack checkpoint
    rack_checkpoint_path = trainer.save_rack_checkpoint(rack_builder)
    logger.info(f"Saved rack checkpoint: {rack_checkpoint_path}")
    
    # List all checkpoints
    checkpoints = trainer.list_checkpoints()
    logger.info(f"Available checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        logger.info(f"  - {cp['path']}: stack {cp['stack_idx']}, epoch {cp['epoch']}, loss {cp['loss']}")
    
    return checkpoint_path, rack_checkpoint_path


def test_rack_checkpointing():
    """Test rack-level checkpointing and loading"""
    logger.info("=== Testing Rack Checkpointing ===")
    
    # Create configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Create rack builder
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add stacks
    logger.info("Building rack...")
    stack1 = rack_builder.append_stack(n_blocks=2, precision="full")
    stack2 = rack_builder.append_stack(n_blocks=2, precision="half")
    stack3 = rack_builder.append_stack(n_blocks=2, precision="bfloat16")
    
    # Save rack
    rack_path = rack_builder.save_rack("./test_checkpoints/rack_test.pt")
    logger.info(f"Saved rack to: {rack_path}")
    
    # Create new rack builder and load
    logger.info("Loading rack...")
    new_rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    success = new_rack_builder.load_rack(rack_path)
    
    if success:
        logger.info("Successfully loaded rack!")
        logger.info(f"Loaded {new_rack_builder.current_stacks} stacks")
        
        # Verify stack info
        rack_info = new_rack_builder.get_rack_info()
        logger.info(f"Rack info: {rack_info}")
    else:
        logger.error("Failed to load rack!")
    
    return success


def test_stack_checkpointing():
    """Test stack-level checkpointing and loading"""
    logger.info("=== Testing Stack Checkpointing ===")
    
    # Create configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Create rack builder
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add stack
    logger.info("Adding stack...")
    stack1 = rack_builder.append_stack(n_blocks=3, precision="full")
    
    # Save individual stack
    stack_path = rack_builder.save_stack(0, "./test_checkpoints/stack_0.pt")
    logger.info(f"Saved stack 0 to: {stack_path}")
    
    # Create new rack builder and load stack
    logger.info("Loading stack...")
    new_rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    new_rack_builder.append_stack(n_blocks=3, precision="full")  # Create empty stack
    
    success = new_rack_builder.load_stack(0, stack_path)
    
    if success:
        logger.info("Successfully loaded stack!")
        stack_info = new_rack_builder.get_stack_info(0)
        logger.info(f"Stack info: {stack_info}")
    else:
        logger.error("Failed to load stack!")
    
    return success


def test_training_resumption():
    """Test training resumption from checkpoint"""
    logger.info("=== Testing Training Resumption ===")
    
    # Create configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Create trainer and rack builder
    trainer = ProgressiveTrainer(config=config)
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Create sample data
    dataloader = create_sample_data(batch_size=2, seq_len=8, vocab_size=1000)
    
    # Add and train first stack
    logger.info("Training first stack...")
    stack1 = rack_builder.append_stack(n_blocks=2, precision="full")
    results1 = trainer.train_rack(rack_builder, dataloader, target_stacks=1)
    
    # Save checkpoint
    checkpoint_path = trainer.save_progressive_checkpoint(
        stack_idx=0,
        rack_builder=rack_builder,
        epoch=1,
        loss=results1.get('final_loss', 0.5)
    )
    
    # Create new trainer and rack builder
    logger.info("Creating new trainer and rack builder...")
    new_trainer = ProgressiveTrainer(config=config)
    new_rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add empty stack
    new_rack_builder.append_stack(n_blocks=2, precision="full")
    
    # Restore from checkpoint
    logger.info("Restoring from checkpoint...")
    success = new_trainer.restore_from_checkpoint(checkpoint_path, new_rack_builder)
    
    if success:
        logger.info("Successfully restored from checkpoint!")
        
        # Verify restoration
        training_info = new_trainer.get_training_info()
        logger.info(f"Training info: {training_info}")
        
        rack_info = new_rack_builder.get_rack_info()
        logger.info(f"Rack info: {rack_info}")
    else:
        logger.error("Failed to restore from checkpoint!")
    
    return success


def main():
    """Main function to run all checkpointing tests"""
    logger.info("Starting Checkpointing Tests")
    
    # Create test directory
    Path("./test_checkpoints").mkdir(exist_ok=True)
    
    try:
        # Test progressive checkpointing
        checkpoint_path, rack_checkpoint_path = test_progressive_checkpointing()
        
        # Test rack checkpointing
        rack_success = test_rack_checkpointing()
        
        # Test stack checkpointing
        stack_success = test_stack_checkpointing()
        
        # Test training resumption
        resumption_success = test_training_resumption()
        
        # Summary
        logger.info("=== Test Summary ===")
        logger.info(f"Progressive checkpointing: ✅")
        logger.info(f"Rack checkpointing: {'✅' if rack_success else '❌'}")
        logger.info(f"Stack checkpointing: {'✅' if stack_success else '❌'}")
        logger.info(f"Training resumption: {'✅' if resumption_success else '❌'}")
        
        if all([rack_success, stack_success, resumption_success]):
            logger.info("🎉 All checkpointing tests passed!")
        else:
            logger.error("❌ Some checkpointing tests failed!")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
