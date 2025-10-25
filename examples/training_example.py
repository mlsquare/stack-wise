#!/usr/bin/env python3
"""
Training Example

This example demonstrates training with the Block/Stack/Rack architecture:
- Block: Standard transformer block (attention + FFN + layer norm + residual)
- Stack: Collection of multiple blocks
- Rack: Final model containing multiple stacks

Training modes:
- blockwise: Train each block independently
- stackwise: Train each stack independently
- rackwise: Train the entire rack together

Usage:
    python examples/training_example.py
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.architecture import Block, Stack, Rack, create_rack_from_config
from training.trainer import Trainer
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataloader(batch_size: int = 4, seq_len: int = 16, vocab_size: int = 1000):
    """Create a dummy dataloader for testing"""
    # Create dummy data
    input_ids = torch.randint(0, vocab_size, (100, seq_len))  # 100 samples
    labels = input_ids.clone()  # Same as input for language modeling
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def example_1_blockwise_training():
    """Example 1: Block-wise training (each block independently)"""
    logger.info("🔧 Example 1: Block-wise Training")
    logger.info("=" * 50)
    
    # Create a simple rack
    blocks = [Block(d_model=256, d_ff=1024, n_heads=4) for _ in range(6)]
    stacks = [Stack(blocks[:3], stack_id=0), Stack(blocks[3:], stack_id=1)]
    rack = Rack(stacks, vocab_size=1000, d_model=256)
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(batch_size=2, seq_len=8, vocab_size=1000)
    
    # Create configuration for block-wise training
    config = StackWiseConfig.from_yaml("config.yaml")
    config.training.training_architecture = "blockwise"
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train the architecture
    results = trainer.train_architecture(rack, dataloader)
    
    logger.info(f"✅ Block-wise training completed")
    logger.info(f"   Training architecture: {results['training_architecture']}")
    logger.info(f"   Total blocks trained: {results['total_blocks']}")
    logger.info(f"   Number of results: {len(results['results'])}")
    
    return results


def example_2_stackwise_training():
    """Example 2: Stack-wise training (each stack independently)"""
    logger.info("\n📚 Example 2: Stack-wise Training")
    logger.info("=" * 50)
    
    # Create a simple rack
    blocks = [Block(d_model=256, d_ff=1024, n_heads=4) for _ in range(6)]
    stacks = [Stack(blocks[:3], stack_id=0), Stack(blocks[3:], stack_id=1)]
    rack = Rack(stacks, vocab_size=1000, d_model=256)
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(batch_size=2, seq_len=8, vocab_size=1000)
    
    # Create configuration for stack-wise training
    config = StackWiseConfig.from_yaml("config.yaml")
    config.training.training_architecture = "stackwise"
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train the architecture
    results = trainer.train_architecture(rack, dataloader)
    
    logger.info(f"✅ Stack-wise training completed")
    logger.info(f"   Training architecture: {results['training_architecture']}")
    logger.info(f"   Total stacks trained: {results['total_stacks']}")
    logger.info(f"   Number of results: {len(results['results'])}")
    
    return results


def example_3_rackwise_training():
    """Example 3: Rack-wise training (entire model together)"""
    logger.info("\n🏗️ Example 3: Rack-wise Training")
    logger.info("=" * 50)
    
    # Create a simple rack
    blocks = [Block(d_model=256, d_ff=1024, n_heads=4) for _ in range(6)]
    stacks = [Stack(blocks[:3], stack_id=0), Stack(blocks[3:], stack_id=1)]
    rack = Rack(stacks, vocab_size=1000, d_model=256)
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(batch_size=2, seq_len=8, vocab_size=1000)
    
    # Create configuration for rack-wise training
    config = StackWiseConfig.from_yaml("config.yaml")
    config.training.training_architecture = "rackwise"
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train the architecture
    results = trainer.train_architecture(rack, dataloader)
    
    logger.info(f"✅ Rack-wise training completed")
    logger.info(f"   Training architecture: {results['training_architecture']}")
    logger.info(f"   Average loss: {results['avg_loss']:.4f}")
    logger.info(f"   Parameters: {results['parameters']:,}")
    
    return results


def example_4_create_from_config():
    """Example 4: Create and train rack from configuration"""
    logger.info("\n⚙️ Example 4: Create from Configuration")
    logger.info("=" * 50)
    
    # Load configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Create rack from config
    rack = create_rack_from_config(config.to_dict())
    
    # Create dummy dataloader
    dataloader = create_dummy_dataloader(batch_size=2, seq_len=8, vocab_size=1000)
    
    # Test different training architectures
    training_modes = ['blockwise', 'stackwise', 'rackwise']
    
    for mode in training_modes:
        logger.info(f"\nTesting {mode} training:")
        
        # Update configuration
        config.training.training_architecture = mode
        
        # Create trainer
        trainer = ArchitectureTrainer(config)
        
        # Train the architecture
        results = trainer.train_architecture(rack, dataloader)
        
        logger.info(f"   ✅ {mode} training completed")
        logger.info(f"   Architecture: {results['training_architecture']}")
    
    return rack


def example_5_architecture_comparison():
    """Example 5: Compare different architectures"""
    logger.info("\n📊 Example 5: Architecture Comparison")
    logger.info("=" * 50)
    
    # Create different rack configurations
    configs = [
        {"n_stacks": 2, "blocks_per_stack": 2},
        {"n_stacks": 3, "blocks_per_stack": 2},
        {"n_stacks": 2, "blocks_per_stack": 4},
    ]
    
    results = {}
    
    for i, config_dict in enumerate(configs):
        logger.info(f"\nConfiguration {i+1}: {config_dict}")
        
        # Create blocks and stacks
        total_blocks = config_dict["n_stacks"] * config_dict["blocks_per_stack"]
        blocks = [Block(d_model=128, d_ff=512, n_heads=4) for _ in range(total_blocks)]
        
        stacks = []
        for stack_idx in range(config_dict["n_stacks"]):
            start_idx = stack_idx * config_dict["blocks_per_stack"]
            end_idx = start_idx + config_dict["blocks_per_stack"]
            stack_blocks = blocks[start_idx:end_idx]
            stacks.append(Stack(stack_blocks, stack_id=stack_idx))
        
        rack = Rack(stacks, vocab_size=500, d_model=128)
        
        # Test forward pass
        input_ids = torch.randint(0, 500, (2, 8))
        
        with torch.no_grad():
            logits = rack(input_ids)
        
        results[f"config_{i+1}"] = {
            "config": config_dict,
            "parameters": rack.get_parameter_count(),
            "trainable_parameters": rack.get_trainable_parameter_count(),
            "num_stacks": len(rack.stacks),
            "total_blocks": sum(len(stack.blocks) for stack in rack.stacks),
            "output_shape": logits.shape
        }
        
        logger.info(f"   Parameters: {rack.get_parameter_count():,}")
        logger.info(f"   Stacks: {len(rack.stacks)}")
        logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
        logger.info(f"   Output shape: {logits.shape}")
    
    return results


def main():
    """Run all examples"""
    logger.info("🧠 StackWise Training Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Block-wise training
        example_1_blockwise_training()
        
        # Example 2: Stack-wise training
        example_2_stackwise_training()
        
        # Example 3: Rack-wise training
        example_3_rackwise_training()
        
        # Example 4: Create from config
        example_4_create_from_config()
        
        # Example 5: Architecture comparison
        example_5_architecture_comparison()
        
        logger.info("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
