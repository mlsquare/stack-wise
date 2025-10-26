#!/usr/bin/env python3
"""
Simple Architecture Example

This example demonstrates the architecture configuration:
- Specify n_stacks and blocks_per_stack
- Cleaner configuration with helper functions
- Intuitive architecture creation

Usage:
    python examples/simple_architecture_example.py
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.architecture import (
    create_simple_rack,
    create_rack_from_config,
    create_block_spec,
    create_stack_from_spec,
    create_rack_from_specs
)
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_rack():
    """Example 1: Create a simple rack with n_stacks and blocks_per_stack"""
    logger.info("🔧 Example 1: Simple Rack Creation")
    logger.info("=" * 50)
    
    # Create a simple rack: 2 stacks, 4 blocks per stack
    rack = create_simple_rack(
        n_stacks=2,
        blocks_per_stack=4,
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,  # GQA is determined by n_kv_heads < n_heads
        vocab_size=10000,
        attention_preset="efficient_gqa"  # Use efficient GQA preset
    )
    
    # Test the rack
    input_ids = torch.randint(0, 10000, (2, 16))
    
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"✅ Simple rack created:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Blocks per stack: {rack.stacks[0].num_blocks}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def example_2_different_architectures():
    """Example 2: Create different architectures easily"""
    logger.info("\n📚 Example 2: Different Architectures")
    logger.info("=" * 50)
    
    architectures = [
        {"name": "Small", "n_stacks": 1, "blocks_per_stack": 2},
        {"name": "Medium", "n_stacks": 2, "blocks_per_stack": 3},
        {"name": "Large", "n_stacks": 3, "blocks_per_stack": 4},
        {"name": "XL", "n_stacks": 4, "blocks_per_stack": 6}
    ]
    
    racks = []
    
    for arch in architectures:
        rack = create_simple_rack(
            n_stacks=arch["n_stacks"],
            blocks_per_stack=arch["blocks_per_stack"],
            d_model=256,  # Smaller for demo
            d_ff=1024,
            n_heads=4,
            vocab_size=5000
        )
        
        total_blocks = arch["n_stacks"] * arch["blocks_per_stack"]
        
        logger.info(f"✅ {arch['name']} architecture:")
        logger.info(f"   Stacks: {arch['n_stacks']}")
        logger.info(f"   Blocks per stack: {arch['blocks_per_stack']}")
        logger.info(f"   Total blocks: {total_blocks}")
        logger.info(f"   Parameters: {rack.get_parameter_count():,}")
        
        racks.append(rack)
    
    return racks


def example_3_heterogeneous_architecture():
    """Example 3: Create heterogeneous architecture with different block types"""
    logger.info("\n🔀 Example 3: Heterogeneous Architecture")
    logger.info("=" * 50)
    
    # Create different block specifications
    encoder_spec = create_block_spec(
        d_model=256,
        d_ff=1024,
        n_heads=4,
        attention_preset="bert_style"
    )
    
    decoder_spec = create_block_spec(
        d_model=256,
        d_ff=1024,
        n_heads=4,
        attention_preset="gpt_style"
    )
    
    # Create stack specifications
    stack_specs = [
        {
            "stack_id": 0,
            "n_blocks": 3,
            "block_spec": encoder_spec,
            "freeze_blocks": False
        },
        {
            "stack_id": 1,
            "n_blocks": 3,
            "block_spec": decoder_spec,
            "freeze_blocks": False
        }
    ]
    
    # Create rack
    rack = create_rack_from_specs(
        vocab_size=5000,
        d_model=256,
        stack_specs=stack_specs,
        tie_embeddings=True
    )
    
    # Test the rack
    input_ids = torch.randint(0, 5000, (2, 8))
    
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"✅ Heterogeneous rack created:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def example_4_from_config():
    """Example 4: Create rack from configuration"""
    logger.info("\n⚙️ Example 4: Rack from Configuration")
    logger.info("=" * 50)
    
    # Load configuration
    config = StackWiseConfig.from_yaml("../config.yaml")
    
    # Create rack from config
    rack = create_rack_from_config(config.to_dict())
    
    # Test the rack - use vocab_size from config (1000)
    input_ids = torch.randint(0, 1000, (2, 8))
    
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"✅ Rack created from config:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Blocks per stack: {rack.stacks[0].num_blocks}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def example_5_training_scenarios():
    """Example 5: Different training scenarios"""
    logger.info("\n🎯 Example 5: Training Scenarios")
    logger.info("=" * 50)
    
    scenarios = [
        {
            "name": "Layer-wise Training",
            "n_stacks": 4,
            "blocks_per_stack": 1,
            "description": "Each block is its own stack"
        },
        {
            "name": "Block-wise Training", 
            "n_stacks": 2,
            "blocks_per_stack": 4,
            "description": "Groups of blocks per stack"
        },
        {
            "name": "Stack-wise Training",
            "n_stacks": 1,
            "blocks_per_stack": 8,
            "description": "All blocks in one stack"
        }
    ]
    
    for scenario in scenarios:
        rack = create_simple_rack(
            n_stacks=scenario["n_stacks"],
            blocks_per_stack=scenario["blocks_per_stack"],
            d_model=256,
            d_ff=1024,
            n_heads=4,
            vocab_size=5000
        )
        
        total_blocks = scenario["n_stacks"] * scenario["blocks_per_stack"]
        
        logger.info(f"✅ {scenario['name']}:")
        logger.info(f"   {scenario['description']}")
        logger.info(f"   Stacks: {scenario['n_stacks']}")
        logger.info(f"   Blocks per stack: {scenario['blocks_per_stack']}")
        logger.info(f"   Total blocks: {total_blocks}")
        logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    
    return scenarios


def example_6_memory_efficient_architectures():
    """Example 6: Memory-efficient architectures"""
    logger.info("\n💾 Example 6: Memory-Efficient Architectures")
    logger.info("=" * 50)
    
    # Different memory-efficient configurations
    configs = [
        {
            "name": "Small Model",
            "n_stacks": 1,
            "blocks_per_stack": 2,
            "d_model": 128,
            "d_ff": 512
        },
        {
            "name": "Medium Model",
            "n_stacks": 2,
            "blocks_per_stack": 3,
            "d_model": 256,
            "d_ff": 1024
        },
        {
            "name": "Large Model",
            "n_stacks": 3,
            "blocks_per_stack": 4,
            "d_model": 512,
            "d_ff": 2048
        }
    ]
    
    for config in configs:
        rack = create_simple_rack(
            n_stacks=config["n_stacks"],
            blocks_per_stack=config["blocks_per_stack"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            n_heads=4,
            vocab_size=5000
        )
        
        total_blocks = config["n_stacks"] * config["blocks_per_stack"]
        
        logger.info(f"✅ {config['name']}:")
        logger.info(f"   Stacks: {config['n_stacks']}")
        logger.info(f"   Blocks per stack: {config['blocks_per_stack']}")
        logger.info(f"   Total blocks: {total_blocks}")
        logger.info(f"   d_model: {config['d_model']}")
        logger.info(f"   d_ff: {config['d_ff']}")
        logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    
    return configs


def main():
    """Run all examples"""
    logger.info("🧠 StackWise Architecture Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Simple rack creation
        example_1_simple_rack()
        
        # Example 2: Different architectures
        example_2_different_architectures()
        
        # Example 3: Heterogeneous architecture
        example_3_heterogeneous_architecture()
        
        # Example 4: From configuration
        example_4_from_config()
        
        # Example 5: Training scenarios
        example_5_training_scenarios()
        
        # Example 6: Memory-efficient architectures
        example_6_memory_efficient_architectures()
        
        logger.info("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
