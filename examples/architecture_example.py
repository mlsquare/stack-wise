#!/usr/bin/env python3
"""
Architecture Example

This example demonstrates the helper functions for creating
Block/Stack/Rack architectures:

1. create_block_spec() - Define block specifications
2. create_stack_from_spec() - Create stacks from block specs
3. create_rack_from_specs() - Create racks from stack specs
4. create_rack_from_config() - Create racks from configuration

Usage:
    python examples/architecture_example.py
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.architecture import (
    create_block_spec,
    create_stack_from_spec,
    create_rack_from_specs,
    create_rack_from_config
)
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_block_spec():
    """Example 1: Create block specifications"""
    logger.info("🔧 Example 1: Block Specifications")
    logger.info("=" * 50)
    
    # Create different block specifications
    block_specs = {
        "small": create_block_spec(
            d_model=256,
            d_ff=1024,
            n_heads=4,
            attention_type="standard"
        ),
        "medium": create_block_spec(
            d_model=512,
            d_ff=2048,
            n_heads=8,
            n_kv_heads=2,
            attention_type="gqa"
        ),
        "large": create_block_spec(
            d_model=1024,
            d_ff=4096,
            n_heads=16,
            n_kv_heads=4,
            attention_type="mla",
            kernel_type="gaussian",
            kernel_dim=128
        )
    }
    
    for name, spec in block_specs.items():
        logger.info(f"✅ {name.capitalize()} block spec:")
        logger.info(f"   d_model: {spec['d_model']}")
        logger.info(f"   d_ff: {spec['d_ff']}")
        logger.info(f"   n_heads: {spec['n_heads']}")
        logger.info(f"   attention_type: {spec['attention_type']}")
    
    return block_specs


def example_2_stack_from_spec():
    """Example 2: Create stacks from block specifications"""
    logger.info("\n📚 Example 2: Stacks from Block Specs")
    logger.info("=" * 50)
    
    # Create block specification
    block_spec = create_block_spec(
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,
        attention_type="gqa"
    )
    
    # Create different stacks
    stacks = []
    
    # Stack 1: 3 blocks
    stack1 = create_stack_from_spec(
        stack_id=0,
        n_blocks=3,
        block_spec=block_spec,
        freeze_blocks=False
    )
    stacks.append(stack1)
    
    # Stack 2: 4 blocks
    stack2 = create_stack_from_spec(
        stack_id=1,
        n_blocks=4,
        block_spec=block_spec,
        freeze_blocks=False
    )
    stacks.append(stack2)
    
    # Stack 3: 2 blocks (frozen)
    stack3 = create_stack_from_spec(
        stack_id=2,
        n_blocks=2,
        block_spec=block_spec,
        freeze_blocks=True
    )
    stacks.append(stack3)
    
    # Test stacks
    for i, stack in enumerate(stacks):
        logger.info(f"✅ Stack {i}:")
        logger.info(f"   Blocks: {stack.num_blocks}")
        logger.info(f"   Parameters: {stack.get_parameter_count():,}")
        logger.info(f"   Trainable: {stack.get_trainable_parameter_count():,}")
        
        # Test forward pass
        x = torch.randn(2, 8, 512)
        with torch.no_grad():
            output = stack(x)
        logger.info(f"   Input shape: {x.shape}")
        logger.info(f"   Output shape: {output.shape}")
    
    return stacks


def example_3_rack_from_specs():
    """Example 3: Create rack from stack specifications"""
    logger.info("\n🏗️ Example 3: Rack from Stack Specs")
    logger.info("=" * 50)
    
    # Create different block specifications for different stacks
    encoder_spec = create_block_spec(
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,
        attention_type="gqa",
        attention_mode="bidirectional"
    )
    
    decoder_spec = create_block_spec(
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,
        attention_type="gqa",
        attention_mode="causal"
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
            "n_blocks": 4,
            "block_spec": decoder_spec,
            "freeze_blocks": False
        },
        {
            "stack_id": 2,
            "n_blocks": 2,
            "block_spec": encoder_spec,
            "freeze_blocks": True  # Frozen stack
        }
    ]
    
    # Create rack from specifications
    rack = create_rack_from_specs(
        vocab_size=10000,
        d_model=512,
        stack_specs=stack_specs,
        tie_embeddings=True
    )
    
    # Test rack
    logger.info(f"✅ Rack created:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    logger.info(f"   Trainable: {rack.get_trainable_parameter_count():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 10000, (2, 8))
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def example_4_rack_from_config():
    """Example 4: Create rack from configuration"""
    logger.info("\n⚙️ Example 4: Rack from Configuration")
    logger.info("=" * 50)
    
    # Load configuration
    config = StackWiseConfig.from_yaml("config.yaml")
    
    # Create rack from config
    rack = create_rack_from_config(config.to_dict())
    
    # Test rack
    logger.info(f"✅ Rack created from config:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 10000, (2, 8))
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def example_5_heterogeneous_architecture():
    """Example 5: Create heterogeneous architecture with different block types"""
    logger.info("\n🔀 Example 5: Heterogeneous Architecture")
    logger.info("=" * 50)
    
    # Create different block specifications
    small_spec = create_block_spec(
        d_model=256,
        d_ff=1024,
        n_heads=4,
        attention_type="standard"
    )
    
    medium_spec = create_block_spec(
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,
        attention_type="gqa"
    )
    
    large_spec = create_block_spec(
        d_model=1024,
        d_ff=4096,
        n_heads=16,
        n_kv_heads=4,
        attention_type="mla",
        kernel_type="gaussian",
        kernel_dim=128
    )
    
    # Create stack specifications with different block types
    stack_specs = [
        {
            "stack_id": 0,
            "n_blocks": 2,
            "block_spec": small_spec,
            "freeze_blocks": False
        },
        {
            "stack_id": 1,
            "n_blocks": 3,
            "block_spec": medium_spec,
            "freeze_blocks": False
        },
        {
            "stack_id": 2,
            "n_blocks": 1,
            "block_spec": large_spec,
            "freeze_blocks": True  # Frozen large block
        }
    ]
    
    # Create rack
    rack = create_rack_from_specs(
        vocab_size=5000,
        d_model=256,  # Use smallest dimension for compatibility
        stack_specs=stack_specs,
        tie_embeddings=True
    )
    
    # Test rack
    logger.info(f"✅ Heterogeneous rack created:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 5000, (2, 8))
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def example_6_training_ready_architecture():
    """Example 6: Create training-ready architecture"""
    logger.info("\n🎯 Example 6: Training-Ready Architecture")
    logger.info("=" * 50)
    
    # Create block specification for training
    block_spec = create_block_spec(
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,
        attention_type="gqa",
        attention_mode="bidirectional",  # For training
        dropout=0.1,
        freeze_up_proj=True
    )
    
    # Create stack specifications for different training phases
    stack_specs = [
        {
            "stack_id": 0,
            "n_blocks": 4,
            "block_spec": block_spec,
            "freeze_blocks": False  # Trainable
        },
        {
            "stack_id": 1,
            "n_blocks": 4,
            "block_spec": block_spec,
            "freeze_blocks": False  # Trainable
        }
    ]
    
    # Create rack
    rack = create_rack_from_specs(
        vocab_size=10000,
        d_model=512,
        stack_specs=stack_specs,
        tie_embeddings=True
    )
    
    # Test rack
    logger.info(f"✅ Training-ready rack created:")
    logger.info(f"   Stacks: {len(rack.stacks)}")
    logger.info(f"   Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
    logger.info(f"   Parameters: {rack.get_parameter_count():,}")
    logger.info(f"   Trainable: {rack.get_trainable_parameter_count():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 10000, (2, 16))
    with torch.no_grad():
        logits = rack(input_ids)
    
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    
    return rack


def main():
    """Run all examples"""
    logger.info("🧠 StackWise Improved Architecture Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Block specifications
        example_1_block_spec()
        
        # Example 2: Stacks from block specs
        example_2_stack_from_spec()
        
        # Example 3: Rack from stack specs
        example_3_rack_from_specs()
        
        # Example 4: Rack from config
        example_4_rack_from_config()
        
        # Example 5: Heterogeneous architecture
        example_5_heterogeneous_architecture()
        
        # Example 6: Training-ready architecture
        example_6_training_ready_architecture()
        
        logger.info("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
