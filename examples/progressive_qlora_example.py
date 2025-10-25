#!/usr/bin/env python3
"""
Progressive QLoRA Example

This example demonstrates the progressive QLoRA approach where:
1. QLoRA adapters are added to each stack as it's created
2. Different QLoRA configurations can be used per stack
3. Supports multiple strategies: simplified, progressive, variable

Usage:
    python examples/progressive_qlora_example.py
"""

import torch
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.progressive_rack_builder import ProgressiveRackBuilder
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_qlora_strategies():
    """Demonstrate different QLoRA strategies"""
    
    logger.info("=== Progressive QLoRA Strategies ===")
    
    # Strategy 1: Simplified QLoRA (all stacks same config)
    logger.info("\n1. Simplified QLoRA Strategy")
    config1 = StackWiseConfig()
    config1.training.progressive.qlora_enabled = True
    config1.training.progressive.qlora_strategy = "simplified"
    config1.training.progressive.qlora_rank = 16
    config1.training.progressive.qlora_alpha = 32
    
    builder1 = ProgressiveRackBuilder(config=config1, building_mode="append")
    
    # Add stacks - all get same QLoRA config
    for i in range(4):
        stack = builder1.append_stack(n_blocks=2)
        qlora_info = builder1.qlora_adapters.get(i, {})
        logger.info(f"Stack {i}: rank={qlora_info.get('rank', 'N/A')}, alpha={qlora_info.get('alpha', 'N/A')}")
    
    # Strategy 2: Progressive QLoRA (increasing rank)
    logger.info("\n2. Progressive QLoRA Strategy (Increasing Rank)")
    config2 = StackWiseConfig()
    config2.training.progressive.qlora_enabled = True
    config2.training.progressive.qlora_strategy = "progressive"
    config2.training.progressive.qlora_rank = 8  # Base rank
    config2.training.progressive.qlora_alpha = 16  # Base alpha
    config2.training.progressive.qlora_rank_pattern = "increasing"
    config2.training.progressive.qlora_alpha_pattern = "constant"
    
    builder2 = ProgressiveRackBuilder(config=config2, building_mode="append")
    
    # Add stacks - rank increases: 8, 16, 32, 64
    for i in range(4):
        stack = builder2.append_stack(n_blocks=2)
        qlora_info = builder2.qlora_adapters.get(i, {})
        logger.info(f"Stack {i}: rank={qlora_info.get('rank', 'N/A')}, alpha={qlora_info.get('alpha', 'N/A')}")
    
    # Strategy 3: Progressive QLoRA (decreasing rank)
    logger.info("\n3. Progressive QLoRA Strategy (Decreasing Rank)")
    config3 = StackWiseConfig()
    config3.training.progressive.qlora_enabled = True
    config3.training.progressive.qlora_strategy = "progressive"
    config3.training.progressive.qlora_rank = 64  # Base rank
    config3.training.progressive.qlora_alpha = 128  # Base alpha
    config3.training.progressive.qlora_rank_pattern = "decreasing"
    config3.training.progressive.qlora_alpha_pattern = "decreasing"
    
    builder3 = ProgressiveRackBuilder(config=config3, building_mode="append")
    
    # Add stacks - rank decreases: 64, 32, 16, 8
    for i in range(4):
        stack = builder3.append_stack(n_blocks=2)
        qlora_info = builder3.qlora_adapters.get(i, {})
        logger.info(f"Stack {i}: rank={qlora_info.get('rank', 'N/A')}, alpha={qlora_info.get('alpha', 'N/A')}")
    
    # Strategy 4: Variable QLoRA (custom config per stack)
    logger.info("\n4. Variable QLoRA Strategy (Custom Config)")
    config4 = StackWiseConfig()
    config4.training.progressive.qlora_enabled = True
    config4.training.progressive.qlora_strategy = "variable"
    config4.training.progressive.qlora_configs = {
        0: {"rank": 4, "alpha": 8},    # Small QLoRA
        1: {"rank": 16, "alpha": 32},  # Medium QLoRA
        2: {"rank": 32, "alpha": 64},  # Large QLoRA
        3: {"rank": 64, "alpha": 128}  # Very large QLoRA
    }
    
    builder4 = ProgressiveRackBuilder(config=config4, building_mode="append")
    
    # Add stacks - custom config per stack
    for i in range(4):
        stack = builder4.append_stack(n_blocks=2)
        qlora_info = builder4.qlora_adapters.get(i, {})
        logger.info(f"Stack {i}: rank={qlora_info.get('rank', 'N/A')}, alpha={qlora_info.get('alpha', 'N/A')}")

def demonstrate_building_modes():
    """Demonstrate QLoRA with different building modes"""
    
    logger.info("\n=== QLoRA with Different Building Modes ===")
    
    # Append mode (left to right)
    logger.info("\n1. Append Mode (Left to Right)")
    config = StackWiseConfig()
    config.training.progressive.qlora_enabled = True
    config.training.progressive.qlora_strategy = "progressive"
    config.training.progressive.qlora_rank = 8
    config.training.progressive.qlora_rank_pattern = "increasing"
    
    builder_append = ProgressiveRackBuilder(config=config, building_mode="append")
    
    for i in range(3):
        stack = builder_append.append_stack(n_blocks=2)
        qlora_info = builder_append.qlora_adapters.get(i, {})
        logger.info(f"Appended Stack {i}: rank={qlora_info.get('rank', 'N/A')}")
    
    # Prepend mode (right to left)
    logger.info("\n2. Prepend Mode (Right to Left)")
    builder_prepend = ProgressiveRackBuilder(config=config, building_mode="prepend")
    
    for i in range(3):
        stack = builder_prepend.prepend_stack(n_blocks=2)
        qlora_info = builder_prepend.qlora_adapters.get(i, {})
        logger.info(f"Prepended Stack {i}: rank={qlora_info.get('rank', 'N/A')}")

def demonstrate_qlora_training_strategies():
    """Demonstrate QLoRA training strategies"""
    
    logger.info("\n=== QLoRA Training Strategies ===")
    
    config = StackWiseConfig()
    config.training.progressive.qlora_enabled = True
    config.training.progressive.qlora_strategy = "progressive"
    config.training.progressive.qlora_rank = 16
    config.training.progressive.qlora_rank_pattern = "increasing"
    
    builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add some stacks
    for i in range(3):
        stack = builder.append_stack(n_blocks=2)
        qlora_info = builder.qlora_adapters.get(i, {})
        logger.info(f"Stack {i}: rank={qlora_info.get('rank', 'N/A')}")
    
    # Demonstrate training strategies
    logger.info("\nTraining Strategy 1: Frozen Trunk")
    trunk_indices = [0, 1]  # First two stacks are trunk
    builder.freeze_trunk(trunk_indices)
    logger.info(f"Frozen trunk stacks: {trunk_indices}")
    logger.info("Result: No training on trunk stacks")
    
    logger.info("\nTraining Strategy 2: QLoRA Trunk")
    # Enable QLoRA training for trunk
    builder.enable_qlora_training(trunk_indices)
    builder.freeze_all_but_qlora(trunk_indices)
    logger.info(f"QLoRA training enabled for trunk stacks: {trunk_indices}")
    logger.info("Result: Only LoRA adapters are trainable in trunk")

def main():
    """Main demonstration"""
    
    logger.info("Progressive QLoRA Demonstration")
    logger.info("=" * 50)
    
    # Demonstrate different QLoRA strategies
    demonstrate_qlora_strategies()
    
    # Demonstrate building modes
    demonstrate_building_modes()
    
    # Demonstrate training strategies
    demonstrate_qlora_training_strategies()
    
    logger.info("\n=== Benefits of Progressive QLoRA ===")
    logger.info("✅ Flexible QLoRA configuration per stack")
    logger.info("✅ Progressive patterns (increasing/decreasing)")
    logger.info("✅ Custom configurations per stack")
    logger.info("✅ Works with both append and prepend modes")
    logger.info("✅ Supports different training strategies")
    logger.info("✅ Memory efficient (only needed adapters)")

if __name__ == "__main__":
    main()
