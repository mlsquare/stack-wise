#!/usr/bin/env python3
"""
Simplified QLoRA Example for Progressive Training

This example demonstrates the simplified QLoRA approach where:
1. QLoRA adapters are added to ALL stacks
2. Training is controlled through parameter freezing
3. Frozen trunk: all params (including LoRA) are frozen
4. QLoRA trunk: only LoRA adapters are trainable
"""

import torch
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.progressive_rack_builder import ProgressiveRackBuilder
from training.progressive_trainer import ProgressiveTrainer
from config.base import StackWiseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate simplified QLoRA approach"""
    
    # Create configuration
    config = StackWiseConfig()
    # Set model parameters
    config.model.vocab_size = 10000
    config.model.d_model = 512
    config.model.d_ff = 2048
    config.model.n_heads = 8
    config.model.n_kv_heads = 2
    # Set progressive training parameters
    config.training.progressive.enabled = True
    config.training.progressive.qlora_enabled = True
    config.training.progressive.qlora_rank = 16
    config.training.progressive.qlora_alpha = 32
    config.training.progressive.qlora_strategy = "simplified"
    
    # Create progressive rack builder (fully config-driven)
    rack = ProgressiveRackBuilder(
        config=config,
        building_mode="append",
        default_precision="full"
    )
    
    logger.info("=== Simplified QLoRA Progressive Training ===")
    
    # Step 1: Add QLoRA adapters to ALL stacks (simplified approach)
    logger.info("Step 1: Adding QLoRA adapters to all stacks...")
    rack.add_qlora_to_all_stacks(rank=16, alpha=32)
    
    # Step 2: Build initial rack with 2 stacks
    logger.info("Step 2: Building initial rack...")
    stack1 = rack.append_stack(n_blocks=4, precision="full")
    stack2 = rack.append_stack(n_blocks=4, precision="full")
    final_rack = rack.build_rack()
    
    logger.info(f"Built rack with {len(final_rack.stacks)} stacks")
    logger.info(f"Total parameters: {final_rack.get_parameter_count():,}")
    logger.info(f"Stack 1: {stack1.stack_id} with {len(stack1.blocks)} blocks")
    logger.info(f"Stack 2: {stack2.stack_id} with {len(stack2.blocks)} blocks")
    
    # Step 3: Demonstrate different training strategies
    logger.info("\n=== Training Strategy Examples ===")
    
    # Strategy 1: Frozen Trunk (all params frozen)
    logger.info("Strategy 1: Frozen Trunk")
    trunk_indices = [0]  # First stack (stack1) is trunk
    rack.freeze_trunk(trunk_indices)
    logger.info(f"Frozen trunk stacks: {trunk_indices} (stack1)")
    logger.info("Result: No training on trunk stacks")
    
    # Strategy 2: QLoRA Trunk (only LoRA adapters trainable)
    logger.info("\nStrategy 2: QLoRA Trunk")
    # First, unfreeze to demonstrate QLoRA
    for stack_idx in trunk_indices:
        if stack_idx in rack.qlora_adapters:
            rack.enable_qlora_training([stack_idx])
    
    rack.freeze_all_but_qlora(trunk_indices)
    logger.info(f"Applied QLoRA freezing to trunk stacks: {trunk_indices} (stack1)")
    logger.info("Result: Only LoRA adapters are trainable")
    
    # Strategy 3: Full Precision New Stack
    logger.info("\nStrategy 3: Full Precision New Stack")
    new_stack = rack.append_stack(n_blocks=4, precision="full")
    logger.info(f"Added new stack {new_stack.stack_id} with full precision")
    logger.info(f"New stack has {len(new_stack.blocks)} blocks")
    logger.info("Result: New stack trains in full precision")
    
    # Step 4: Show QLoRA adapter information
    logger.info("\n=== QLoRA Adapter Information ===")
    for stack_idx, adapter_info in rack.qlora_adapters.items():
        logger.info(f"Stack {stack_idx}: rank={adapter_info['rank']}, "
                   f"alpha={adapter_info['alpha']}, enabled={adapter_info['enabled']}")
    
    # Step 5: Demonstrate training control
    logger.info("\n=== Training Control Examples ===")
    
    # Enable QLoRA training for specific stacks
    rack.enable_qlora_training([0, 1])
    logger.info("Enabled QLoRA training for stacks 0 and 1")
    
    # Disable QLoRA training (freeze everything)
    rack.disable_qlora_training([0])
    logger.info("Disabled QLoRA training for stack 0 (fully frozen)")
    
    logger.info("\n=== Benefits of Simplified QLoRA ===")
    logger.info("✅ One QLoRA setup for all stacks")
    logger.info("✅ Easy switching between frozen and QLoRA modes")
    logger.info("✅ Memory efficient (only LoRA adapters trainable)")
    logger.info("✅ Consistent interface for all stacks")
    logger.info("✅ Simplified management and configuration")

if __name__ == "__main__":
    main()
