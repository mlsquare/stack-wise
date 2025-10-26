#!/usr/bin/env python3
"""
Dual-LoRA Example

This example demonstrates the dual-LoRA approach:
1. Always add LoRA adapters to each stack as it's created
2. Add additional QLoRA adapters to the entire trunk when new stacks are added
3. If QLoRA only: trunk is frozen except for all adapters (stack LoRA + trunk QLoRA)

Usage:
    python examples/dual_lora_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.progressive_rack_builder import ProgressiveRackBuilder
from src.config.base import StackWiseConfig

def demonstrate_dual_lora():
    """Demonstrate the dual-LoRA approach"""
    
    print("=== Dual-LoRA Approach Demonstration ===")
    
    # Create configuration with dual-LoRA
    config = StackWiseConfig()
    # Set model parameters
    config.model.vocab_size = 10000
    config.model.d_model = 512
    config.model.d_ff = 2048
    config.model.n_heads = 8
    config.model.n_kv_heads = 2
    # Set progressive training parameters
    config.training.progressive.qlora_enabled = True
    config.training.progressive.qlora_strategy = "simplified"
    config.training.progressive.qlora_rank = 16  # Stack LoRA rank
    config.training.progressive.qlora_alpha = 32  # Stack LoRA alpha
    
    # Progressive QLoRA configuration
    config.training.progressive.progressive_qlora = True
    config.training.progressive.progressive_qlora_rank = 8  # Progressive QLoRA rank (smaller)
    config.training.progressive.progressive_qlora_alpha = 16  # Progressive QLoRA alpha
    
    # Create rack builder
    rack = ProgressiveRackBuilder(config=config, building_mode="append")
    
    print(f"Initial state: {rack.current_stacks} stacks")
    print(f"LoRA adapters: {len(rack.qlora_adapters)}")
    
    # Add stacks progressively
    for i in range(4):
        print(f"\n--- Adding Stack {i} ---")
        print(f"Before: current_stacks = {rack.current_stacks}")
        
        # Add stack
        stack = rack.append_stack(n_blocks=2)
        
        print(f"After: current_stacks = {rack.current_stacks}")
        print(f"Stack ID: {stack.stack_id}")
        print(f"Total LoRA adapters: {len(rack.qlora_adapters)}")
        
        # Show LoRA configurations
        print("LoRA configurations:")
        for adapter_key, adapter_info in rack.qlora_adapters.items():
            if adapter_key == stack.stack_id:
                print(f"  Stack {adapter_key}: rank={adapter_info['rank']}, alpha={adapter_info['alpha']} (stack LoRA)")
            elif isinstance(adapter_key, str) and adapter_key.startswith(f"progressive_qlora_{stack.stack_id}"):
                print(f"  {adapter_key}: rank={adapter_info['rank']}, alpha={adapter_info['alpha']} (progressive QLoRA)")
    
    print(f"\n=== Final State ===")
    print(f"Total stacks: {rack.current_stacks}")
    print(f"Total LoRA adapters: {len(rack.qlora_adapters)}")
    
    # Show all LoRA configurations
    print("\nAll LoRA configurations:")
    for adapter_key, adapter_info in rack.qlora_adapters.items():
        if isinstance(adapter_key, int):
            print(f"Stack {adapter_key}: rank={adapter_info['rank']}, alpha={adapter_info['alpha']} (stack LoRA)")
        elif adapter_key.startswith("progressive_qlora_"):
            stack_idx = adapter_key.split("_")[-1]
            print(f"Progressive QLoRA for stack {stack_idx}: rank={adapter_info['rank']}, alpha={adapter_info['alpha']}")

def demonstrate_training_strategies():
    """Demonstrate training strategies with dual-LoRA"""
    
    print("\n=== Training Strategies with Dual-LoRA ===")
    
    # Create configuration
    config = StackWiseConfig()
    # Set model parameters
    config.model.vocab_size = 10000
    config.model.d_model = 512
    config.model.d_ff = 2048
    config.model.n_heads = 8
    config.model.n_kv_heads = 2
    # Set progressive training parameters
    config.training.progressive.qlora_enabled = True
    config.training.progressive.progressive_qlora = True
    config.training.progressive.qlora_rank = 16
    config.training.progressive.progressive_qlora_rank = 8
    
    rack = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add some stacks
    for i in range(3):
        stack = rack.append_stack(n_blocks=2)
        print(f"Added stack {stack.stack_id}")
    
    print(f"\nTotal LoRA adapters: {len(rack.qlora_adapters)}")
    
    # Strategy 1: Frozen Trunk
    print("\n1. Frozen Trunk Strategy")
    trunk_indices = [0, 1]  # First two stacks are trunk
    rack.freeze_trunk(trunk_indices)
    print(f"Frozen trunk stacks: {trunk_indices}")
    print("Result: No training on trunk stacks (all parameters frozen)")
    
    # Strategy 2: QLoRA Trunk (dual-LoRA)
    print("\n2. QLoRA Trunk Strategy (Dual-LoRA)")
    rack.enable_qlora_training(trunk_indices)
    rack.freeze_all_but_qlora(trunk_indices)
    print(f"QLoRA training enabled for trunk stacks: {trunk_indices}")
    print("Result: Only LoRA adapters are trainable (stack LoRA + progressive QLoRA)")
    
    # Show which adapters are trainable
    print("\nTrainable adapters for trunk stacks:")
    for stack_idx in trunk_indices:
        print(f"Stack {stack_idx}:")
        if stack_idx in rack.qlora_adapters:
            stack_lora = rack.qlora_adapters[stack_idx]
            print(f"  - Stack LoRA: rank={stack_lora['rank']}, alpha={stack_lora['alpha']}")
        
        progressive_qlora_key = f"progressive_qlora_{stack_idx}"
        if progressive_qlora_key in rack.qlora_adapters:
            progressive_qlora = rack.qlora_adapters[progressive_qlora_key]
            print(f"  - Progressive QLoRA: rank={progressive_qlora['rank']}, alpha={progressive_qlora['alpha']}")

def main():
    """Main demonstration"""
    
    print("Dual-LoRA Approach Demonstration")
    print("=" * 50)
    
    # Demonstrate dual-LoRA approach
    demonstrate_dual_lora()
    
    # Demonstrate training strategies
    demonstrate_training_strategies()
    
    print("\n=== Benefits of Dual-LoRA ===")
    print("✅ Always add LoRA adapters to each stack")
    print("✅ Additional progressive QLoRA adapters to entire trunk when new stacks are added")
    print("✅ Dual-LoRA training: both stack LoRA and progressive QLoRA are trainable")
    print("✅ Flexible training strategies")
    print("✅ Progressive complexity as racks grow")

if __name__ == "__main__":
    main()
