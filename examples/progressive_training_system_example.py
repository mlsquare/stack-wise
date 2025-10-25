"""
Example usage of the new Progressive Training System.

This example demonstrates how to use the new progressive training system
with dual-LoRA approach and different training modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import ProgressiveTrainer, ProgressiveRackBuilder
from src.config.base import StackWiseConfig


def create_sample_data(batch_size: int = 4, seq_len: int = 128, d_model: int = 512):
    """Create sample data for demonstration."""
    # Create sample input data
    x = torch.randn(batch_size, seq_len, d_model)
    y = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Create DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def demonstrate_basic_progressive_training():
    """Demonstrate basic progressive training."""
    print("🚀 Basic Progressive Training")
    print("=" * 50)
    
    # Create configuration
    config = StackWiseConfig()
    config.training.progressive.enabled = True
    config.training.progressive.qlora_enabled = True
    config.training.progressive.progressive_qlora = True
    
    # Create progressive rack builder
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add stacks progressively
    stack1 = rack_builder.append_stack(n_blocks=4, precision="full")
    stack2 = rack_builder.append_stack(n_blocks=4, precision="half")
    
    print(f"✅ Created rack with {rack_builder.current_stacks} stacks")
    print(f"✅ Stack 1: {stack1.stack_id} with {len(stack1.blocks)} blocks")
    print(f"✅ Stack 2: {stack2.stack_id} with {len(stack2.blocks)} blocks")
    
    return rack_builder


def demonstrate_dual_lora_approach():
    """Demonstrate dual-LoRA approach."""
    print("\n🔄 Dual-LoRA Approach")
    print("=" * 50)
    
    # Create configuration with dual-LoRA
    config = StackWiseConfig()
    config.training.progressive.enabled = True
    config.training.progressive.qlora_enabled = True
    config.training.progressive.progressive_qlora = True
    config.training.progressive.qlora_rank = 16
    config.training.progressive.progressive_qlora_rank = 8
    
    # Create progressive rack builder
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add stacks progressively
    for i in range(3):
        stack = rack_builder.append_stack(n_blocks=2, precision="full")
        print(f"✅ Added stack {stack.stack_id} with LoRA adapters")
    
    # Show LoRA adapters
    stack_lora_count = len([k for k in rack_builder.qlora_adapters.keys() if isinstance(k, int)])
    progressive_lora_count = len([k for k in rack_builder.qlora_adapters.keys() if k.startswith('progressive_qlora_')])
    
    print(f"✅ Stack LoRA adapters: {stack_lora_count}")
    print(f"✅ Progressive QLoRA adapters: {progressive_lora_count}")
    
    return rack_builder


def demonstrate_progressive_training():
    """Demonstrate progressive training with trainer."""
    print("\n🎯 Progressive Training with Trainer")
    print("=" * 50)
    
    # Create configuration
    config = StackWiseConfig()
    config.training.progressive.enabled = True
    config.training.progressive.qlora_enabled = True
    config.training.progressive.progressive_qlora = True
    
    # Create progressive rack builder
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add stacks progressively
    for i in range(2):
        stack = rack_builder.append_stack(n_blocks=4, precision="full")
        print(f"✅ Added stack {stack.stack_id}")
    
    # Create sample data
    dataloader = create_sample_data()
    
    # Create progressive trainer
    trainer = ProgressiveTrainer(config=config)
    
    # Train progressively
    print("🚀 Starting progressive training...")
    results = trainer.train_rack(rack_builder, dataloader, target_stacks=2)
    
    print(f"✅ Progressive training completed with {len(results)} results")
    
    return results


def demonstrate_precision_modes():
    """Demonstrate different precision modes."""
    print("\n⚡ Precision Modes")
    print("=" * 50)
    
    # Create configuration
    config = StackWiseConfig()
    config.training.progressive.enabled = True
    config.training.progressive.qlora_enabled = True
    
    # Create progressive rack builder
    rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")
    
    # Add stacks with different precision modes
    precision_modes = ["full", "half", "bfloat16", "nvfp4"]
    
    for i, precision in enumerate(precision_modes):
        stack = rack_builder.append_stack(n_blocks=2, precision=precision)
        print(f"✅ Added stack {stack.stack_id} with precision: {precision}")
    
    print(f"✅ Created rack with {rack_builder.current_stacks} stacks")
    
    return rack_builder


def main():
    """Main demonstration function."""
    print("🚀 Progressive Training System Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic progressive training
        rack_builder1 = demonstrate_basic_progressive_training()
        
        # Example 2: Dual-LoRA approach
        rack_builder2 = demonstrate_dual_lora_approach()
        
        # Example 3: Progressive training with trainer
        results = demonstrate_progressive_training()
        
        # Example 4: Precision modes
        rack_builder3 = demonstrate_precision_modes()
        
        print("\n🎉 All Progressive Training System examples completed successfully!")
        print("\n📊 Summary:")
        print(f"  - Basic progressive training: ✅")
        print(f"  - Dual-LoRA approach: ✅")
        print(f"  - Progressive training with trainer: ✅")
        print(f"  - Precision modes: ✅")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
