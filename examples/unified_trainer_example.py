"""
Example usage of the new UnifiedTrainer.

This example demonstrates how to use the new modular unified trainer
with different training modes and configurations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import UnifiedTrainer, TrainingConfig


def create_sample_model_layers(num_layers: int = 12, d_model: int = 512) -> list:
    """Create sample model layers for demonstration."""
    layers = []
    
    for i in range(num_layers):
        # Simple MLP layer for demonstration
        layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        layers.append(layer)
    
    return layers


def create_sample_dataloader(batch_size: int = 4, seq_len: int = 512, vocab_size: int = 1000) -> DataLoader:
    """Create sample dataloader for demonstration."""
    # Create random input data
    input_ids = torch.randint(0, vocab_size, (100, seq_len))  # 100 samples
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def example_layerwise_training():
    """Example of layer-wise training."""
    print("=== Layer-wise Training Example ===")
    
    # Create configuration
    config = TrainingConfig(
        mode="layerwise",
        block_size=1,
        time_step_masking=False,
        quantization_enabled=False,
        qlora_enabled=False,
        epochs_per_block=1,
        learning_rate=1e-4
    )
    
    # Create trainer
    trainer = UnifiedTrainer(config)
    
    # Create sample data
    model_layers = create_sample_model_layers(num_layers=6)
    dataloader = create_sample_dataloader()
    
    # Train
    trainer.train_all_layers(dataloader, model_layers)
    
    # Get training info
    info = trainer.get_training_info()
    print(f"Training completed. Info: {info}")


def example_blockwise_training():
    """Example of block-wise training."""
    print("=== Block-wise Training Example ===")
    
    # Create configuration
    config = TrainingConfig(
        mode="blockwise",
        block_size=4,
        time_step_masking=False,
        quantization_enabled=False,
        qlora_enabled=False,
        epochs_per_block=1,
        learning_rate=1e-4
    )
    
    # Create trainer
    trainer = UnifiedTrainer(config)
    
    # Create sample data
    model_layers = create_sample_model_layers(num_layers=12)
    dataloader = create_sample_dataloader()
    
    # Train
    trainer.train_all_layers(dataloader, model_layers)
    
    # Get training info
    info = trainer.get_training_info()
    print(f"Training completed. Info: {info}")


def example_fused_training_with_quantization():
    """Example of fused training with quantization and QLoRA."""
    print("=== Fused Training with Quantization Example ===")
    
    # Create configuration
    config = TrainingConfig(
        mode="fused",
        block_size=4,
        fusion_mode="frozen",
        time_step_masking=True,
        num_time_steps=5,
        time_step_bins=[0, 1, 2, 3, 4],
        time_step_mask_fractions={0: 0.15, 2: 0.50, 4: 0.90},
        quantization_enabled=True,
        quantization_type="fp16",
        qlora_enabled=True,
        qlora_rank=16,
        epochs_per_block=1,
        learning_rate=1e-4
    )
    
    # Create trainer
    trainer = UnifiedTrainer(config)
    
    # Create sample data
    model_layers = create_sample_model_layers(num_layers=12)
    dataloader = create_sample_dataloader()
    
    # Train
    trainer.train_all_layers(dataloader, model_layers)
    
    # Get training info
    info = trainer.get_training_info()
    print(f"Training completed. Info: {info}")


def example_time_step_masking():
    """Example of time-step-based masking."""
    print("=== Time-step-based Masking Example ===")
    
    # Create configuration
    config = TrainingConfig(
        mode="blockwise",
        block_size=2,
        time_step_masking=True,
        num_time_steps=10,
        time_step_bins=list(range(10)),
        time_step_mask_fractions={
            0: 0.15,  # Early time steps: low masking
            5: 0.50,  # Middle time steps: medium masking
            9: 0.90   # Late time steps: high masking
        },
        store_all_time_steps=False,
        time_step_cache_size=1,
        quantization_enabled=False,
        qlora_enabled=False,
        epochs_per_block=1,
        learning_rate=1e-4
    )
    
    # Create trainer
    trainer = UnifiedTrainer(config)
    
    # Create sample data
    model_layers = create_sample_model_layers(num_layers=8)
    dataloader = create_sample_dataloader()
    
    # Train
    trainer.train_all_layers(dataloader, model_layers)
    
    # Get training info
    info = trainer.get_training_info()
    print(f"Training completed. Info: {info}")


if __name__ == "__main__":
    print("UnifiedTrainer Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_layerwise_training()
        print()
        
        example_blockwise_training()
        print()
        
        example_fused_training_with_quantization()
        print()
        
        example_time_step_masking()
        print()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
