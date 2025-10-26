# Unified Training Module

A comprehensive training framework for stack-wise transformer training with support for layer-wise, block-wise, and fused training modes.

## Features

- **Unified Architecture**: Single trainer supporting all training modes
- **Modular Components**: Pluggable strategies for masking, quantization, and caching
- **Time-Step-Based Masking**: Efficient progressive masking with discrete time steps
- **Quantization Support**: NF FP8/FP16 quantization with QLoRA adapters
- **Memory-Efficient Caching**: Time-step-aware activation caching
- **Configuration-Driven**: All behavior controlled by configuration

## Quick Start

```python
from src.training import UnifiedTrainer, TrainingConfig

# Create configuration
config = TrainingConfig(
    mode="blockwise",
    block_size=4,
    time_step_masking=True,
    quantization_enabled=True,
    qlora_enabled=True
)

# Create trainer
trainer = UnifiedTrainer(config)

# Train model
trainer.train_all_layers(dataloader, model_layers)
```

## Training Modes

### Progressive Training
```python
config = TrainingConfig(
    strategy="progressive",
    progressive=ProgressiveConfig(
        target_stacks=8,
        trunk_strategy="frozen",
        new_stack_precision="full"
    )
)
```

### End-to-End Stack-wise Training
```python
config = TrainingConfig(
    strategy="end_to_end",
    end_to_end_scope="stackwise",
    time_step_masking=False
)
```

### End-to-End Rack-wise Training
```python
config = TrainingConfig(
    strategy="end_to_end",
    end_to_end_scope="rackwise",
    time_step_masking=True,
    quantization_enabled=True,
    qlora_enabled=True
)
```

## Time-Step-Based Masking

```python
config = TrainingConfig(
    time_step_masking=True,
    num_time_steps=10,
    time_step_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    time_step_mask_fractions={
        0: 0.15,  # Early time steps: low masking
        5: 0.50,  # Middle time steps: medium masking
        9: 0.90   # Late time steps: high masking
    },
    store_all_time_steps=False,
    time_step_cache_size=1
)
```

## Quantization and QLoRA

```python
config = TrainingConfig(
    quantization_enabled=True,
    quantization_type="nf_fp8",  # or "fp16", "fp32"
    mixed_precision=True,
    backbone_quantized=True,
    adapters_full_precision=True,
    qlora_enabled=True,
    qlora_rank=16,
    qlora_alpha=32
)
```

## Configuration Options

### Training Strategy
- `strategy`: Training strategy ("progressive", "end_to_end")
  - progressive: Build and train stacks one by one
  - end_to_end: Train the entire model at once
- `end_to_end_scope`: End-to-end training scope ("stackwise", "rackwise") - only used when strategy="end_to_end"
  - stackwise: Train each stack independently
  - rackwise: Train the entire rack together
- `progressive`: Progressive training configuration

### Time-Step Masking
- `time_step_masking`: Enable time-step-based masking
- `num_time_steps`: Number of time steps
- `time_step_bins`: List of time step bins
- `time_step_mask_fractions`: Mask fractions per time step
- `store_all_time_steps`: Whether to store all time steps
- `time_step_cache_size`: Number of time steps to cache

### Quantization
- `quantization_enabled`: Enable quantization
- `quantization_type`: Type of quantization ("nf_fp8", "fp16", "fp32")
- `mixed_precision`: Enable mixed precision training
- `backbone_quantized`: Keep backbone quantized
- `adapters_full_precision`: Keep adapters in full precision

### QLoRA
- `qlora.enabled`: Enable QLoRA adapters
- `qlora.rank`: QLoRA rank
- `qlora.alpha`: QLoRA alpha
- `qlora.dropout`: QLoRA dropout
- `qlora.lr`: QLoRA learning rate
- `qlora.progressive_enabled`: Enable progressive QLoRA
- `qlora.strategy`: QLoRA strategy (simplified, progressive, variable)

### Caching and Saving
- `cache_mode`: Cache mode ("stack", "rack")
- `cache_dir`: Cache directory
- `save_stacks`: Always save individual stacks (default: true)
- `save_rack`: Optionally save entire rack (default: false)

## Architecture

```
src/training/
├── core/                    # Core training components
│   ├── unified_trainer.py  # Main unified trainer
│   ├── block_trainer.py    # Block-based training logic
│   └── fusion_trainer.py  # Fusion-specific logic
├── strategies/             # Training strategies
│   ├── masking/           # Masking strategies
│   ├── quantization/      # Quantization strategies
│   └── caching/           # Caching strategies
├── utils/                 # Utility modules
└── legacy/               # Legacy modules (deprecated)
```

## Examples

See `examples/unified_trainer_example.py` for comprehensive usage examples.

## Architecture Overview

The training module provides a unified framework with:

1. **Hierarchical Trainers**: BlockTrainer → StackTrainer → RackTrainer
2. **Modular Strategies**: Masking, quantization, and caching components
3. **Configuration-Driven**: All behavior controlled by configuration
4. **Progressive Training**: Support for growing model architectures

## Performance Benefits

- **Memory Efficiency**: 2-4x reduction with quantization
- **Training Speed**: Faster with QLoRA adapters
- **Scalability**: Support for large models
- **Flexibility**: Easy mode switching
- **Maintainability**: Clean, modular code
