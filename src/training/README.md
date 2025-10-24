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

### Layer-wise Training (block_size=1)
```python
config = TrainingConfig(
    mode="layerwise",
    block_size=1,
    time_step_masking=False
)
```

### Block-wise Training (block_size=4)
```python
config = TrainingConfig(
    mode="blockwise", 
    block_size=4,
    time_step_masking=False
)
```

### Fused Training with Quantization
```python
config = TrainingConfig(
    mode="fused",
    block_size=4,
    fusion_mode="frozen",
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

### Basic Training
- `mode`: Training mode ("layerwise", "blockwise", "fused")
- `block_size`: Number of layers per block
- `fusion_mode`: Fusion strategy ("frozen", "trainable")

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
- `qlora_enabled`: Enable QLoRA adapters
- `qlora_rank`: QLoRA rank
- `qlora_alpha`: QLoRA alpha
- `qlora_dropout`: QLoRA dropout

### Caching
- `cache_mode`: Cache mode ("layerwise", "fusion")
- `cache_dir`: Cache directory
- `fusion_evaluation`: Enable fusion evaluation
- `save_fused_checkpoints`: Save fused checkpoints

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

## Migration from Legacy

The new `UnifiedTrainer` replaces the legacy `LayerwiseTrainer`. To migrate:

1. Replace `LayerwiseTrainer` with `UnifiedTrainer`
2. Update configuration to use `TrainingConfig`
3. Use the new modular architecture

## Performance Benefits

- **Memory Efficiency**: 2-4x reduction with quantization
- **Training Speed**: Faster with QLoRA adapters
- **Scalability**: Support for large models
- **Flexibility**: Easy mode switching
- **Maintainability**: Clean, modular code
