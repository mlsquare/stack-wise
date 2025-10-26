# Checkpointing Guide

This guide explains how to use the checkpointing functionality in the Stack-Wise progressive training system.

## Overview

The Stack-Wise system provides comprehensive checkpointing capabilities for:
- **Progressive Training Checkpoints**: Save/load training state during progressive training
- **Rack-level Checkpoints**: Save/load complete racks with all stacks and LoRA adapters
- **Stack-level Checkpoints**: Save/load individual stacks
- **Training Resumption**: Resume training from any checkpoint

## Quick Start

### Basic Progressive Checkpointing

```python
from src.training import ProgressiveTrainer, ProgressiveRackBuilder
from src.config.base import StackWiseConfig

# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Create trainer and rack builder
trainer = ProgressiveTrainer(config=config)
rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")

# Add and train first stack
stack1 = rack_builder.append_stack(n_blocks=2, precision="full")
results1 = trainer.train_rack(rack_builder, dataloader, target_stacks=1)

# Save progressive checkpoint
checkpoint_path = trainer.save_progressive_checkpoint(
    stack_idx=0,
    rack_builder=rack_builder,
    epoch=1,
    loss=results1.get('final_loss', 0.5)
)

# Continue training...
stack2 = rack_builder.append_stack(n_blocks=2, precision="half")
results2 = trainer.train_rack(rack_builder, dataloader, target_stacks=2)
```

### Loading from Checkpoint

```python
# Create new trainer and rack builder
new_trainer = ProgressiveTrainer(config=config)
new_rack_builder = ProgressiveRackBuilder(config=config, building_mode="append")

# Add empty stack
new_rack_builder.append_stack(n_blocks=2, precision="full")

# Restore from checkpoint
success = new_trainer.restore_from_checkpoint(checkpoint_path, new_rack_builder)

if success:
    print("Successfully restored from checkpoint!")
    # Continue training...
```

## Progressive Training Checkpoints

### Saving Progressive Checkpoints

```python
# Save checkpoint after training a stack
checkpoint_path = trainer.save_progressive_checkpoint(
    stack_idx=0,                    # Current stack index
    rack_builder=rack_builder,      # Progressive rack builder
    optimizer=optimizer,            # Optional: optimizer state
    epoch=1,                       # Optional: epoch number
    loss=0.5                      # Optional: loss value
)
```

### Loading Progressive Checkpoints

```python
# Load checkpoint data
checkpoint_data = trainer.load_progressive_checkpoint(checkpoint_path)

if checkpoint_data:
    print(f"Checkpoint from: {checkpoint_data['timestamp']}")
    print(f"Stack index: {checkpoint_data['stack_idx']}")
    print(f"Loss: {checkpoint_data['loss']}")
```

### Restoring Training State

```python
# Restore complete training state
success = trainer.restore_from_checkpoint(
    checkpoint_path=checkpoint_path,
    rack_builder=rack_builder
)

if success:
    # Training state fully restored
    # Can continue training from where it left off
    pass
```

## Rack-level Checkpointing

### Saving Complete Racks

```python
# Save complete rack with all stacks and LoRA adapters
rack_path = rack_builder.save_rack("./checkpoints/complete_rack.pt")

# Save with trainer
rack_checkpoint_path = trainer.save_rack_checkpoint(
    rack_builder=rack_builder,
    optimizer=optimizer  # Optional
)
```

### Loading Complete Racks

```python
# Load complete rack
success = rack_builder.load_rack("./checkpoints/complete_rack.pt")

if success:
    print(f"Loaded rack with {rack_builder.current_stacks} stacks")
    
    # Get rack information
    rack_info = rack_builder.get_rack_info()
    print(f"Rack info: {rack_info}")
```

## Stack-level Checkpointing

### Saving Individual Stacks

```python
# Save individual stack
stack_path = rack_builder.save_stack(
    stack_idx=0,
    path="./checkpoints/stack_0.pt"
)
```

### Loading Individual Stacks

```python
# Load individual stack
success = rack_builder.load_stack(
    stack_idx=0,
    path="./checkpoints/stack_0.pt"
)

if success:
    stack_info = rack_builder.get_stack_info(0)
    print(f"Stack info: {stack_info}")
```

## Checkpoint Management

### Listing Available Checkpoints

```python
# List all available checkpoints
checkpoints = trainer.list_checkpoints()

for checkpoint in checkpoints:
    print(f"Path: {checkpoint['path']}")
    print(f"Stack: {checkpoint['stack_idx']}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Loss: {checkpoint['loss']}")
    print(f"Timestamp: {checkpoint['timestamp']}")
    print("---")
```

### Checkpoint Information

```python
# Get training information
training_info = trainer.get_training_info()
print(f"Training info: {training_info}")

# Get rack information
rack_info = rack_builder.get_rack_info()
print(f"Rack info: {rack_info}")

# Get stack information
stack_info = rack_builder.get_stack_info(0)
print(f"Stack 0 info: {stack_info}")
```

## Configuration

### Checkpoint Directory

```yaml
# config.yaml
training:
  checkpoint_dir: "./checkpoints"  # Checkpoint directory
  save_interval: 100               # Save every N steps
```

### Programmatic Configuration

```python
# Set checkpoint directory
trainer.checkpoint_dir = Path("./my_checkpoints")
trainer.save_interval = 50  # Save every 50 steps
```

## Advanced Features

### LoRA Adapter Checkpointing

The system automatically saves and loads LoRA adapter states:

```python
# LoRA adapters are automatically included in checkpoints
checkpoint_path = trainer.save_progressive_checkpoint(
    stack_idx=0,
    rack_builder=rack_builder  # Includes LoRA adapter states
)

# When loading, LoRA adapters are automatically restored
success = trainer.restore_from_checkpoint(checkpoint_path, rack_builder)
```

### Activation Cache Checkpointing

```python
# Activation caches are saved with progressive checkpoints
checkpoint_data = trainer.load_progressive_checkpoint(checkpoint_path)

if 'activation_cache' in checkpoint_data:
    # Activation cache is available
    activation_cache = checkpoint_data['activation_cache']
    print(f"Cached activations: {list(activation_cache.keys())}")
```

### Precision-aware Checkpointing

```python
# Different precision settings are preserved
rack_info = rack_builder.get_rack_info()
for stack_info in rack_info['stack_info']:
    print(f"Stack {stack_info['stack_idx']}: {stack_info['precision']}")
```

## Best Practices

### 1. Regular Checkpointing

```python
# Save checkpoints regularly during training
for epoch in range(num_epochs):
    # ... training code ...
    
    if epoch % save_interval == 0:
        checkpoint_path = trainer.save_progressive_checkpoint(
            stack_idx=current_stack,
            rack_builder=rack_builder,
            epoch=epoch,
            loss=current_loss
        )
```

### 2. Checkpoint Validation

```python
# Always validate checkpoint loading
success = trainer.restore_from_checkpoint(checkpoint_path, rack_builder)

if not success:
    print("Failed to restore from checkpoint!")
    # Handle error appropriately
```

### 3. Checkpoint Cleanup

```python
# List and manage checkpoints
checkpoints = trainer.list_checkpoints()

# Keep only recent checkpoints
if len(checkpoints) > max_checkpoints:
    # Remove old checkpoints
    for old_checkpoint in checkpoints[max_checkpoints:]:
        os.remove(old_checkpoint['path'])
```

### 4. Error Handling

```python
try:
    checkpoint_path = trainer.save_progressive_checkpoint(
        stack_idx=0,
        rack_builder=rack_builder
    )
    print(f"Checkpoint saved: {checkpoint_path}")
except Exception as e:
    print(f"Failed to save checkpoint: {e}")
    # Handle error appropriately
```

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Ensure the checkpoint file exists and path is correct
2. **State mismatch**: Ensure rack builder has the same structure as when checkpoint was saved
3. **Memory issues**: Use `map_location='cpu'` when loading checkpoints on CPU
4. **Permission errors**: Ensure write permissions for checkpoint directory

### Debug Information

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check checkpoint contents
checkpoint_data = trainer.load_progressive_checkpoint(checkpoint_path)
print(f"Checkpoint keys: {list(checkpoint_data.keys())}")
```

## Examples

See the following examples for complete working code:
- `examples/checkpointing_example.py` - Full progressive training with checkpointing
- `examples/simple_checkpointing_test.py` - Basic checkpointing tests

## API Reference

### ProgressiveTrainer Methods

- `save_progressive_checkpoint()` - Save progressive training checkpoint
- `load_progressive_checkpoint()` - Load progressive training checkpoint
- `restore_from_checkpoint()` - Restore complete training state
- `save_rack_checkpoint()` - Save complete rack checkpoint
- `list_checkpoints()` - List available checkpoints

### ProgressiveRackBuilder Methods

- `save_rack()` - Save complete rack
- `load_rack()` - Load complete rack
- `save_stack()` - Save individual stack
- `load_stack()` - Load individual stack
- `get_rack_info()` - Get rack information
- `get_stack_info()` - Get stack information
