# 🏗️ StackWise Architecture: Block, Stack, and Rack

This document explains the new hierarchical architecture introduced in StackWise, which provides a clearer and more intuitive naming system for transformer components.

## 📋 Architecture Overview

The new architecture follows a physical analogy that makes the structure more intuitive:

```
Rack (Final Model)
├── Stack 1 (Collection of Blocks)
│   ├── Block 1 (Standard Transformer Block)
│   ├── Block 2 (Standard Transformer Block)
│   └── Block 3 (Standard Transformer Block)
├── Stack 2 (Collection of Blocks)
│   ├── Block 4 (Standard Transformer Block)
│   ├── Block 5 (Standard Transformer Block)
│   └── Block 6 (Standard Transformer Block)
└── ... (More Stacks)
```

## 🔧 Component Definitions

### Block
A **Block** is the standard transformer block containing:
- **Self-attention mechanism** (MHA, GQA, MLA, or kernel-based)
- **Feed-forward network** (SwiGLU with optional frozen up-projections)
- **Layer normalization** (pre-norm style)
- **Residual connections** (around both attention and FFN)

```python
from model.architecture import Block

# Create a single block
block = Block(
    d_model=512,
    d_ff=2048,
    n_heads=8,
    n_kv_heads=2,
    attention_type="gqa",
    attention_mode="bidirectional"
)
```

### Stack
A **Stack** is a collection of multiple Blocks, useful for:
- **Block-wise training**: Train groups of blocks together
- **Memory management**: Organize blocks into logical groups
- **Fusion training**: Train multiple blocks with frozen/trainable options

```python
from model.architecture import Stack

# Create a stack with multiple blocks
blocks = [Block(...) for _ in range(4)]
stack = Stack(blocks, stack_id=0)
```

### Rack
A **Rack** is the final model containing:
- **Input embeddings**
- **Multiple Stacks** of Blocks
- **Output layer** (language model head)
- **Positional encoding** (RoPE if enabled)

```python
from model.architecture import Rack

# Create the complete model
stacks = [Stack(...), Stack(...)]
rack = Rack(
    stacks=stacks,
    vocab_size=50000,
    d_model=512,
    tie_embeddings=True
)
```

## 🎯 Training Modes

The new architecture supports three training modes:

### 1. Block-wise Training
Train each block independently:
```python
from training.architecture_trainer import ArchitectureTrainer

config.training.training_architecture = "blockwise"
trainer = ArchitectureTrainer(config)
results = trainer.train_architecture(rack, dataloader)
```

### 2. Stack-wise Training
Train each stack independently:
```python
config.training.training_architecture = "stackwise"
trainer = ArchitectureTrainer(config)
results = trainer.train_architecture(rack, dataloader)
```

### 3. Rack-wise Training
Train the entire model together:
```python
config.training.training_architecture = "rackwise"
trainer = ArchitectureTrainer(config)
results = trainer.train_architecture(rack, dataloader)
```

## ⚙️ Configuration

The configuration system has been updated to support the new architecture:

```yaml
model:
  # Architecture configuration
  architecture:
    n_stacks: 2        # Number of stacks
    blocks_per_stack: 4  # Number of blocks per stack

training:
  # Training architecture modes
  training_architecture: "blockwise"  # blockwise | stackwise | rackwise
```

## 🚀 Quick Start

### 1. Create a Rack from Configuration
```python
from model.architecture import create_rack_from_config
from config.base import StackWiseConfig

# Load configuration
config = StackWiseConfig.from_yaml("config.yaml")

# Create rack
rack = create_rack_from_config(config.to_dict())
```

### 2. Train the Architecture
```python
from training.architecture_trainer import ArchitectureTrainer

# Create trainer
trainer = ArchitectureTrainer(config)

# Train the architecture
results = trainer.train_architecture(rack, dataloader)
```

### 3. Use the Trained Model
```python
# Forward pass
input_ids = torch.randint(0, 50000, (batch_size, seq_len))
logits = rack(input_ids)

# Get model information
print(f"Parameters: {rack.get_parameter_count():,}")
print(f"Stacks: {len(rack.stacks)}")
print(f"Total blocks: {sum(len(stack.blocks) for stack in rack.stacks)}")
```

## 📊 Benefits of the New Architecture

### 1. **Clearer Naming**
- **Block**: Standard transformer block (what was previously called "layer")
- **Stack**: Collection of blocks (logical grouping)
- **Rack**: Complete model (final assembly)

### 2. **Better Organization**
- Hierarchical structure makes the model easier to understand
- Clear separation between individual components and complete model
- Intuitive naming that matches physical hardware analogy

### 3. **Flexible Training**
- **Block-wise**: Train each block independently (layer-wise training)
- **Stack-wise**: Train groups of blocks together (block-wise training)
- **Rack-wise**: Train the entire model together (end-to-end training)

### 4. **Memory Efficiency**
- Stacks can be frozen/unfrozen independently
- Blocks can be trained individually to reduce memory usage
- Support for different training strategies per stack

## 🔄 Migration from Old Architecture

The new architecture is backward compatible:

### Old Naming → New Naming
- `MLGKALayer` → `Block` (with attention + FFN + layer norm + residual)
- `model_layers` → `rack.stacks[].blocks`
- `layerwise_training` → `mode` (with values "layerwise", "blockwise", "fused")
- `blockwise_training` → `stackwise_training`

### Configuration Updates
```yaml
# Old configuration
training:
  mode: "layerwise"
  block_size: 4

# New configuration  
training:
  training_architecture: "blockwise"  # or "stackwise" or "rackwise"
  architecture:
    # Use architecture.n_stacks and architecture.blocks_per_stack instead
    n_stacks: 2
    blocks_per_stack: 4
```

## 📚 Examples

See the following examples for detailed usage:

- `examples/architecture_example.py` - Basic architecture usage
- `examples/new_architecture_training.py` - Training examples
- `examples/gpt2_fusion/` - GPT-2 fusion training with new architecture

## 🎯 Future Enhancements

The new architecture enables several future enhancements:

1. **Stack-specific Training**: Different training strategies per stack
2. **Dynamic Stacking**: Add/remove stacks during training
3. **Stack Fusion**: Merge multiple stacks into one
4. **Stack Quantization**: Different quantization per stack
5. **Stack Caching**: Efficient caching per stack

## 📝 Summary

The new Block/Stack/Rack architecture provides:

- ✅ **Clearer naming** that matches physical hardware analogy
- ✅ **Better organization** with hierarchical structure
- ✅ **Flexible training** modes for different scenarios
- ✅ **Backward compatibility** with existing code
- ✅ **Future extensibility** for advanced features

This architecture makes StackWise more intuitive to use while maintaining all the powerful features of the original system.
