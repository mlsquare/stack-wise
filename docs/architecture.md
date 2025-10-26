# 🏗️ StackWise Architecture Guide

This document provides a comprehensive guide to the StackWise architecture system, covering the Block-Stack-Rack paradigm, configuration management, and helper functions.

## 📋 Architecture Overview

StackWise uses a **hierarchical architecture** that provides unprecedented training flexibility:

```
Rack (Complete Model)
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

## ⚙️ Configuration

### **config.yaml**
```yaml
model:
  # Model dimensions
  d_model: 4096
  n_heads: 32
  n_kv_heads: 8
  d_ff: 14336
  
  # Architecture configuration
  architecture:
    n_stacks: 2        # Number of stacks
    blocks_per_stack: 4  # Number of blocks per stack
```

### **Configuration Classes**
```python
@dataclass
class ArchitectureConfig(BaseConfig):
    """Architecture configuration for stacks and blocks."""
    n_stacks: int = 2
    blocks_per_stack: int = 4

@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration."""
    # ... other fields ...
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
```

## 🛠️ Helper Functions

### 1. **Block Specification**
```python
from model.architecture import create_block_spec

# Create a block specification
block_spec = create_block_spec(
    d_model=512,
    d_ff=2048,
    n_heads=8,
    n_kv_heads=2,
    attention_type="gqa",
    attention_mode="bidirectional"
)
```

### 2. **Stack from Block Spec**
```python
from model.architecture import create_stack_from_spec

# Create a stack with identical blocks
stack = create_stack_from_spec(
    stack_id=0,
    n_blocks=4,
    block_spec=block_spec,
    freeze_blocks=False
)
```

### 3. **Simple Rack Creation**
```python
from model.architecture import create_simple_rack

# Create a simple rack: 2 stacks, 4 blocks per stack
rack = create_simple_rack(
    n_stacks=2,
    blocks_per_stack=4,
    d_model=512,
    d_ff=2048,
    n_heads=8,
    vocab_size=10000
)
```

### 4. **Rack from Stack Specs**
```python
from model.architecture import create_rack_from_specs

# Create stack specifications
stack_specs = [
    {
        "stack_id": 0,
        "n_blocks": 4,
        "block_spec": block_spec,
        "freeze_blocks": False
    },
    {
        "stack_id": 1,
        "n_blocks": 4,
        "block_spec": block_spec,
        "freeze_blocks": True  # Frozen stack
    }
]

# Create rack
rack = create_rack_from_specs(
    vocab_size=10000,
    d_model=512,
    stack_specs=stack_specs,
    tie_embeddings=True
)
```

### 5. **From Configuration**
```python
from model.architecture import create_rack_from_config
from config.base import StackWiseConfig

# Load configuration and create rack
config = StackWiseConfig.from_yaml("config.yaml")
rack = create_rack_from_config(config.to_dict())
```

## 🎯 Training Modes

The architecture supports three training modes:

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

## 📚 Examples

### 1. **Simple Architecture**
```python
# 2 stacks, 4 blocks per stack = 8 total blocks
rack = create_simple_rack(
    n_stacks=2,
    blocks_per_stack=4,
    d_model=512,
    d_ff=2048,
    n_heads=8,
    vocab_size=10000
)
```

### 2. **Heterogeneous Architecture**
```python
# Different block types for different stacks
encoder_spec = create_block_spec(
    d_model=256, attention_mode="bidirectional"
)
decoder_spec = create_block_spec(
    d_model=256, attention_mode="causal"
)

# Create stack specifications
stack_specs = [
    {"stack_id": 0, "n_blocks": 3, "block_spec": encoder_spec},
    {"stack_id": 1, "n_blocks": 3, "block_spec": decoder_spec}
]

# Create rack
rack = create_rack_from_specs(5000, 256, stack_specs)
```

### 3. **Different Model Sizes**
```python
# Small model
small = create_simple_rack(n_stacks=1, blocks_per_stack=2, ...)

# Medium model  
medium = create_simple_rack(n_stacks=2, blocks_per_stack=3, ...)

# Large model
large = create_simple_rack(n_stacks=3, blocks_per_stack=4, ...)
```

### 4. **Training Configurations**
```python
# Layer-wise: 8 stacks, 1 block each
layerwise = create_simple_rack(n_stacks=8, blocks_per_stack=1, ...)

# Block-wise: 2 stacks, 4 blocks each
blockwise = create_simple_rack(n_stacks=2, blocks_per_stack=4, ...)

# Stack-wise: 1 stack, 8 blocks
stackwise = create_simple_rack(n_stacks=1, blocks_per_stack=8, ...)
```

### 5. **Training-Ready Architecture**
```python
# Architecture optimized for training
block_spec = create_block_spec(
    d_model=512, d_ff=2048, n_heads=8,
    attention_type="gqa", attention_mode="bidirectional",
    dropout=0.1, freeze_up_proj=True
)

stack_specs = [
    {"stack_id": 0, "n_blocks": 4, "block_spec": block_spec, "freeze_blocks": False},
    {"stack_id": 1, "n_blocks": 4, "block_spec": block_spec, "freeze_blocks": False}
]
rack = create_rack_from_specs(10000, 512, stack_specs)
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

## 📊 Benefits

### 1. **Clearer Naming**
- **Block**: Standard transformer block
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

### 5. **Intuitive Configuration**
- **No redundancy**: Only specify what you need
- **Clear intent**: `n_stacks` and `blocks_per_stack` are self-explanatory
- **Less confusion**: No need to calculate total blocks

### 6. **Better Helper Functions**
```python
# Simple cases
rack = create_simple_rack(n_stacks=2, blocks_per_stack=4, ...)

# Complex cases
rack = create_rack_from_specs(vocab_size, d_model, stack_specs)
```

## 🎯 Future Enhancements

The architecture enables several future enhancements:

1. **Stack-specific Training**: Different training strategies per stack
2. **Dynamic Stacking**: Add/remove stacks during training
3. **Stack Fusion**: Merge multiple stacks into one
4. **Stack Quantization**: Different quantization per stack
5. **Stack Caching**: Efficient caching per stack

## 📝 Summary

The StackWise architecture provides:

- ✅ **Clearer naming** that matches physical hardware analogy
- ✅ **Better organization** with hierarchical structure
- ✅ **Flexible training** modes for different scenarios
- ✅ **Intuitive configuration** with simplified parameters
- ✅ **Helper functions** that reduce boilerplate code
- ✅ **Memory efficiency** for large model training
- ✅ **Future extensibility** for advanced features

This architecture makes StackWise more intuitive to use while maintaining all the powerful features needed for revolutionary layer-wise transformer training.