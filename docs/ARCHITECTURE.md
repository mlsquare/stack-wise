# 🎯 Architecture: n_stacks + blocks_per_stack

This document explains the simplified architecture configuration that removes `n_blocks` and uses only `n_stacks` and `blocks_per_stack` for cleaner, more intuitive configuration.

## 📋 Key Changes

### ✅ **Removed Complexity**
- **No more `n_blocks`**: Eliminated redundant parameter
- **Simplified configuration**: Just specify `n_stacks` and `blocks_per_stack`
- **Cleaner helper functions**: More intuitive API

### ✅ **Better Configuration**
```yaml
# Before (Complex)
model:
  n_blocks: 8
  architecture:
    n_blocks: 8
    n_stacks: 2
    blocks_per_stack: 4

# After (Simple)
model:
  architecture:
    n_stacks: 2
    blocks_per_stack: 4
```

## 🔧 New Configuration Structure

### **config.yaml**
```yaml
model:
  # Model dimensions
  d_model: 4096
  n_heads: 32
  n_kv_heads: 8
  d_ff: 14336
  
  # Architecture configuration (simplified)
  architecture:
    n_stacks: 2        # Number of stacks
    blocks_per_stack: 4  # Number of blocks per stack
    
  # DEPRECATED: Use architecture.n_stacks and architecture.blocks_per_stack instead
  # n_layers: 8  # DEPRECATED - use architecture configuration
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

### 1. **Simple Rack Creation**
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

### 2. **From Configuration**
```python
from model.architecture import create_rack_from_config
from config.base import StackWiseConfig

# Load configuration and create rack
config = StackWiseConfig.from_yaml("config.yaml")
rack = create_rack_from_config(config.to_dict())
```

### 3. **Heterogeneous Architecture**
```python
from model.architecture import create_block_spec, create_rack_from_specs

# Create different block specifications
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

## 🎯 Benefits

### 1. **Simplified Configuration**
- **No redundancy**: Only specify what you need
- **Clear intent**: `n_stacks` and `blocks_per_stack` are self-explanatory
- **Less confusion**: No need to calculate total blocks

### 2. **Intuitive API**
```python
# Before: Confusing
n_blocks = 8
n_stacks = 2
blocks_per_stack = 4  # Wait, 2 * 4 = 8, but what if they don't match?

# After: Clear
n_stacks = 2
blocks_per_stack = 4  # Obviously 2 stacks of 4 blocks each
```

### 3. **Better Helper Functions**
```python
# Simple cases
rack = create_simple_rack(n_stacks=2, blocks_per_stack=4, ...)

# Complex cases
rack = create_rack_from_specs(vocab_size, d_model, stack_specs)
```

### 4. **Training Scenarios**
```python
# Layer-wise training: Each block is its own stack
rack = create_simple_rack(n_stacks=8, blocks_per_stack=1, ...)

# Block-wise training: Groups of blocks
rack = create_simple_rack(n_stacks=2, blocks_per_stack=4, ...)

# Stack-wise training: All blocks in one stack
rack = create_simple_rack(n_stacks=1, blocks_per_stack=8, ...)
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

### 2. **Different Sizes**
```python
# Small model
small = create_simple_rack(n_stacks=1, blocks_per_stack=2, ...)

# Medium model  
medium = create_simple_rack(n_stacks=2, blocks_per_stack=3, ...)

# Large model
large = create_simple_rack(n_stacks=3, blocks_per_stack=4, ...)
```

### 3. **Training Configurations**
```python
# Layer-wise: 8 stacks, 1 block each
layerwise = create_simple_rack(n_stacks=8, blocks_per_stack=1, ...)

# Block-wise: 2 stacks, 4 blocks each
blockwise = create_simple_rack(n_stacks=2, blocks_per_stack=4, ...)

# Stack-wise: 1 stack, 8 blocks
stackwise = create_simple_rack(n_stacks=1, blocks_per_stack=8, ...)
```

## 🔄 Migration Guide

### From Old to New Configuration

1. **Update config.yaml**:
   ```yaml
   # Old
   n_blocks: 8
   architecture:
     n_blocks: 8
     n_stacks: 2
     blocks_per_stack: 4
   
   # New
   architecture:
     n_stacks: 2
     blocks_per_stack: 4
   ```

2. **Update code**:
   ```python
   # Old
   n_blocks = config.model.n_blocks
   
   # New
   n_stacks = config.model.architecture.n_stacks
   blocks_per_stack = config.model.architecture.blocks_per_stack
   ```

3. **Use helper functions**:
   ```python
   # Old: Manual calculation
   total_blocks = n_stacks * blocks_per_stack
   
   # New: Direct specification
   rack = create_simple_rack(n_stacks=2, blocks_per_stack=4, ...)
   ```

## 🎉 Summary

The simplified architecture provides:

- ✅ **Cleaner configuration**: No redundant `n_blocks` parameter
- ✅ **Intuitive API**: `n_stacks` and `blocks_per_stack` are self-explanatory
- ✅ **Better helper functions**: `create_simple_rack()` for common cases
- ✅ **Backward compatibility**: Deprecation warnings guide migration
- ✅ **Training flexibility**: Easy to configure different training scenarios

This makes StackWise much more intuitive to use while maintaining all the powerful features of the original system.
