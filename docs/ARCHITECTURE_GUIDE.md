# 🚀 Architecture: Helper Functions and Configuration Updates

This document explains the improvements made to the StackWise architecture system, focusing on better configuration management and intuitive helper functions.

## 📋 Key Improvements

### 1. **Configuration Updates**
- ✅ **Simplified architecture**: Use `n_stacks` and `blocks_per_stack` instead of `n_blocks`
- ✅ **Deprecation handling**: Backward compatibility with warnings
- ✅ **Clearer structure**: Better organization of architecture parameters

### 2. **Helper Functions**
- ✅ **`create_block_spec()`**: Define block specifications once
- ✅ **`create_stack_from_spec()`**: Create stacks from block specs
- ✅ **`create_rack_from_specs()`**: Create racks from stack specs
- ✅ **`create_rack_from_config()`**: Create racks from configuration

## 🔧 Configuration Changes

### Before (Old Configuration)
```yaml
model:
  # ... other parameters
```

### After (New Configuration)
```yaml
model:
  architecture:
    n_stacks: 2        # Number of stacks
    blocks_per_stack: 4  # Number of blocks per stack
  # DEPRECATED: Use architecture configuration instead
  
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

### 3. **Rack from Stack Specs**
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

## 🎯 Benefits

### 1. **Intuitive Design**
- **Block specs**: Define once, use many times
- **Stack specs**: Clear specification of stack properties
- **Rack specs**: Easy composition of complete models

### 2. **Reduced Boilerplate**
```python
# Before: Manual block creation
blocks = []
for i in range(4):
    block = Block(
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_kv_heads=2,
        attention_type="gqa",
        attention_mode="bidirectional"
    )
    blocks.append(block)
stack = Stack(blocks, stack_id=0)

# After: Helper function
block_spec = create_block_spec(
    d_model=512, d_ff=2048, n_heads=8, n_kv_heads=2,
    attention_type="gqa", attention_mode="bidirectional"
)
stack = create_stack_from_spec(0, 4, block_spec)
```

### 3. **Better Configuration Management**
- **Deprecation warnings**: Clear migration path
- **Backward compatibility**: Existing configs still work
- **Type safety**: Better validation and error messages

### 4. **Flexible Architecture Creation**
```python
# Different block types for different stacks
encoder_spec = create_block_spec(
    d_model=512, attention_mode="bidirectional"
)
decoder_spec = create_block_spec(
    d_model=512, attention_mode="causal"
)

# Create heterogeneous architecture
stack_specs = [
    {"stack_id": 0, "n_blocks": 3, "block_spec": encoder_spec},
    {"stack_id": 1, "n_blocks": 3, "block_spec": decoder_spec}
]
rack = create_rack_from_specs(10000, 512, stack_specs)
```

## 📚 Examples

### 1. **Simple Architecture**
```python
# Create a simple 2-stack architecture
block_spec = create_block_spec(d_model=256, d_ff=1024, n_heads=4)
stack_specs = [
    {"stack_id": 0, "n_blocks": 3, "block_spec": block_spec},
    {"stack_id": 1, "n_blocks": 3, "block_spec": block_spec}
]
rack = create_rack_from_specs(5000, 256, stack_specs)
```

### 2. **Heterogeneous Architecture**
```python
# Different block types for different purposes
small_spec = create_block_spec(d_model=256, n_heads=4, attention_type="mha")
large_spec = create_block_spec(d_model=512, n_heads=8, attention_type="gqa")

stack_specs = [
    {"stack_id": 0, "n_blocks": 2, "block_spec": small_spec},
    {"stack_id": 1, "n_blocks": 4, "block_spec": large_spec}
]
rack = create_rack_from_specs(10000, 256, stack_specs)
```

### 3. **Training-Ready Architecture**
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

## 🔄 Migration Guide

### From Old to New Configuration

1. **Update config.yaml**:
   ```yaml
   # Old
   
   # New
   n_blocks: 8
   ```

2. **Update code**:
   ```python
   # Old
   
   # New
   n_blocks = config.model.n_blocks
   ```

3. **Use helper functions**:
   ```python
   # Old: Manual creation
   blocks = [Block(...) for _ in range(n_blocks)]
   stack = Stack(blocks, stack_id=0)
   
   # New: Helper functions
   block_spec = create_block_spec(...)
   stack = create_stack_from_spec(0, n_blocks, block_spec)
   ```

## 🎉 Summary

The improved architecture provides:

- ✅ **Better naming**: `n_blocks` instead of `n_layers`
- ✅ **Helper functions**: Reduce boilerplate code
- ✅ **Backward compatibility**: Existing code still works
- ✅ **Deprecation warnings**: Clear migration path
- ✅ **Flexible composition**: Easy to create complex architectures
- ✅ **Type safety**: Better validation and error messages

This makes StackWise more intuitive to use while maintaining all the powerful features of the original system.
